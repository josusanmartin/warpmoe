#include "warpmoe/cuda_iface.hpp"
#include "warpmoe/format.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t err,const char* expr,const char* file,int line){
 if(err!=cudaSuccess)throw std::runtime_error(std::string(file)+":"+std::to_string(line)+" "+expr+": "+cudaGetErrorString(err));
}
#define CHECK(call) check((call),#call,__FILE__,__LINE__)
template<class T> class DeviceBuffer {
 T* ptr_=nullptr;std::size_t count_=0;
 public:
 explicit DeviceBuffer(std::size_t n):count_(n){
 if(n)CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_),n*sizeof(T)));
 }
 ~DeviceBuffer(){if(ptr_)cudaFree(ptr_);}
 DeviceBuffer(const DeviceBuffer&)=delete;
 DeviceBuffer& operator=(const DeviceBuffer&)=delete;
 T* get()const{return ptr_;}
 std::size_t bytes()const{return count_*sizeof(T);}
 void upload(const std::vector<T>& v){
 if(v.size()!=count_)throw std::invalid_argument("upload size mismatch");
 CHECK(cudaMemcpy(ptr_,v.data(),bytes(),cudaMemcpyHostToDevice));
 }
 std::vector<T> download()const{
 std::vector<T> v(count_);
 CHECK(cudaMemcpy(v.data(),ptr_,bytes(),cudaMemcpyDeviceToHost));
 return v;
 }
};
class CapturedGraph {
 cudaGraph_t graph_=nullptr;cudaGraphExec_t exec_=nullptr;
 public:
 CapturedGraph(cudaStream_t s,const std::function<void()>& record){
 CHECK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
 record();CHECK(cudaStreamEndCapture(s,&graph_));
 CHECK(cudaGraphInstantiate(&exec_,graph_,0));
 }
 ~CapturedGraph(){if(exec_)cudaGraphExecDestroy(exec_);if(graph_)cudaGraphDestroy(graph_);}
 void launch(cudaStream_t s)const{CHECK(cudaGraphLaunch(exec_,s));}
};
class GpuClock {
 cudaEvent_t start_{},stop_{};
 public:
 GpuClock(){CHECK(cudaEventCreate(&start_));CHECK(cudaEventCreate(&stop_));}
 ~GpuClock(){cudaEventDestroy(start_);cudaEventDestroy(stop_);}
 float time_us(const CapturedGraph& graph,cudaStream_t stream){
 CHECK(cudaEventRecord(start_,stream));graph.launch(stream);
 CHECK(cudaEventRecord(stop_,stream));CHECK(cudaEventSynchronize(stop_));
 float ms=0;CHECK(cudaEventElapsedTime(&ms,start_,stop_));return ms*1000;
 }
};
struct Timings{double median,p95,min;};
Timings measure(const CapturedGraph& graph,cudaStream_t stream,int warmup,int iters){
 GpuClock clock;
 for(int i=0;i<warmup;++i)graph.launch(stream);
 CHECK(cudaStreamSynchronize(stream));
 std::vector<float> times;times.reserve(iters);
 for(int i=0;i<iters;++i)times.push_back(clock.time_us(graph,stream));
 std::sort(times.begin(),times.end());
 return {times[times.size()/2],times[std::min(times.size()-1,(times.size()*95+99)/100-1)],times.front()};
}
void validate(const std::vector<float>& a,const std::vector<float>& b,const char* label,
 float atol=0.004f,float rtol=0.004f){
 if(a.size()!=b.size())throw std::runtime_error("length mismatch");
 float max_abs=0,max_rel=0;std::size_t fail=0;
 for(std::size_t i=0;i<a.size();++i){
 float diff=std::abs(a[i]-b[i]);
 max_abs=std::max(max_abs,diff);
 max_rel=std::max(max_rel,diff/std::max(std::abs(b[i]),1e-8f));
 if(!std::isfinite(a[i])||diff>atol+rtol*std::abs(b[i]))++fail;
 }
 std::cout<<"validate "<<label<<": max_abs="<<max_abs<<
 " max_relative="<<max_rel<<" failures="<<fail<<"\n";
 if(fail)throw std::runtime_error(std::string(label)+" correctness failed");
}
int positive(const char* v){
 int x=std::stoi(v);if(x<=0)throw std::invalid_argument("expected positive integer");return x;
}
}
int main(int argc,char** argv){
 try{
 std::string name="tiny";int batch=1,iters=30,warmup=8,blocks=0;bool check_cpu=false;
 for(int i=1;i<argc;++i){
 std::string flag=argv[i];
 if(flag=="--help"){
 std::cout<<"warpmoe_bench [--shape tiny|qwen] [--batch N] [--iters N] [--warmup N] [--blocks N] [--check-cpu]\n";
 return 0;
 }
 if(flag=="--check-cpu"){check_cpu=true;continue;}
 if(i+1>=argc)throw std::invalid_argument("missing value for "+flag);
 if(flag=="--shape")name=argv[++i];
 else if(flag=="--batch")batch=positive(argv[++i]);
 else if(flag=="--iters")iters=positive(argv[++i]);
 else if(flag=="--warmup")warmup=positive(argv[++i]);
 else if(flag=="--blocks")blocks=positive(argv[++i]);
 else throw std::invalid_argument("unknown flag "+flag);
 }
 warpmoe::Shape s;
 if(name=="tiny")s={batch,32,32,8,2,16};
 else if(name=="qwen")s={batch,2048,768,128,8,128};
 else throw std::invalid_argument("shape must be tiny or qwen");
 s.validate();if(name=="tiny")check_cpu=true;
 CHECK(cudaSetDevice(0));
 cudaDeviceProp prop{};CHECK(cudaGetDeviceProperties(&prop,0));
 if(prop.major<8)throw std::runtime_error("Ampere sm_80+ required");
 if(!blocks)blocks=prop.multiProcessorCount*2;
 std::cout<<"GPU: "<<prop.name<<", SM count "<<prop.multiProcessorCount<<
 ", workers "<<blocks<<"\n";
 std::cout<<"Generating deterministic synthetic INT4 weights and inputs...\n";
 auto w=warpmoe::make_weights(s),in=warpmoe::make_inputs(s);
 DeviceBuffer<std::uint8_t> packed(w.packed.size());packed.upload(w.packed);
 DeviceBuffer<float> scales(w.scales.size());scales.upload(w.scales);
 DeviceBuffer<float> x(in.x.size());x.upload(in.x);
 DeviceBuffer<std::int32_t> route(in.route.size());route.upload(in.route);
 DeviceBuffer<float> scores(in.scores.size());scores.upload(in.scores);
 DeviceBuffer<float> intermediate(s.pairs()*s.intermediate);
 DeviceBuffer<float> output(std::size_t(s.batch)*s.hidden);
 DeviceBuffer<int> ready(s.pairs()),ticket(1);
 warpmoe::TicketPlan plan;
 warpmoe::DeviceProblem p{s.batch,s.hidden,s.intermediate,s.experts,s.topk,s.group_size,
 plan.gate_tiles(s),plan.down_tiles(s),packed.get(),scales.get(),x.get(),
 route.get(),scores.get(),intermediate.get(),output.get(),ready.get(),ticket.get()};
 cudaStream_t stream=nullptr;CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 CapturedGraph baseline(stream,[&]{
 CHECK(cudaMemsetAsync(output.get(),0,output.bytes(),stream));
 warpmoe::launch_two_kernel(p,stream);
 });
 CapturedGraph persistent(stream,[&]{
 CHECK(cudaMemsetAsync(output.get(),0,output.bytes(),stream));
 CHECK(cudaMemsetAsync(ready.get(),0,ready.bytes(),stream));
 CHECK(cudaMemsetAsync(ticket.get(),0,ticket.bytes(),stream));
 warpmoe::launch_persistent(p,blocks,stream);
 });
 CHECK(cudaGetLastError());
 baseline.launch(stream);CHECK(cudaStreamSynchronize(stream));
 auto a=output.download();
 persistent.launch(stream);CHECK(cudaStreamSynchronize(stream));
 auto b=output.download();
 validate(b,a,"persistent vs graph-two-kernel");
 if(check_cpu){
 auto oracle=warpmoe::reference_forward(w,in);
 validate(a,oracle,"two-kernel vs CPU");
 validate(b,oracle,"persistent vs CPU");
 }
 auto ta=measure(baseline,stream,warmup,iters);
 auto tb=measure(persistent,stream,warmup,iters);
 auto bytes=warpmoe::unique_expert_weight_bytes(s,in.route);
 std::cout<<"Minimum distinct-expert packed weight bytes: "<<bytes<<"\n";
 std::cout<<"Baseline graph median="<<ta.median<<" us, p95="<<ta.p95<<" us\n";
 std::cout<<"Persistent median="<<tb.median<<" us, p95="<<tb.p95<<" us\n";
 std::cout<<"This is NOT Marlin/vLLM: both paths use naive GEMV.\n";
 std::cout<<"path,shape,batch,median_us,p95_us,min_us,distinct_weight_bytes,worker_blocks\n";
 std::printf("graph_two_kernel,%s,%d,%.3f,%.3f,%.3f,%zu,%d\n",name.c_str(),
 batch,ta.median,ta.p95,ta.min,bytes,blocks);
 std::printf("persistent,%s,%d,%.3f,%.3f,%.3f,%zu,%d\n",name.c_str(),
 batch,tb.median,tb.p95,tb.min,bytes,blocks);
 CHECK(cudaStreamDestroy(stream));return 0;
 }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}
}
