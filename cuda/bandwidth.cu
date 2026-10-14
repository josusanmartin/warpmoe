#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#define CUDA_OK(expr) do { auto e=(expr); if(e!=cudaSuccess)throw std::runtime_error(std::string(#expr)+": "+cudaGetErrorString(e)); } while(0)
__global__ void read_stream(const uint4* __restrict__ input,std::size_t elements,unsigned int* __restrict__ output){
 unsigned int checksum=0;
 for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<elements;i+=std::size_t(blockDim.x)*gridDim.x){
 uint4 v=input[i];checksum^=v.x+3u*v.y+5u*v.z+7u*v.w;
 }
 __shared__ unsigned int scratch[256];scratch[threadIdx.x]=checksum;__syncthreads();
 for(int step=128;step;step>>=1){
 if(threadIdx.x<step)scratch[threadIdx.x]^=scratch[threadIdx.x+step];
 __syncthreads();
 }
 if(threadIdx.x==0)output[blockIdx.x]=scratch[0];
}
int main(int argc,char** argv){
 try{
 int mib=512,iters=30;
 for(int i=1;i<argc;++i){
 std::string name=argv[i];
 if(name=="--help"){std::cout<<"warpmoe_bandwidth [--mib 512] [--iters 30]\n";return 0;}
 if(i+1>=argc)throw std::invalid_argument("missing value");
 if(name=="--mib")mib=std::stoi(argv[++i]);
 else if(name=="--iters")iters=std::stoi(argv[++i]);
 else throw std::invalid_argument("unknown flag "+name);
 }
 if(mib<16||iters<=0)throw std::invalid_argument("mib must be >=16, iters positive");
 cudaDeviceProp prop{};CUDA_OK(cudaGetDeviceProperties(&prop,0));
 int blocks=prop.multiProcessorCount*8;
 std::size_t bytes=std::size_t(mib)*1024*1024;
 std::size_t elements=bytes/sizeof(uint4);
 uint4* input=nullptr;unsigned int* output=nullptr;
 CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&input),bytes));
 CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&output),blocks*sizeof(unsigned int)));
 CUDA_OK(cudaMemset(input,0x5a,bytes));
 cudaEvent_t start{},stop{};CUDA_OK(cudaEventCreate(&start));CUDA_OK(cudaEventCreate(&stop));
 for(int i=0;i<10;++i)read_stream<<<blocks,256>>>(input,elements,output);
 CUDA_OK(cudaDeviceSynchronize());
 std::vector<float> us;
 for(int i=0;i<iters;++i){
 CUDA_OK(cudaEventRecord(start));
 read_stream<<<blocks,256>>>(input,elements,output);
 CUDA_OK(cudaEventRecord(stop));CUDA_OK(cudaEventSynchronize(stop));
 float ms=0;CUDA_OK(cudaEventElapsedTime(&ms,start,stop));
 us.push_back(ms*1000.0f);
 }
 CUDA_OK(cudaGetLastError());std::sort(us.begin(),us.end());
 float median=us[us.size()/2];
 double gbps=double(bytes)/(median*1e-6)/1e9;
 std::cout<<"GPU "<<prop.name<<"\nRead-only streaming; input "<<mib<<" MiB\n"
 <<"Median "<<median<<" us; estimated requested-load BW "<<gbps<<" GB/s\n"
 <<"Check ncu dram__bytes for actual DRAM traffic.\n"
 <<"mib,median_us,requested_gb_s\n"<<mib<<","<<median<<","<<gbps<<"\n";
 CUDA_OK(cudaEventDestroy(start));CUDA_OK(cudaEventDestroy(stop));
 CUDA_OK(cudaFree(input));CUDA_OK(cudaFree(output));return 0;
 }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}
}
