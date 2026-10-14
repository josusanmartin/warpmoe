#include "warpmoe/cuda_iface.hpp"
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
namespace warpmoe {
namespace {
constexpr int kWarps=4,kThreads=32*kWarps,kGateRows=64,kDownRows=128;
__device__ __forceinline__ float dequant_weight(const DeviceProblem& p,int expert,int projection,int row,int col){
 const std::size_t elems=std::size_t(p.hidden)*p.intermediate;
 const std::size_t nibble=std::size_t(expert)*(3*elems)+std::size_t(projection)*elems+
 std::size_t(row)*(projection==2?p.intermediate:p.hidden)+col;
 std::uint8_t byte=p.packed[nibble>>1];
 int q=int((nibble&1)?(byte>>4):(byte&0xf))-8;
 int gate_scales=p.intermediate*(p.hidden/p.group_size);
 int cols=projection==2?p.intermediate:p.hidden;
 int offset=projection==0?0:projection==1?gate_scales:2*gate_scales;
 std::size_t per=2ULL*gate_scales+std::size_t(p.hidden)*(p.intermediate/p.group_size);
 std::size_t index=std::size_t(expert)*per+offset+
 std::size_t(row)*(cols/p.group_size)+col/p.group_size;
 return float(q)*p.scales[index];
}
__device__ __forceinline__ float warp_sum(float value){
 for(int offset=16;offset>0;offset>>=1)value+=__shfl_down_sync(0xffffffff,value,offset);
 return value;
}
__device__ void gate_up_tile(const DeviceProblem& p,int pair,int tile){
 int warp=int(threadIdx.x)/32,lane=int(threadIdx.x)%32;
 int b=pair/p.topk,e=p.route[pair];
 int end=min((tile+1)*kGateRows,p.intermediate);
 for(int row=tile*kGateRows+warp;row<end;row+=kWarps){
 float gate=0,up=0;
 for(int c=lane;c<p.hidden;c+=32){
 float x=p.x[std::size_t(b)*p.hidden+c];
 gate=fmaf(x,dequant_weight(p,e,0,row,c),gate);
 up=fmaf(x,dequant_weight(p,e,1,row,c),up);
 }
 gate=warp_sum(gate);up=warp_sum(up);
 if(lane==0){
 float activated=gate/(1.0f+expf(-gate));
 p.intermediate_buffer[std::size_t(pair)*p.intermediate+row]=activated*up;
 }
 }
}
__device__ void down_tile(const DeviceProblem& p,int pair,int tile){
 int warp=int(threadIdx.x)/32,lane=int(threadIdx.x)%32;
 int b=pair/p.topk,e=p.route[pair];float scale=p.scores[pair];
 int end=min((tile+1)*kDownRows,p.hidden);
 for(int row=tile*kDownRows+warp;row<end;row+=kWarps){
 float sum=0;
 for(int c=lane;c<p.intermediate;c+=32){
 float a=p.intermediate_buffer[std::size_t(pair)*p.intermediate+c];
 sum=fmaf(a,dequant_weight(p,e,2,row,c),sum);
 }
 sum=warp_sum(sum);
 if(lane==0)atomicAdd(&p.output[std::size_t(b)*p.hidden+row],sum*scale);
 }
}
__global__ void gate_up_kernel(DeviceProblem p){
 int task=int(blockIdx.x);gate_up_tile(p,task/p.gate_tiles,task%p.gate_tiles);
}
__global__ void down_kernel(DeviceProblem p){
 int task=int(blockIdx.x);down_tile(p,task/p.down_tiles,task%p.down_tiles);
}
__global__ void persistent_kernel(DeviceProblem p){
 __shared__ int ticket_shared;
 int pairs=p.batch*p.topk,gate_total=pairs*p.gate_tiles;
 int task_total=gate_total+pairs*p.down_tiles;
 while(true){
 if(threadIdx.x==0)ticket_shared=atomicAdd(p.ticket,1);
 __syncthreads();int ticket=ticket_shared;
 if(ticket>=task_total)break;
 if(ticket<gate_total){
 int pair=ticket/p.gate_tiles,tile=ticket%p.gate_tiles;
 gate_up_tile(p,pair,tile);
 __syncthreads();
 if(threadIdx.x==0){
 cuda::atomic_ref<int,cuda::thread_scope_device> done(p.ready[pair]);
 done.fetch_add(1,cuda::memory_order_release);
 }
 }else{
 int index=ticket-gate_total;
 int pair=index/p.down_tiles,tile=index%p.down_tiles;
 if(threadIdx.x==0){
 cuda::atomic_ref<int,cuda::thread_scope_device> done(p.ready[pair]);
 while(done.load(cuda::memory_order_acquire)<p.gate_tiles)__nanosleep(32);
 }
 __syncthreads();down_tile(p,pair,tile);
 }
 __syncthreads();
 }
}
}
void launch_two_kernel(DeviceProblem p,cudaStream_t stream){
 gate_up_kernel<<<p.batch*p.topk*p.gate_tiles,kThreads,0,stream>>>(p);
 down_kernel<<<p.batch*p.topk*p.down_tiles,kThreads,0,stream>>>(p);
}
void launch_persistent(DeviceProblem p,int worker_blocks,cudaStream_t stream){
 persistent_kernel<<<worker_blocks,kThreads,0,stream>>>(p);
}
} // namespace warpmoe
