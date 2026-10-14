#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace warpmoe {
struct DeviceProblem {
 int batch,hidden,intermediate,experts,topk,group_size,gate_tiles,down_tiles;
 const std::uint8_t* packed;
 const float* scales;
 const float* x;
 const std::int32_t* route;
 const float* scores;
 float* intermediate_buffer;
 float* output;
 int* ready;
 int* ticket;
};
void launch_two_kernel(DeviceProblem p,cudaStream_t stream);
void launch_persistent(DeviceProblem p,int worker_blocks,cudaStream_t stream);
} // namespace warpmoe
