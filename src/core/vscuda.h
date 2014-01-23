#ifndef VSCUDA
#define VSCUDA 

#include "VSCudaHelper.h"
#include <stdint.h>

// CUDA Memory Management
void vscuda_free(void *data);
void vscuda_malloc3D(cudaPitchedPtr *ptr, int width, int height, int bytesPerSample);
void vscuda_malloc(uint8_t **ptr, size_t size);
void vscuda_memcpyAsync(void* dst, const void* src, size_t size, cudaMemcpyKind kind, cudaStream_t stream);
void vscuda_memcpy2DAsync(void* dst, size_t dstStride, const void* src, size_t srcStride, size_t widthInBytes, size_t height, cudaMemcpyKind kind, cudaStream_t stream);

// CUDA Device Management
void vscuda_deviceSynchronize();

// CUDA Stream Management
void vscuda_createStreams(VSCUDAStream *vsStreams, int numberOfStreams);
void vscuda_destroyStreams(VSCUDAStream *vsStreams, int numberOfStreams);

#endif