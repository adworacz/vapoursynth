/**
	Contains basic cuda interaction functions.
	Mostly to add a layer of seperation for C++11
	compatibility between Vapoursynth and CUDA.
*/

#include "vscuda.h"
#include "vslog.h"
#include <assert.h>

// CUDA Memory Management
void vscuda_free(void *data) {
	if (data) {
		CHECKCUDA(cudaFree(data));	
	} else {
		vsWarning("Called free on an empty pointer.");
	}
}

void vscuda_malloc3D(cudaPitchedPtr *ptr, int width, int height, int bytesPerSample) {
	CHECKCUDA(cudaMalloc3D(ptr, make_cudaExtent(width * bytesPerSample, height, 1)));
	assert(ptr->ptr);
    if (!(ptr->ptr))
        vsFatal("Failed to allocate memory for plane on the GPU. Out of memory.");
}

void vscuda_malloc(uint8_t **ptr, size_t size) {
	CHECKCUDA(cudaMalloc(ptr, size));
	assert(*ptr);
	if (!(*ptr))
		vsFatal("Failed to allocate memory on the GPU. Out of memory");
}

void vscuda_memcpyAsync(void* dst, const void* src, size_t size, cudaMemcpyKind kind, cudaStream_t stream) {
	CHECKCUDA(cudaMemcpyAsync(dst, src, size, kind, stream));
}

void vscuda_memcpy2DAsync(void* dst, size_t dstStride, const void* src, size_t srcStride, size_t widthInBytes, size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
	CHECKCUDA(cudaMemcpy2DAsync(dst, dstStride, src, srcStride, widthInBytes, height, kind, stream));
}


// CUDA Device Management
void vscuda_deviceSynchronize() {
	CHECKCUDA(cudaDeviceSynchronize());
}


// CUDA Stream Management
void vscuda_createStream(VSCUDAStream *vsStream) {
	CHECKCUDA(cudaStreamCreate(&(vsStream->stream)));
}

void vscuda_destroyStream(VSCUDAStream vsStream) {
	CHECKCUDA(cudaStreamDestroy(vsStream.stream));
}