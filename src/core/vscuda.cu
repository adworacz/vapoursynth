/**
	Contains basic cuda interaction functions.
	Mostly to add a layer of seperation for C++11
	compatibility between Vapoursynth and CUDA.
*/

#include "vscuda.h"

void vscuda_deviceSynchronize() {
	CHECKCUDA(cudaDeviceSynchronize());
}

void vscuda_createStream(VSCUDAStream *vsStream) {
	CHECKCUDA(cudaStreamCreate(&(vsStream->stream)));
}

void vscuda_destroyStream(VSCUDAStream vsStream) {
	CHECKCUDA(cudaStreamDestroy(vsStream.stream));
}