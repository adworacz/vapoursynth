#ifndef VSCUDA
#define VSCUDA 

#include "VSCudaHelper.h"

// CUDA Device Management
void vscuda_deviceSynchronize();

// CUDA Stream Management
void vscuda_createStream(VSCUDAStream *vsStream);
void vscuda_destroyStream(VSCUDAStream vsStream);

#endif