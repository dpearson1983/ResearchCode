#ifndef _BISPEC_H_
#define _BISPEC_H_

#include <vector_types.h>

__constant__ int4 d_Ngrid[1];
__constant__ int d_N[1];
__constant__ float d_binWidth[1];
__constant__ int d_numBins[1];
__constant__ float2 d_klim[2];

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ > 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

__global__ void calcBk(float4 *dk3d, int4 *k, unsigned int *N_tri, double *Bk) {
    int tid_x = threadIdx.x + blockDim.x*blockIdx.x;
    int tid_y = threadIdx.y + blockDim.y*blockIdx.y;
    if (blockIdx.x >= blockIdx.y && tid_x >= tid_y && tid_x < d_N[0] && tid_y < d_N[0]) {
        int4 k_3 = {-k[tid_x].x - k[tid_y].x, -k[tid_x].y - k[tid_y].y, -k[tid_x].z - k[tid_y].z, 0};
        int i3 = k_3.x + d_Ngrid[0].x/2;
        int j3 = k_3.y + d_Ngrid[0].y/2;
        int k3 = k_3.z + d_Ngrid[0].z/2;
    }
}

#endif

 int4 N_grid,
                       int N, float binWidth, int numBins, int totBins, float2 k_lim
