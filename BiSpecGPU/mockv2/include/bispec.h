#ifndef _BISPEC_H_
#define _BISPEC_H_

#include <cuda.h>
#include <vector_types.h>

__constant__ int d_Ngrid[4];
__constant__ int d_N[1];
__constant__ float d_binWidth[1];
__constant__ int d_numBins[1];
__constant__ float d_klim[2];

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
    if (blockIdx.x >= blockIdx.y) {
//         __shared__ double Bk_local[4096];
        int tid_x = threadIdx.x + blockDim.x*blockIdx.x;
        int tid_y = threadIdx.y + blockDim.y*blockIdx.y;
        if (tid_x >= tid_y && tid_x < d_N[0] && tid_y < d_N[0]) {
            float4 dk_1 = dk3d[k[tid_y].w];
            float4 dk_2 = dk3d[k[tid_x].w];
            int4 k_3 = {-k[tid_x].x - k[tid_y].x, -k[tid_x].y - k[tid_y].y, -k[tid_x].z - k[tid_y].z, 0};
            int i3 = k_3.x + d_Ngrid[0]/2;
            int j3 = k_3.y + d_Ngrid[1]/2;
            int k3 = k_3.z + d_Ngrid[2]/2;
            if (i3 >= 0 && j3 >= 0 && k3 >= 0 && i3 < d_Ngrid[0] && j3 < d_Ngrid[1] && k3 < d_Ngrid[2]) {
                k_3.w = k3 + d_Ngrid[2]*(j3 + d_Ngrid[1]*i3);
                float4 dk_3 = dk3d[k_3.w];
                if (dk_3.z < d_klim[1] && dk_3.z >= d_klim[0]) {
                    float grid_cor = dk_1.w*dk_2.w*dk_3.w;
                    double val = (dk_1.x*dk_2.x*dk_3.x - dk_1.x*dk_2.y*dk_3.y - dk_1.y*dk_2.x*dk_3.y -
                                  dk_1.y*dk_2.y*dk_3.x)*grid_cor;
                    int ik1 = (dk_1.z - d_klim[0])/d_binWidth[0];
                    int ik2 = (dk_2.z - d_klim[0])/d_binWidth[0];
                    int ik3 = (dk_3.z - d_klim[0])/d_binWidth[0];
                    int bin = ik3 + d_numBins[0]*(ik2 + d_numBins[0]*ik1);
                    atomicAdd(&Bk[bin], val);
                    atomicAdd(&N_tri[bin], 1);
                }
            }
        }
    }
}

__global__ void normBk(unsigned int *N_tri, double *Bk, float norm, int totBins) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    if (tid < totBins && N_tri[tid] > 0) {
        Bk[tid] /= (norm*N_tri[tid]);
    }
}

#endif
