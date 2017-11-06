#ifndef _BISPEC_H_
#define _BISPEC_H_

#include <cuda.h>
#include <vector_types.h>

__constant__ int d_Ngrid[4];
__constant__ int d_N;
__constant__ float d_binWidth;
__constant__ int d_numBins;
__constant__ float2 d_klim;

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

__global__ void calcBk(float4 *dk3d, int4 *k, double *Bk) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    int xShift = d_Ngrid[0]/2;
    int yShift = d_Ngrid[1]/2;
    int zShift = d_Ngrid[2]/2;
    
    __shared__ double Bk_local[4096];
    for (int i = threadIdx.x*4; i < threadIdx.x*4 + 4; ++i)
        Bk_local[i] = 0;
    __syncthreads();
    
    if (tid < d_N) {
        int4 k_1 = k[tid];
        float4 dk_1 = dk3d[k_1.w];
        int ik1 = (dk_1.z - d_klim.x)/d_binWidth;
        for (int i = tid; i < d_N; ++i) {
            int4 k_2 = k[i];
            float4 dk_2 = dk3d[k_2.w];
            int4 k_3 = {-k_1.x - k_2.x, -k_1.y - k_2.y, -k_1.z - k_2.z, 0};
            int i3, j3, k3;
            i3 = k_3.x + xShift;
            j3 = k_3.y + yShift;
            k3 = k_3.z + zShift;
            if (i3 >= 0 && j3 >= 0 && k3 >= 0 && i3 < d_Ngrid[0] && j3 < d_Ngrid[1] && k3 < d_Ngrid[2]) {
                k_3.w = k3 + d_Ngrid[2]*(j3 + d_Ngrid[1]*i3);
                float4 dk_3 = dk3d[k_3.w];
                if (dk_3.z < d_klim.y && dk_3.z >= d_klim.x) {
                    float grid_cor = dk_1.w*dk_2.w*dk_3.w;
                    double val = (dk_1.x*dk_2.x*dk_3.x - dk_1.x*dk_2.y*dk_3.y - dk_1.y*dk_2.x*dk_3.y - 
                                  dk_1.y*dk_2.y*dk_3.x)*grid_cor;
                    int ik2 = (dk_2.z - d_klim.x)/d_binWidth;
                    int ik3 = (dk_3.z - d_klim.x)/d_binWidth;
                    int bin = ik3 + d_numBins*(ik2 + d_numBins*ik1);
                    atomicAdd(&Bk_local[bin], val);
                }
            }
        }
        __syncthreads();
        
        for (int i = threadIdx.x*4; i < threadIdx.x*4 + 4; ++i) {
            atomicAdd(&Bk[i], Bk_local[i]);
        }
    }
}

__global__ void calcNtri(float4 *dk3d, int4 *k, unsigned int *N_tri) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    int xShift = d_Ngrid[0]/2;
    int yShift = d_Ngrid[1]/2;
    int zShift = d_Ngrid[2]/2;
    
    __shared__ unsigned int Ntri_local[4096];
    for (int i = threadIdx.x*4; i < threadIdx.x*4 + 4; ++i)
        Ntri_local[i] = 0;
    __syncthreads();
    
    if (tid < d_N) {
        int4 k_1 = k[tid];
        float4 dk_1 = dk3d[k_1.w];
        int ik1 = (dk_1.z - d_klim.x)/d_binWidth;
        for (int i = tid; i < d_N; ++i) {
            int4 k_2 = k[i];
            float4 dk_2 = dk3d[k_2.w];
            int4 k_3 = {-k_1.x - k_2.x, -k_1.y - k_2.y, -k_1.z - k_2.z, 0};
            int i3 = k_3.x + xShift;
            int j3 = k_3.y + yShift;
            int k3 = k_3.z + zShift;
            if (i3 >= 0 && j3 >= 0 && k3 >= 0 && i3 < d_Ngrid[0] && j3 < d_Ngrid[1] && k3 < d_Ngrid[2]) {
                k_3.w = k3 + d_Ngrid[2]*(j3 + d_Ngrid[1]*i3);
                float4 dk_3 = dk3d[k_3.w];
                if (dk_3.z < d_klim.y && dk_3.z >= d_klim.x) {
                    int ik2 = (dk_2.z - d_klim.x)/d_binWidth;
                    int ik3 = (dk_3.z - d_klim.x)/d_binWidth;
                    int bin = ik3 + d_numBins*(ik2 + d_numBins*ik1);
                    atomicAdd(&Ntri_local[bin], 1);
                }
            }
        }
        __syncthreads();

        for (int i = threadIdx.x*4; i < threadIdx.x*4 + 4; ++i) {
            atomicAdd(&N_tri[i], Ntri_local[i]);
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
