#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <fftw3.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <vector_types.h>
#include <harppi.h>
#include <galaxy.h>
#include <powerspec.h>
#include <tpods.h>
#include <constants.h>
#include "include/fileReader.h"

void fftfreq(std::vector<double> &k, int N, double L) {
    double dk = (2.0*pi)/L;
    for (int i = 0; i <= N/2; ++i)
        k[i] = i*dk;
    for (int i = N/2 + 1; i < N; ++i)
        k[i] = (i - N)*dk;
}

void myfreq(std::vector<double> &k, int N, double L) {
    double dk = (2.0*pi)/L;
    for (int i = 0; i < N; ++i)
        k[i] = (int(i - N/2))*dk;
}

double gridCorCIC(vec3<double> k, vec3<double> dr) {
    double sincx = sin(0.5*k.x*dr.x + 1E-17)/(0.5*k.x*dr.x + 1E-17);
    double sincy = sin(0.5*k.y*dr.y + 1E-17)/(0.5*k.y*dr.y + 1E-17);
    double sincz = sin(0.5*k.z*dr.z + 1E-17)/(0.5*k.z*dr.z + 1E-17);
    double prodsinc = sincx*sincy*sincz;
    
    return 1.0/(prodsinc*prodsinc);
}

int kMatch(double ks, std::vector<double> &kb, double L) {
    double dk = (2.0*pi)/L;
    int i = 0;
    bool found = false;
    int N = kb.size();
    int index1 = floor(ks/dk + 0.5);
    for (int j = 0; j < N; ++j) {
        int index2 = floor(kb[j]/dk + 0.5);
        if (index2 == index1) {
            i = j;
            found = true;
        }
    }
    
    if (found) return i;
    
    return -10000;
}

// Finds the bispectrum bin for a k_1, k_2, k_3 triplet
__device__ int getBkBin(double k1, double k2, double k3, double d_binWidth, int d_numBins, double kmin) {
/*    if (k1 > k2) {
        double temp = k1;
        k1 = k2;
        k2 = temp;
    }
    if (k1 > k3) {
        double temp = k1;
        k1 = k3;
        k3 = temp;
    }
    if (k2 > k3) {
        double temp = k2;
        k2 = k3;
        k3 = temp;
    }  */  
    int i = (k1 - kmin)/d_binWidth;
    int j = (k2 - kmin)/d_binWidth;
    int k = (k3 - kmin)/d_binWidth;
    int bin = k + d_numBins*(j + d_numBins*i);
    return bin;
}

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

// The function below will calculate the bispectrum utilizing the GPU. The variables in the function call
// are:
//      1.    dk3d: Pointer to the GPU memory location of the delta(k) cube. The x element is the real part
//                  the y element is the imaginary part, the z element is the magnitude of the k-vector,
//                  and the w element is the grid correction.
//      2.    kvec: Pointer to the GPU memory location of the k vectors whose magnitudes are between k_min
//                  (k_lim.x) and k_max (k_lim.y). These are stored as integers where each component is These
//                  integer multiple of the fundamental frequency in that coordinate direction. The w 
//                  element is the index of the location in the delta(k) cube.
//      3.   N_tri: Pointer to the GPU memory location to count the number of triangle in each bispectrum
//                  bin.
//      4.      Bk: Pointer to the GPU memory location to store the binned bispectrum
//      5.  N_grid: The grid dimensions of dk3d
//      6.       N: The number of k vectors stored in kvec
//      7.binWidth: The width of the bins for the bispectrum, the width is in terms of a single k
//      8. numBins: The number of bispectrum bins for a single k (i.e. k_1)
//      9. totBins: The total number of bins for the bispectrum, i.e. numBins^3
//     10.   k_lim: The minimum (x) and maximum (y) k values to be binned
//     11.      SN: The power spectrum shotnoise
//     12.    term: The extra term needed for the bispectrum shotnoise calculation
__global__ void calcBk(double4 *dk3d, int4 *kvec, unsigned int *N_tri, double *Bk, int4 N_grid,
                       int N, double binWidth, int numBins, int totBins, double2 k_lim,
                       double SN, double term, double mult) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    int xShift = N_grid.x/2;
    int yShift = N_grid.y/2;
    int zShift = N_grid.z/2;
    SN *= -1.0;
    
    if (tid < N) {
        int4 k_1 = kvec[tid];
        k_1.x *= -1;
        k_1.y *= -1;
        k_1.z *= -1;
        double4 dk_1 = dk3d[k_1.w];
//         double P_1 = (dk_1.x*dk_1.x + dk_1.y*dk_1.y - SN)*dk_1.w*dk_1.w;
//         double P_1 = dk_1.x*dk_1.x + dk_1.y*dk_1.y - SN;
        double P_1 = fma(dk_1.x, dk_1.x, fma(dk_1.y, dk_1.y, SN))*dk_1.w*dk_1.w;
        for (int i = 0; i < N; ++i) {
            int4 k_2 = kvec[i];
            double4 dk_2 = dk3d[k_2.w];
//             double P_2 = (dk_2.x*dk_2.x + dk_2.y*dk_2.y - SN)*dk_2.w*dk_2.w;
//             double P_2 = dk_2.x*dk_2.x + dk_2.y*dk_2.y - SN;
            int4 k_3 = {k_1.x - k_2.x, k_1.y - k_2.y, k_1.z - k_2.z, 0};
            int i3, j3, k3;
            i3 = k_3.x + xShift;
            j3 = k_3.y + yShift;
            k3 = k_3.z + zShift;
            if (i3 >= 0 && j3 >= 0 && k3 >= 0 && i3 < N_grid.x && j3 < N_grid.y && k3 < N_grid.z) {
                k_3.w = k3 + N_grid.z*(j3 + N_grid.y*i3);
                double4 dk_3 = dk3d[k_3.w];
                if (dk_3.z < k_lim.y && dk_3.z >= k_lim.x) {
//                     double P_3 = (dk_3.x*dk_3.x + dk_3.y*dk_3.y - SN)*dk_3.w*dk_3.w;
//                     double P_3 = dk_3.x*dk_3.x + dk_3.y*dk_3.y - SN;
                    double P_2 = fma(dk_2.x, dk_2.x, fma(dk_2.y, dk_2.y, SN))*dk_2.w*dk_2.w;
                    double P_3 = fma(dk_3.x, dk_3.x, fma(dk_3.y, dk_3.y, SN))*dk_3.w*dk_3.w;
                    double grid_cor = dk_1.w*dk_2.w*dk_3.w;
                    double val = (dk_1.x*dk_2.x*dk_3.x - dk_1.x*dk_2.y*dk_3.y - dk_1.y*dk_2.x*dk_3.y - dk_1.y*dk_2.y*dk_3.x);
                    val *= grid_cor;
                    val -= ((P_1 + P_2 + P_3)*mult + term);
                    int bin = getBkBin(dk_1.z, dk_2.z, dk_3.z, binWidth, numBins, k_lim.x);
                    atomicAdd(&Bk[bin], val);
                    atomicAdd(&N_tri[bin], 1);
                }
            }
        }
    }
}

// This function normalizes the bispectrum measurements
__global__ void normBk(unsigned int *N_tri, double *Bk, double norm, int d_totBins) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    if (tid < d_totBins && N_tri[tid] > 0) {
        Bk[tid] /= (norm*N_tri[tid]);
    }
}

int main(int argc, char *argv[]) {
    std::cout << "bispecGPU v0.1: This is not yet working software." << std::endl;
    
    parameters p(argv[1]);
    p.print();
    
    std::ofstream fout;
    
    vec3<double> L = {p.getd("Lx"), p.getd("Ly"), p.getd("Lz")};
    vec3<double> r_min = {p.getd("xmin"), p.getd("ymin"), p.getd("zmin")};
    vec3<double> galpk_nbw = {0.0, 0.0, 0.0}, ranpk_nbw = {0.0, 0.0, 0.0};
    vec3<double> galbk_nbw = {0.0, 0.0, 0.0}, ranbk_nbw = {0.0, 0.0, 0.0};
    vec3<int> N = {p.geti("Nx"), p.geti("Ny"), p.geti("Nz")};
    vec3<double> Delta_k = {double(2.0*pi)/L.x, double(2.0*pi)/L.y, double(2.0*pi)/L.z};
    vec3<double> dr = {L.x/double(N.x), L.y/double(N.y), L.z/double(N.z)};
    double2 k_lim = {p.getd("k_min"), p.getd("k_max")};
    double *nden_gal, *nden_ran;
    
    std::vector<std::string> hdus(2);
    hdus[0] = p.gets("hdus", 0);
    hdus[1] = p.gets("hdus", 1);
    
    double start = omp_get_wtime();
    double nbar_ran = readFits(p.gets("randomsFile"), hdus, 1, nden_ran, L, N, r_min,
             ranpk_nbw, ranbk_nbw, p.getd("P_w"), galFlags::INPUT_WEIGHT|galFlags::CIC,
             p.getd("Omega_M"), p.getd("Omega_L"), p.getd("z_min"), p.getd("z_max"));
    std::cout << "    Time to read in and bin randoms: " << omp_get_wtime() - start << " s" 
    << std::endl;
    
    std::cout << "Grid dimensions: " << N.x << ", " << N.y << ", " << N.z << std::endl;
    std::cout << "Box size: " << L.x << ", " << L.y << ", " << L.z << std::endl;
    
    start = omp_get_wtime();
    double nbar_gal = readFits(p.gets("galaxyFile"), hdus, 1, nden_gal, L, N, r_min, galpk_nbw, galbk_nbw, p.getd("P_w"),
             galFlags::INPUT_WEIGHT|galFlags::CIC, p.getd("Omega_M"), p.getd("Omega_L"), 
             p.getd("z_min"), p.getd("z_max"));
    std::cout << "    Time to read in and bin galaxys: " << omp_get_wtime() - start << std::endl;
    
    std::cout << "nbar_ran = " << nbar_ran << std::endl;
    std::cout << "nbar_gal = " << nbar_gal << std::endl;
    

    int N_tot = N.x*N.y*N.z;
    double alpha = galpk_nbw.x/ranpk_nbw.x;
    const double shotnoise = galpk_nbw.y + alpha*alpha*ranpk_nbw.y;
    fftw_complex *delta = new fftw_complex[N_tot];
    
    std::cout << "alpha = " << alpha << std::endl;
    std::cout << "shotnoise = " << shotnoise << std::endl;
    
    std::cout << "Calculating delta(r)..." << std::endl;
    for (int i = 0; i < N.x; ++i) {
        for (int j = 0; j < N.y; ++j) {
            for (int l = 0; l < N.z; ++l) {
                int index = l + N.z*(j + N.y*i);
                delta[index][0] = nden_gal[index] - alpha*nden_ran[index];
                delta[index][1] = 0.0;
            }
        }
    }
    
    delete[] nden_gal;
    delete[] nden_ran;
    
    std::cout << "Initializing power spectrum..." << std::endl;
    vec2<double> klim = {k_lim.x, k_lim.y};
    powerspec<double> Pk(p.geti("numKVals"), klim, 0);
    std::cout << "Calculating power spectrum..." << std::endl;
    Pk.calc(delta, L, N, klim, shotnoise, p.gets("wisdomFile"), pkFlags::GRID_COR|pkFlags::CIC|pkFlags::C2C);
    std::cout << "Normalizing power spectrum..." << std::endl;
    Pk.norm(galpk_nbw.z, 0);
    Pk.print();
    Pk.writeFile(p.gets("PkOutFile"), 0);
    
    int4 N_grid;
    N_grid.x = 2*k_lim.y/Delta_k.x + 1 - (int(2*k_lim.y/Delta_k.x) % 2);
    N_grid.y = 2*k_lim.y/Delta_k.y + 1 - (int(2*k_lim.y/Delta_k.y) % 2);
    N_grid.z = 2*k_lim.y/Delta_k.z + 1 - (int(2*k_lim.y/Delta_k.z) % 2);
    N_grid.w = N_grid.x*N_grid.y*N_grid.z;
    std::cout << "Small cube dimensions: (" << N_grid.x << ", " << N_grid.y << ", " << N_grid.z << std::endl;
    std::cout << "Total number of elements: " << N_grid.w << std::endl;
    
    std::vector<int4> kvec;
    double4 *dk3d = new double4[N_grid.w];
    
    for (int i = 0; i < N_grid.w; ++i) {
        dk3d[i].x = 0.0;
        dk3d[i].y = 0.0;
        dk3d[i].z = 0.0;
        dk3d[i].w = 0.0;
    }
    
    std::vector<double> kxs(N_grid.x);
    std::vector<double> kxb(N.x);
    std::vector<double> kys(N_grid.y); 
    std::vector<double> kyb(N.y);
    std::vector<double> kzs(N_grid.z);
    std::vector<double> kzb(N.z);
    
    myfreq(kxs, N_grid.x, L.x);
    fftfreq(kxb, N.x, L.x);
    myfreq(kys, N_grid.y, L.y);
    fftfreq(kyb, N.y, L.y);
    myfreq(kzs, N_grid.z, L.z);
    fftfreq(kzb, N.z, L.z);
    
    fout.open("ks.dat", std::ios::out);
    fout.precision(15);
    for (int i = 0; i < N_grid.x; ++i)
        fout << kxs[i] << "\n";
    for (int i = 0; i < N_grid.y; ++i)
        fout << kys[i] << "\n";
    for (int i = 0; i < N_grid.z; ++i)
        fout << kzs[i] << "\n";
    fout.close();
    
    fout.open("kb.dat", std::ios::out);
    fout.precision(15);
    for (int i = 0; i < N.x; ++i)
        fout << kxb[i] << "\n";
    for (int i = 0; i < N.y; ++i)
        fout << kyb[i] << "\n";
    for (int i = 0; i < N.z; ++i)
        fout << kzb[i] << "\n";
    fout.close();
    
    std::cout << "kzs[" << N_grid.z/2 << "] = " << kzs[N_grid.z/2] << std::endl;
    
    std::cout << "Creating small cube..." << std::endl;
    for (int i = 0; i < N_grid.x; ++i) {
        int i2 = kMatch(kxs[i], kxb, L.x);
        for (int j = 0; j < N_grid.y; ++j) {
            int j2 = kMatch(kys[j], kyb, L.y);
            for (int k = 0; k < N_grid.z; ++k) {
                double k_mag = sqrt(kxs[i]*kxs[i] + kys[j]*kys[j] + kzs[k]*kzs[k]);
                int k2 = kMatch(kzs[k], kzb, L.z);
                int dkindex = k2 + N.z*(j2 + N.y*i2);
                int index = k + N_grid.z*(j + N_grid.y*i);
                if (dkindex >= N_tot || dkindex < 0) {
                    std::cout << "ERROR: index out of range" << std::endl;
                    std::cout << "   dkindex = " << dkindex << std::endl;
                    std::cout << "     N_tot = " << N_tot << std::endl;
                    std::cout << "   (" << i2 << ", " << j2 << ", " << k2 << ")" << std::endl;
                    std::cout << "   (" << i << ", " << j << ", " << k << ")" << std::endl;
                    std::cout << "   (" << kxs[i] << ", " << kys[j] << ", " << kzs[k] << ")" << std::endl;
                    return 0;
                }
                if (index >= N_grid.w || index < 0) {
                    std::cout << "ERROR: index out of range" << std::endl;
                    std::cout << "      index = " << index << std::endl;
                    std::cout << "   N_grid.w = " << N_grid.w << std::endl;
                    std::cout << "   (" << i << ", " << j << ", " << k << ")" << std::endl;
                    std::cout << "   (" << kxs[i] << ", " << kys[j] << ", " << kzs[k] << ")" << std::endl;
                    return 0;
                }
                
                vec3<double> kv = {kxs[i], kys[j], kzs[k]};
                dk3d[index].x = delta[dkindex][0];
                dk3d[index].y = delta[dkindex][1];
//                 dk3d[index].x = 1.0;
//                 dk3d[index].y = 1.0;
                dk3d[index].z = k_mag;
                dk3d[index].w = gridCorCIC(kv, dr);
                if (k_mag >= k_lim.x && k_mag < k_lim.y) {
                    int4 ktemp = {i - N_grid.x/2, j - N_grid.y/2, k - N_grid.z/2, index};
                    kvec.push_back(ktemp);
                }
            }
        }
    }
    
    std::cout << "Total number of k_1/k_2 vectors: " << kvec.size() << std::endl;
    int numKVecs = kvec.size();
    int gpuMem = 0;
    
    // Allocate memory on GPU for the k vectors
    int4 *d_kvec;
    cudaMalloc((void **)&d_kvec, numKVecs*sizeof(int4));
    std::cout << "cudaMalloc d_kvec: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    gpuMem += numKVecs*sizeof(int4);
    
    // Allocate memory on GPU for the small delta(k) cube
    double4 *d_dk3d;
    cudaMalloc((void **)&d_dk3d, N_grid.w*sizeof(double4));
    std::cout << "cudaMalloc d_dk3d: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    gpuMem += N_grid.w*sizeof(double4);
    
    // Determine the size of the grid needed to store B(k) values
//     double gridSpace = Delta_k.x;
//     if (gridSpace < Delta_k.y) gridSpace = Delta_k.y;
//     if (gridSpace < Delta_k.z) gridSpace = Delta_k.z;
    
    int numKBins = p.geti("numKVals");
    double gridSpace = (k_lim.y - k_lim.x)/p.getd("numKVals");
    int totBins = numKBins*numKBins*numKBins;
    
    // Allocate memory on the GPU and host to store B(k)
    double *Bk = new double[totBins];
    double *d_Bk;
    cudaMalloc((void **)&d_Bk, totBins*sizeof(double));
    std::cout << "cudaMalloc d_Bk: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    gpuMem += totBins*sizeof(double);
    
    // Allocate memory on the GPU and host to store the number of triangles per bin
    unsigned int *Ntri = new unsigned int[totBins];
    unsigned int *d_Ntri;
    cudaMalloc((void **)&d_Ntri, totBins*sizeof(unsigned int));
    std::cout << "cudaMalloc d_Ntri: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    gpuMem += totBins*sizeof(unsigned int);
    
    std::cout << "GPU Memory used: " << gpuMem/1E6 << " MB" << std::endl;
    
    for (int i = 0; i < totBins; ++i) {
        Bk[i] = 0.0;
        Ntri[i] = 0;
    }
    
    std::cout << "Initializing things on the GPU..." << std::endl;
    cudaMemcpy(d_Bk, Bk, totBins*sizeof(double), cudaMemcpyHostToDevice);
    std::cout << "cudaMemcpy d_Bk: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    cudaMemcpy(d_Ntri, Ntri, totBins*sizeof(unsigned int), cudaMemcpyHostToDevice);
    std::cout << "cudaMemcpy d_Ntri: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    cudaMemcpy(d_dk3d, dk3d, N_grid.w*sizeof(double4), cudaMemcpyHostToDevice);
    std::cout << "cudaMemcpy d_dk3d: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    cudaMemcpy(d_kvec, &kvec[0], numKVecs*sizeof(int4), cudaMemcpyHostToDevice);
    std::cout << "cudaMemcpy d_kvec: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    
    int numThreads = p.geti("numThreads");
    int numBlocks = ceil(numKVecs/p.getd("numThreads"));
    double term = galbk_nbw.x - alpha*alpha*alpha*ranbk_nbw.x;
    std::cout << "Additional shotnoise term: " << term << std::endl;
    
    cudaEvent_t begin, end;
    float elapsedTime;
    cudaEventCreate(&begin);
    cudaEventRecord(begin, 0);
    calcBk<<<numBlocks, numThreads>>>(d_dk3d, d_kvec, d_Ntri, d_Bk, N_grid, numKVecs, gridSpace, numKBins,
                                      totBins, k_lim, shotnoise, term, galbk_nbw.y);
    cudaEventCreate(&end);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsedTime, begin, end);
    std::cout << "Time to calculate bispectrum: " << elapsedTime << " ms" << std::endl;
        
    numBlocks = ceil(totBins/p.getd("numThreads"));
    
    cudaEventCreate(&begin);
    cudaEventRecord(begin, 0);
    normBk<<<numBlocks, numThreads>>>(d_Ntri, d_Bk, galbk_nbw.z, totBins);
    cudaEventCreate(&end);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsedTime, begin, end);
    std::cout << "Time to normalize bispectrum: " << elapsedTime << " ms" << std::endl;
    
    cudaMemcpy(Bk, d_Bk, totBins*sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "cudaMemcpy Bk: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    cudaMemcpy(Ntri, d_Ntri, totBins*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    std::cout << "cudaMemcpy Ntri: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    
    fout.open(p.gets("outfile").c_str(), std::ios::out);
    fout.precision(15);
    for (int i = 0; i < numKBins; ++i) {
        double k1 = k_lim.x + (i + 0.5)*gridSpace;
        for (int j = 0; j < numKBins; ++j) {
            double k2 = k_lim.x + (j + 0.5)*gridSpace;
            for (int k = 0; k < numKBins; ++k) {
                double k3 = k_lim.x + (k + 0.5)*gridSpace;
                int bin = k + numKBins*(j + numKBins*i);
                
                fout << k1 << " " << k2 << " " << k3 << " " << Bk[bin] << " " << Ntri[bin] <<  "\n";
            }
        }
    }
    fout.close();
    
    cudaFree(d_dk3d);
    cudaFree(d_kvec);
    cudaFree(d_Bk);
    cudaFree(d_Ntri);
    
    delete[] dk3d;
    delete[] Bk;
    delete[] Ntri;
    
    return 0;
}
    
