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

__global__ void calcPk(double4 *dk3d, int4 *kvec, unsigned int *Nk, double *Pk, int4 N_grid, int N,
                       double binWidth, int numBins, double2 k_lim, double SN) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    if (tid < N) {
        int4 k = kvec[tid];
        double4 dk = dk3d[k.w];
        int bin = (dk.z - k_lim.x)/binWidth;
        if (bin < numBins) {
            double pow = (dk.x*dk.x + dk.y*dk.y - SN)*dk.w*dk.w;
            atomicAdd(&Pk[bin], pow);
            atomicAdd(&Nk[bin], 1);
        }
    }
}

__global__ void normPk(unsigned int *Nk, double *Pk, double norm, int numBins) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    if (tid < numBins && Nk[tid] > 0) {
        Pk[tid] /= (norm*Nk[tid]);
    }
}

int main(int argc, char *argv[]) {
    std::cout << "pkGPU v0.1: This is not yet working software." << std::endl;
    
    parameters p(argv[1]);
    p.print();
    
    std::ofstream fout;
    
    vec3<double> L = {p.getd("Lx"), p.getd("Ly"), p.getd("Lz")};
    vec3<double> r_min = {p.getd("xmin"), p.getd("ymin"), p.getd("zmin")};
    vec4<double> gal_nbw = {0.0, 0.0, 0.0, 0.0}, ran_nbw = {0.0, 0.0, 0.0, 0.0};
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
             ran_nbw, p.getd("P_w"), galFlags::INPUT_WEIGHT|galFlags::CIC, p.getd("Omega_M"),
             p.getd("Omega_L"), p.getd("z_min"), p.getd("z_max"));
    std::cout << "    Time to read in and bin randoms: " << omp_get_wtime() - start << " s" 
    << std::endl;
    
    std::cout << "Grid dimensions: " << N.x << ", " << N.y << ", " << N.z << std::endl;
    std::cout << "Box size: " << L.x << ", " << L.y << ", " << L.z << std::endl;
    
    start = omp_get_wtime();
    double nbar_gal = readFits(p.gets("galaxyFile"), hdus, 1, nden_gal, L, N, r_min, gal_nbw, p.getd("P_w"),
             galFlags::INPUT_WEIGHT|galFlags::CIC, p.getd("Omega_M"), p.getd("Omega_L"), 
             p.getd("z_min"), p.getd("z_max"));
    std::cout << "    Time to read in and bin galaxys: " << omp_get_wtime() - start << std::endl;
    
    std::cout << "nbar_ran = " << nbar_ran << std::endl;
    std::cout << "nbar_gal = " << nbar_gal << std::endl;
    

    int N_tot = N.x*N.y*N.z;
    double alpha = gal_nbw.x/ran_nbw.x;
    const double shotnoise = gal_nbw.y + alpha*alpha*ran_nbw.y;
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
    Pk.norm(gal_nbw.z, 0);
    Pk.print();
    Pk.writeFile(p.gets("PkOutFile"), 0);
    
    int4 N_grid = {(2*k_lim.y/Delta_k.x + 1), (2*k_lim.y/Delta_k.y + 1), (2*k_lim.y/Delta_k.z + 1), 0};
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
                dk3d[index].z = k_mag;
                dk3d[index].w = gridCorCIC(kv, dr);
                if (k_mag >= k_lim.x && k_mag < k_lim.y) {
                    int4 ktemp = {i - N_grid.x/2 + 1, j - N_grid.y/2 + 1, k - N_grid.z/2 + 1, index};
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
    
    double *Pkgpu = new double[p.geti("numKVals")];
    double *d_Pk;
    cudaMalloc((void **)&d_Pk, p.geti("numKVals")*sizeof(double));
    std::cout << "cudaMalloc d_Pk: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    gpuMem += p.geti("numKVals")*sizeof(double);
    
    unsigned int *Nk = new unsigned int[p.geti("numKVals")];
    unsigned int *d_Nk;
    cudaMalloc((void **)&d_Nk, p.geti("numKVals")*sizeof(unsigned int));
    std::cout << "cudaMalloc d_Nk: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    gpuMem += p.geti("numKVals")*sizeof(unsigned int);
    
    std::cout << "GPU Memory used: " << gpuMem/1E6 << " MB" << std::endl;
    
    for (int i = 0; i < p.geti("numKVals"); ++i) {
        Pkgpu[i] = 0.0;
        Nk[i] = 0;
    }
    
    cudaMemcpy(d_Pk, Pkgpu, p.geti("numKVals")*sizeof(double), cudaMemcpyHostToDevice);
    std::cout << "cudaMemcpy d_Pk: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    cudaMemcpy(d_Nk, Nk, p.geti("numKVals")*sizeof(unsigned int), cudaMemcpyHostToDevice);
    std::cout << "cudaMemcpy d_Nk: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    cudaMemcpy(d_dk3d, dk3d, N_grid.w*sizeof(double4), cudaMemcpyHostToDevice);
    std::cout << "cudaMemcpy d_dk3d: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    cudaMemcpy(d_kvec, &kvec[0], numKVecs*sizeof(int4), cudaMemcpyHostToDevice);
    std::cout << "cudaMemcpy d_kvec: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    
    int numThreads = p.geti("numThreads");
    int numBlocks = ceil(numKVecs/p.getd("numThreads"));
    double binWidth = (k_lim.y - k_lim.x)/p.getd("numKVals");
    
    cudaEvent_t begin, end;
    float elapsedTime;
    cudaEventCreate(&begin);
    cudaEventRecord(begin, 0);
    calcPk<<<numBlocks, numThreads>>>(d_dk3d, d_kvec, d_Nk, d_Pk, N_grid, numKVecs, binWidth, 
                                      p.geti("numKVals"), k_lim, shotnoise);
    cudaEventCreate(&end);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsedTime, begin, end);
    std::cout << "Time to calculate power spectrum: " << elapsedTime << " ms" << std::endl;
    
    cudaEventCreate(&begin);
    cudaEventRecord(begin, 0);
    normPk<<<1, p.geti("numKVals")>>>(d_Nk, d_Pk, gal_nbw.z, p.geti("numKVals"));
    cudaEventCreate(&end);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsedTime, begin, end);
    std::cout << "Time to normalize bispectrum: " << elapsedTime << " ms" << std::endl;
    
    cudaMemcpy(Pkgpu, d_Pk, p.geti("numKVals")*sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "cudaMemcpy Pkgpu: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    cudaMemcpy(Nk, d_Nk, p.geti("numKVals")*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    std::cout << "cudaMemcpy Nk: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    
    fout.open(p.gets("PkGPUFile").c_str(), std::ios::out);
    fout.precision(15);
    for (int i = 0; i < p.geti("numKVals"); ++i) {
        double k = k_lim.x + (i + 0.5)*binWidth;
        fout << k << " " << Pkgpu[i] << " " << Nk[i] << "\n";
    }
    fout.close();
    
    std::cout << "gal_nbw.w = " << gal_nbw.w << std::endl;
    
    cudaFree(d_dk3d);
    cudaFree(d_kvec);
    cudaFree(d_Pk);
    cudaFree(d_Nk);
    
    delete[] Pkgpu;
    delete[] Nk;
    delete[] dk3d;
    
    return 0;
}
