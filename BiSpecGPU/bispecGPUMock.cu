// Standard library includes
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <vector>

// Third party library includes
#include <fftw3.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <vector_types.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>

// Custom library includes
#include <harppi.h>
#include <galaxy.h>
#include <powerspec.h>
#include <tpods.h>
#include <constants.h>

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

std::string filename(std::string base, int digits, int num, std::string ext) {
    std::stringstream file;
    file << base << std::setw(digits) << std::setfill('0') << num << ext;
    return file.str();
}

// Finds the bispectrum bin for a k_1, k_2, k_3 triplet
__device__ int getBkBin(float k1, float k2, float k3, float d_binWidth, int d_numBins, float kmin) {
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
__global__ void calcBk(float4 *dk3d, int4 *k1, int4 *k2, unsigned int *N_tri, float *Bk, int4 N_grid,
                       int N, float binWidth, int numBins, int totBins, float2 k_lim) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    int xShift = N_grid.x/2;
    int yShift = N_grid.y/2;
    int zShift = N_grid.z/2;
    
    if (tid < N) {
        int4 k_1 = k1[tid];
        k_1.x *= -1;
        k_1.y *= -1;
        k_1.z *= -1;
        float4 dk_1 = dk3d[k_1.w];
        int ik1 = (dk_1.z - k_lim.x)/binWidth;
        for (int i = 0; i < N; ++i) {
            int4 k_2 = k2[i];
            float4 dk_2 = dk3d[k_2.w];
            int4 k_3 = {k_1.x - k_2.x, k_1.y - k_2.y, k_1.z - k_2.z, 0};
            int i3, j3, k3;
            i3 = k_3.x + xShift;
            j3 = k_3.y + yShift;
            k3 = k_3.z + zShift;
            if (i3 >= 0 && j3 >= 0 && k3 >= 0 && i3 < N_grid.x && j3 < N_grid.y && k3 < N_grid.z) {
                k_3.w = k3 + N_grid.z*(j3 + N_grid.y*i3);
                float4 dk_3 = dk3d[k_3.w];
                if (dk_3.z < k_lim.y && dk_3.z >= k_lim.x) {
                    float grid_cor = dk_1.w*dk_2.w*dk_3.w;
                    float val = (dk_1.x*dk_2.x*dk_3.x - dk_1.x*dk_2.y*dk_3.y - dk_1.y*dk_2.x*dk_3.y - dk_1.y*dk_2.y*dk_3.x);
                    val *= grid_cor;
                    int ik2 = (dk_2.z - k_lim.x)/binWidth;
                    int ik3 = (dk_3.z - k_lim.x)/binWidth;
                    int bin = ik3 + numBins*(ik2 + numBins*ik1);
                    atomicAdd(&Bk[bin], val);
                    atomicAdd(&N_tri[bin], 1);
                }
            }
        }
    }
}

// This function normalizes the bispectrum measurements
__global__ void normBk(unsigned int *N_tri, float *Bk, float norm, int d_totBins) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    if (tid < d_totBins && N_tri[tid] > 0) {
        Bk[tid] /= (norm*N_tri[tid]);
    }
}

int main(int argc, char *argv[]) {
    std::cout << "bispecGPUMock v1.0: This will compute the bispectrum from mock galaxy" << std::endl;
    std::cout << "                    catalogs. It is designed to process multiple files," << std::endl;
    std::cout << "                    though using the parameter file a single data file" << std::endl;
    std::cout << "                    could also be processed." << std::endl;
    parameters p(argv[1]);
    p.print();
    
    std::ifstream fin;
    std::ofstream fout;
    
    std::vector<double> red;
    std::vector<double> nz;
    
    fin.open(p.gets("n_vs_z_file").c_str(), std::ios::in);
    while (!fin.eof()) {
        double rtemp, ntemp;
        fin >> rtemp >> ntemp;
        if (!fin.eof()) {
            red.push_back(rtemp);
            nz.push_back(ntemp);
        }
    }
    fin.close();
    
    gsl_spline *nofz = gsl_spline_alloc(gsl_interp_cspline, nz.size());
    gsl_interp_accel *nz_acc = gsl_interp_accel_alloc();
    gsl_spline_init(nofz, &red[0], &nz[0], nz.size());
    
    vec3<double> L = {p.getd("Lx"), p.getd("Ly"), p.getd("Lz")};
    vec3<int> N = {p.geti("Nx"), p.geti("Ny"), p.geti("Nz")};
    vec3<double> dr = {L.x/double(N.x), L.y/double(N.y), L.z/double(N.z)};
    vec3<double> dk = {double(2.0*pi)/L.x, double(2.0*pi)/L.y, double(2.0*pi)/L.z};
    vec3<double> r_min = {p.getd("x_min"), p.getd("y_min"), p.getd("z_min")};
//     vec3<double> r_max = {L.x + r_min.x, L.y + r_min.y, L.z + r_min.z};
    vec3<double> ranpk_nbw = {0.0, 0.0, 0.0}, ranbk_nbw = {0.0, 0.0, 0.0};
    vec2<double> k_lim = {p.getd("k_min"), p.getd("k_max")};
    float2 d_klim = {float(p.getd("k_min")), float(p.getd("k_max"))};
    
    int N_tot = N.x*N.y*N.z;
    double *nden_ran = new double[N_tot];
    int num_rans = 0;
    
    for (int i = 0; i < N_tot; ++i)
        nden_ran[i] = 0.0;
    
    std::cout << "Reading in randoms from file: " << p.gets("ran_file") << std::endl;
    gsl_integration_workspace *w_gsl = gsl_integration_workspace_alloc(100000000);
    fin.open(p.gets("ran_file").c_str(), std::ios::in);
    while (!fin.eof()) {
        double ra, dec, red, temp1;
        fin >> ra >> dec >> red >> temp1;
        if (!fin.eof() && red >= p.getd("red_min") && red < p.getd("red_max")) {
            galaxy<double> ran(ra, dec, red, 0.0, 0.0, 0.0, gsl_spline_eval(nofz, red, nz_acc), 0.0, 0.0);
            ran.cartesian(p.getd("Omega_M"), p.getd("Omega_L"), w_gsl);
            ran.bin(nden_ran, L, N, r_min, ranpk_nbw, ranbk_nbw, p.getd("P_w"), 
                    galFlags::FKP_WEIGHT|galFlags::CIC);
            ++num_rans;
        }
    }
    fin.close();
    
    std::cout << "    Number of randoms: " << num_rans << std::endl;
    
    for (int mock = p.geti("start_num"); mock < p.geti("num_mocks")+p.geti("start_num"); ++mock) {
        std::string in_file = filename(p.gets("in_base"), p.geti("digits"), mock, p.gets("in_ext"));
        std::string pk_file = filename(p.gets("pk_base"), p.geti("digits"), mock, p.gets("pk_ext"));
        std::string bk_file = filename(p.gets("bk_base"), p.geti("digits"), mock, p.gets("bk_ext"));
        
        std::cout << "Processing mock: " << in_file << std::endl;
        
        int num_gals = 0;
        vec3<double> galpk_nbw = {0.0, 0.0, 0.0}, galbk_nbw = {0.0, 0.0, 0.0};
        double *nden_gal = new double[N_tot];
        
        for (int i = 0; i < N_tot; ++i)
            nden_gal[i] = 0.0;
        
        std::cout << "    Reading in and binning galaxies..." << std::endl;
        fin.open(in_file.c_str(), std::ios::in);
        while (!fin.eof()) {
            double ra, dec, red, temp1, temp2;
            fin >> ra >> dec >> red >> temp1 >> temp2;
            if (!fin.eof() && red >= p.getd("red_min") && red < p.getd("red_max")) {
                galaxy<double> gal(ra, dec, red, 0.0, 0.0, 0.0, gsl_spline_eval(nofz, red, nz_acc), 0.0, 
                                   0.0);
                gal.cartesian(p.getd("Omega_M"), p.getd("Omega_L"), w_gsl);
                gal.bin(nden_gal, L, N, r_min, galpk_nbw, galbk_nbw, p.getd("P_w"),
                        galFlags::FKP_WEIGHT|galFlags::CIC);
                ++num_gals;
            }
        }
        fin.close();
        
        const double alpha = galpk_nbw.x/ranpk_nbw.x;
        const double shotnoise = galpk_nbw.y + alpha*alpha*ranpk_nbw.y;
        
        fftw_complex *delta = new fftw_complex[N_tot];
        
        std::cout << "    Calculating overdensity field..." << std::endl;
        for (int i = 0; i < N_tot; ++i) {
            delta[i][0] = nden_gal[i] - alpha*nden_ran[i];
            delta[i][1] = 0.0;
        }
        
        delete[] nden_gal;
        
        std::cout << "    Calculating power spectrum..." << std::endl;
        powerspec<double> Pk(p.geti("num_k_vals"), k_lim, 0);
        Pk.calc(delta, L, N, k_lim, shotnoise, p.gets("wisdom_file"), 
                pkFlags::GRID_COR|pkFlags::CIC|pkFlags::C2C);
        Pk.norm(galpk_nbw.z, 0);
        Pk.writeFile(pk_file, 0);
        
        int4 N_grid;
        N_grid.x = 2*k_lim.y/dk.x + 1 - (int(2*k_lim.y/dk.x) % 2);
        N_grid.y = 2*k_lim.y/dk.y + 1 - (int(2*k_lim.y/dk.y) % 2);
        N_grid.z = 2*k_lim.y/dk.z + 1 - (int(2*k_lim.y/dk.z) % 2);
        N_grid.w = N_grid.x*N_grid.y*N_grid.z;
        std::cout << "Small cube dimensions: (" << N_grid.x << ", " << N_grid.y << ", " << N_grid.z << ")" << std::endl;
        std::cout << "Total number of elements: " << N_grid.w << std::endl;
        
        std::vector<int4> kvec;
        float4 *dk3d = new float4[N_grid.w];
        
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
        
        std::cout << "    Filling small cube for bispectrum calculation..." << std::endl;
        for (int i = 0; i < N_grid.x; ++i) {
            int i2 = kMatch(kxs[i], kxb, L.x);
            for (int j = 0; j < N_grid.y; ++j) {
                int j2 = kMatch(kys[j], kyb, L.y);
                for (int k = 0; k < N_grid.z; ++k) {
                    float k_mag = sqrt(kxs[i]*kxs[i] + kys[j]*kys[j] + kzs[k]*kzs[k]);
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
                        int4 ktemp = {i - N_grid.x/2, j - N_grid.y/2, k - N_grid.z/2, index};
                        kvec.push_back(ktemp);
                    }
                }
            }
        }
        
        int num_k_vecs = kvec.size();
        int gpu_mem = 0;
        
        delete[] delta;
        
        int4 *d_k1;
        cudaMalloc((void **)&d_k1, num_k_vecs*sizeof(int4));
        std::cout << "    cudaMalloc d_k1: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        gpu_mem += num_k_vecs*sizeof(int4);
        
        int4 *d_k2;
        cudaMalloc((void **)&d_k2, num_k_vecs*sizeof(int4));
        std::cout << "    cudaMalloc d_k2: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        gpu_mem += num_k_vecs*sizeof(int4);
        
        float4 *d_dk3d;
        cudaMalloc((void **)&d_dk3d, N_grid.w*sizeof(float4));
        std::cout << "    cudaMalloc d_dk3d: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        gpu_mem += N_grid.w*sizeof(float4);
        
        float grid_space = p.getd("bin_scale")*dk.x;
        int num_bk_bins = ceil((k_lim.y - k_lim.x)/grid_space);
        int tot_bk_bins = num_bk_bins*num_bk_bins*num_bk_bins;
        
        float *Bk = new float[tot_bk_bins];
        float *d_Bk;
        cudaMalloc((void **)&d_Bk, tot_bk_bins*sizeof(float));
        std::cout << "    cudaMalloc d_Bk: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        gpu_mem += tot_bk_bins*sizeof(float);
        
        unsigned int *Ntri = new unsigned int[tot_bk_bins];
        unsigned int *d_Ntri;
        cudaMalloc((void **)&d_Ntri, tot_bk_bins*sizeof(unsigned int));
        std::cout << "    cudaMalloc d_Ntri: "<< cudaGetErrorString(cudaGetLastError()) << std::endl;
        gpu_mem += tot_bk_bins*sizeof(unsigned int);
        
        std::cout << "    GPU Memory used: " << gpu_mem/1E6 << " MB" << std::endl;
        
        for (int i = 0; i < tot_bk_bins; ++i) {
            Bk[i] = 0.0;
            Ntri[i] = 0;
        }
        
        std::cout << "    Copying data to the GPU..." << std::endl;
        cudaMemcpy(d_Bk, Bk, tot_bk_bins*sizeof(float), cudaMemcpyHostToDevice);
        std::cout << "    cudaMemcpy d_Bk: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        cudaMemcpy(d_Ntri, Ntri, tot_bk_bins*sizeof(unsigned int), cudaMemcpyHostToDevice);
        std::cout << "    cudaMemcpy d_Ntri: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        cudaMemcpy(d_dk3d, dk3d, N_grid.w*sizeof(float4), cudaMemcpyHostToDevice);
        std::cout << "    cudaMemcpy d_dk3d: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        cudaMemcpy(d_k1, &kvec[0], num_k_vecs*sizeof(int4), cudaMemcpyHostToDevice);
        std::cout << "    cudaMemcpy d_k1: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        cudaMemcpy(d_k2, &kvec[0], num_k_vecs*sizeof(int4), cudaMemcpyHostToDevice);
        std::cout << "    cudaMemcpy d_k2: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        
        int num_gpu_threads = p.geti("num_gpu_threads");
        int num_blocks = ceil(num_k_vecs/p.getd("num_gpu_threads"));
        
        cudaEvent_t begin, end;
        float elapsedTime;
        cudaEventCreate(&begin);
        cudaEventRecord(begin, 0);
        calcBk<<<num_blocks, num_gpu_threads>>>(d_dk3d, d_k1, d_k2, d_Ntri, d_Bk, N_grid, num_k_vecs, 
                                                grid_space, num_bk_bins, tot_bk_bins, d_klim);
        cudaEventCreate(&end);
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsedTime, begin, end);
        std::cout << "    Time to calculate bispectrum: " << elapsedTime << " ms" << std::endl;
        
        num_blocks = ceil(tot_bk_bins/p.getd("num_gpu_threads"));
        
        cudaEventCreate(&begin);
        cudaEventRecord(begin, 0);
        normBk<<<num_blocks, num_gpu_threads>>>(d_Ntri, d_Bk, float(galbk_nbw.z), tot_bk_bins);
        cudaEventCreate(&end);
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsedTime, begin, end);
        std::cout << "Time to normalize bispectrum: " << elapsedTime << " ms" << std::endl;
        
        cudaMemcpy(Bk, d_Bk, tot_bk_bins*sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "    cudaMemcpy Bk: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        cudaMemcpy(Ntri, d_Ntri, tot_bk_bins*sizeof(unsigned int), cudaMemcpyDeviceToHost);
        std::cout << "    cudaMemcpy Ntri: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        
        fout.open(bk_file.c_str(), std::ios::out);
        fout.precision(15);
        for (int i = 0; i < num_bk_bins; ++i) {
            double k1 = k_lim.x + (i + 0.5)*grid_space;
            for (int j = 0; j < num_bk_bins; ++j) {
                double k2 = k_lim.x + (j + 0.5)*grid_space;
                for (int k = 0; k < num_bk_bins; ++k) {
                    double k3 = k_lim.x + (k + 0.5)*grid_space;
                    int bin = k + num_bk_bins*(j + num_bk_bins*i);
                    
                    fout << k1 << " " << k2 << " " << k3 << " " << Bk[bin] << " " << Ntri[bin] << "\n";
                }
            }
        }
        fout.close();
        
        delete[] Bk;
        delete[] Ntri;
        delete[] dk3d;
        cudaFree(d_k1);
        cudaFree(d_k2);
        cudaFree(d_dk3d);
        cudaFree(d_Bk);
        cudaFree(d_Ntri);
    }
    
    // Clean up before exiting
    delete[] nden_ran;
    gsl_spline_free(nofz);
    gsl_interp_accel_free(nz_acc);
    gsl_integration_workspace_free(w_gsl);
    
    return 0;
}
