#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <fftw3.h>
#include <harppi.h>
#include <galaxy.h>
#include <powerspec.h>
#include <tpods.h>
#include <constants.h>
#include <gpuerrchk.h>
#include "include/bispec.h"
#include "include/add_funcs.h"
#include "include/file_reader.h"
#include "include/density_field.h"
#include "include/cosmology.h"
#include "include/file_check.h"


int main(int argc, char *argv[]) {
    parameters p(argv[1]);
    p.print();
    
    vec3<double> L = {p.getd("Lx"), p.getd("Ly"), p.getd("Lz")};
    vec3<int> N = {p.geti("Nx"), p.geti("Ny"), p.geti("Nz")};
    vec3<double> dr = {L.x/double(N.x), L.y/double(N.y), L.z/double(N.z)};
    vec3<double> dk = {double(2.0*pi)/L.x, double(2.0*pi)/L.y, double(2.0*pi)/L.z};
    vec3<double> r_min = {p.getd("x_min"), p.getd("y_min"), p.getd("z_min")};
    vec3<double> ranpk_nbw = {0.0, 0.0, 0.0}, ranbk_nbw = {0.0, 0.0, 0.0};
    vec2<double> k_lim = {p.getd("k_min"), p.getd("k_max")};
    vec2<double> red_lim = {p.getd("red_min"), p.getd("red_max")};
    float dklim[] = {float(p.getd("k_min")), float(p.getd("k_max"))};
    cosmology cos(p.getd("H_0"), p.getd("Omega_M"), p.getd("Omega_L"), p.getd("Omega_b"), p.getd("Omega_c"),
                  p.getd("tau"), p.getd("T_CMB"));
    
    gpuErrchk(cudaMemcpyToSymbol(d_klim, &dklim, 2*sizeof(float)));
    
    
    
    std::cout << "Reading in and binning randoms file..." << std::endl;
    size_t num_rans;
    densityField nden_ran(L, N, r_min);
    if (p.gets("file_type") == "Patchy" || p.gets("file_type") == "patchy") {
        num_rans = readPatchy(p.gets("randoms_file"), nden_ran, cos, red_lim, p.getd("P_FKP"), true);
    } else if(p.gets("file_type") == "QPM") {
        num_rans = readQPM(p.gets("randoms_file"), nden_ran, red_lim.x, red_lim.y, p.getd("P_FKP"),
                           p.getd("Omega_M"), p.getd("Omega_L"), r_min, L, N, ranpk_nbw, ranbk_nbw, true);
    } else if(p.gets("file_type") == "fits") {
        std::vector<std::string> cols = {"RA", "DEC", "RED", "W_FKP", "W_SYS"};
        num_rans = readFits(p.gets("randoms_file"), nden_ran, cols, true);
    } else {
        std::stringstream message;
        message << "Invalid file_type" << std::endl;
        throw std::runtime_error(message.str());
    }
    
    for (int mock = p.geti("start_num"), mock < p.geti("num_mocks") + p.geti("start_num"); ++mock) {
        std::string in_file = filename(p.gets("in_base"), p.geti("digits"), mock, p.gets("in_ext"));
        std::string pk_file = filename(p.gets("pk_base"), p.geti("digits"), mock, p.gets("pk_ext"));
        std::string bk_file = filename(p.gets("bk_base"), p.geti("digits"), mock, p.gets("bk_ext"));
        
        std::cout << "Processing mock: " << in_file << std::endl;
        
        size_t num_gals;
        vec3<double> galpk_nbw = {0.0, 0.0, 0.0}, galbk_nbw = {0.0, 0.0, 0.0};
        std::vector<double> nden_gal(N.x*N.y*N.z);
        
        std::cout << "    Reading in and binning galaxies..." << std::endl;
        if (p.gets("file_type") == "Patchy" || p.gets("file_type") == "patchy") {
            num_gals = readPatchy(in_file, nden_gal, red_lim.x, red_lim.y, p.getd("P_FKP"), 
                                  p.getd("Omega_M"), p.getd("Omega_L"), r_min, L, N, galpk_nbw, galbk_nbw,
                                  false);
        } else if (p.gets("file_type") == "QPM") {
            num_gals = readQPM(in_file, nden_gal, red_lim.x, red_lim.y, p.getd("P_FKP"), 
                               p.getd("Omega_M"), p.getd("Omega_L"), r_min, L, N, galpk_nbw, galbk_nbw,
                               false);
        } else if (p.gets("file_type") == "fits") {
            std::vector<std::string> cols = {"RA", "DEC", "RED", "W_FKP", "W_SYS"};
            num_gals = readFits(in_file, nden_gal, cols, false);
        }
        
        const double alpha = galpk_nbw.x/ranpk_nbw.x;
        const double shotnoise = galpk_nbw.y + alpha*alpha*ranpk_nbw.y;
        
        fftw_complex *delta = new fftw_complex[N.x*N.y*N.z];
        
        std::cout << "    Calculating overdensity field..." << std::endl;
        for (int i = 0; i < N.x*N.y*N.z; ++i) {
            delta[i][0] = nden_gal[i] - alpha*nden_ran[i];
            delta[i][1] = 0.0;
        }
        
        std::cout << "    Calculating power spectrum..." << std::endl;
        powerspec<double> Pk(p.geti("num_k_vals"), k_lim, 0);
        Pk.calc(delta, L, N, k_lim, shotnoise, p.gets("wisdom_file"), 
                pkFlags::GRID_COR|pkFlags::CIC|pkFlags::C2C);
        Pk.norm(galpk_nbw.z, 0);
        Pk.writeFile(pk_file, 0);
        
        int N_grid[4];
        N_grid[0] = 2*k_lim.y/dk.x + 1 - (int(2*k_lim.y/dk.x) % 2);
        N_grid[1] = 2*k_lim.y/dk.y + 1 - (int(2*k_lim.y/dk.y) % 2);
        N_grid[2] = 2*k_lim.y/dk.z + 1 - (int(2*k_lim.y/dk.z) % 2);
        N_grid[3] = N_grid[0]*N_grid[1]*N_grid[2];
        std::cout << "    Small cube dimension: (" << N_grid[0] << ", " << N_grid[1] << ", " << N_grid[2];
        std::cout << ")" << std::endl;
        std::cout << "    Total number of elements: " << N_grid[3] << std::endl;
        
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
        
        delete[] delta;
        
        int4 *d_k;
        cudaMalloc((void **)&d_k1, num_k_vecs*sizeof(int4));
        std::cout << "    cudaMalloc d_k1: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        gpu_mem += num_k_vecs*sizeof(int4);
        
        float4 *d_dk3d;
        cudaMalloc((void **)&d_dk3d, N_grid.w*sizeof(float4));
        std::cout << "    cudaMalloc d_dk3d: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        gpu_mem += N_grid.w*sizeof(float4);
        
        float grid_space = 0.008;
        int num_bk_bins = ceil((k_lim.y - k_lim.x)/grid_space);
        int tot_bk_bins = num_bk_bins*num_bk_bins*num_bk_bins;
        
        double *Bk = new double[tot_bk_bins];
        double *d_Bk;
        cudaMalloc((void **)&d_Bk, tot_bk_bins*sizeof(double));
        std::cout << "    cudaMalloc d_Bk: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        gpu_mem += tot_bk_bins*sizeof(double);
        
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
        cudaMemcpy(d_Bk, Bk, tot_bk_bins*sizeof(double), cudaMemcpyHostToDevice);
        std::cout << "    cudaMemcpy d_Bk: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        cudaMemcpy(d_Ntri, Ntri, tot_bk_bins*sizeof(unsigned int), cudaMemcpyHostToDevice);
        std::cout << "    cudaMemcpy d_Ntri: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        cudaMemcpy(d_dk3d, dk3d, N_grid.w*sizeof(float4), cudaMemcpyHostToDevice);
        std::cout << "    cudaMemcpy d_dk3d: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        cudaMemcpy(d_k1, &kvec[0], num_k_vecs*sizeof(int4), cudaMemcpyHostToDevice);
        std::cout << "    cudaMemcpy d_k1: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        
        dim3 num_gpu_threads(p.geti("num_gpu_threads"), p.geti("num_gpu_threads"));
        dim3 num_blocks(ceil(num_k_vecs/p.getd("num_gpu_threads")),
                        ceil(num_k_vecs/p.getd("num_gpu_threads")));
        
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
        
        cudaMemcpy(Bk, d_Bk, tot_bk_bins*sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << "    cudaMemcpy Bk: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        cudaMemcpy(Ntri, d_Ntri, tot_bk_bins*sizeof(unsigned int), cudaMemcpyDeviceToHost);
        std::cout << "    cudaMemcpy Ntri: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        
        fout.open(bk_file.c_str(), std::ios::out);
//         fout.precision(15);
        for (int i = 0; i < num_bk_bins; ++i) {
            double k1 = k_lim.x + (i + 0.5)*grid_space;
            for (int j = 0; j < num_bk_bins; ++j) {
                double k2 = k_lim.x + (j + 0.5)*grid_space;
                for (int k = 0; k < num_bk_bins; ++k) {
                    double k3 = k_lim.x + (k + 0.5)*grid_space;
                    int bin = k + num_bk_bins*(j + num_bk_bins*i);
                    
                    fout << std::setprecision(3) << k1 << " " << k2 << " " << k3 << " " << std::setprecision(15) << Bk[bin] << " " << Ntri[bin] << "\n";
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
        
