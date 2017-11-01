/* bispecGPUMock v. 2.0
 * David W. Pearson
 * October 30, 2017
 * 
 * This code will calculate the bispectrum of galaxies from mock catalogues utilizing the GPU to speed up
 * the O(N^2) calculation. The original approach had each thread taking on k_1 vector and then run over
 * other vectors as k_2, starting with k_2 = k_1 and skipping other vectors that had already been used as
 * k_1. This is not the best approach, as some threads will have significantly longer execution times than
 * others.
 * 
 * In this version, the calculation will take place in a 2D grid of 2D thread blocks. Thus, each thread 
 * will calculate the bispectrum contribution from a single k_1, k_2 pair. This way, each thread should
 * execute quickly. Additionally, some thread blocks will only have a little over half their threads 
 * actually doing work, per the condition mentioned above, while other thread blocks will be skipped 
 * entirely. It is hoped that the relatively quick execution of the skipped blocks, along with the faster
 * single thread execution, will speed up the code significantly.
 * 
 * A downside to this approach, however, is that the k-range must be adjusted so that the total number of
 * thread blocks per dimension remains at or below 65535. Since the maximum number of threads per block
 * is 1024, a 2D block can be at most 32x32. So the maximum number of k-vectors in the range of interest
 * is then 2097120.
 * 
 * In addition to the potential speed-up, the other reason for this rewrite is to fix the mess of the 
 * original code. This has been attempted by creating several different classes to make the code in this
 * file a bit cleaner. Unfortunately, due to time constraints I won't be able to fully refactor the code
 * into the most cleanly object-oriented version possible (it would require a lot more classes, as well
 * as creating some base classes and redefining others to be derived classes).
 */


// Standard library includes
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>

// Third party library includes
#include <fftw3.h>
#include <omp.h>
#include <gsl/gsl_spline.h>
#include <cuda.h>
#include <vector_types.h>

// Custom includes
#include "include/harppi.h"
#include "include/galaxy.h"
#include "include/tpods.h"
#include "include/constants.h"
#include "include/gpuerrchk.h"
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
    vec3<double> dk = {double(2.0*pi)/L.x, double(2.0*pi)/L.y, double(2.0*pi)/L.z};
    vec3<double> r_min = {p.getd("x_min"), p.getd("y_min"), p.getd("z_min")};
    vec3<double> dr = {L.x/double(N.x), L.y/double(N.y), L.z/double(N.z)};
    vec2<double> k_lim = {p.getd("k_min"), p.getd("k_max")};
    vec2<double> red_lim = {p.getd("red_min"), p.getd("red_max")};
    float dklim[] = {float(p.getd("k_min")), float(p.getd("k_max"))};
    cosmology cosmo(p.getd("H_0"), p.getd("Omega_M"), p.getd("Omega_L"), p.getd("Omega_b"), p.getd("Omega_c"),
                  p.getd("tau"), p.getd("T_CMB"));
    float binWidth = p.getd("binWidth");
    int numBins = ceil((k_lim.y - k_lim.x)/binWidth);
    int totBins = numBins*numBins*numBins;
    int N_tot = N.x*N.y*N.z;
    gsl_spline *NofZ;
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    
    gpuErrchk(cudaMemcpyToSymbol(d_klim, &dklim, 2*sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(d_binWidth, &binWidth, sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(d_numBins, &numBins, sizeof(float)));
    
    if (p.gets("fileType") == "QPM") {
        if (check_file_exists(p.gets("nz_file"))) {
            std::ifstream fin(p.gets("nz_file").c_str());
            std::vector<double> zin;
            std::vector<double> nin;
            while (!fin.eof()) {
                double zt, nt;
                fin >> zt >> nt;
                if (!fin.eof()) {
                    zin.push_back(zt);
                    nin.push_back(nt);
                }
            }
            fin.close();
            
            NofZ = gsl_spline_alloc(gsl_interp_cspline, nin.size());
            gsl_spline_init(NofZ, zin.data(), nin.data(), nin.size());
        }
    }
    
    std::cout << "Reading in and binning randoms file..." << std::endl;
    size_t num_rans;
    densityField nden_ran(L, N, r_min);
    if (p.gets("fileType") == "Patchy" || p.gets("fileType") == "patchy") {
        num_rans = readPatchy(p.gets("ran_file"), nden_ran, cosmo, red_lim, p.getd("P_FKP"), true);
    } else if(p.gets("fileType") == "QPM") {
        num_rans = readQPM(p.gets("ran_file"), nden_ran, cosmo, red_lim, p.getd("P_FKP"), true, NofZ,
                           acc);
    } else {
        std::stringstream message;
        message << "Invalid fileType" << std::endl;
        throw std::runtime_error(message.str());
    }
    
    fftw_complex *delta = new fftw_complex[N_tot];
        
    fftw_init_threads();
    fftw_import_wisdom_from_filename(p.gets("wisdom_file").c_str());
    fftw_plan_with_nthreads(omp_get_max_threads());
    fftw_plan dr2dk = fftw_plan_dft_3d(N.x, N.y, N.z, delta, delta, FFTW_FORWARD, FFTW_MEASURE);
    fftw_export_wisdom_to_filename(p.gets("wisdom_file").c_str());
    
    for (int mock = p.geti("start_num"); mock < p.geti("num_mocks") + p.geti("start_num"); ++mock) {
        std::string in_file = filename(p.gets("in_base"), p.geti("digits"), mock, p.gets("in_ext"));
        std::string pk_file = filename(p.gets("pk_base"), p.geti("digits"), mock, p.gets("pk_ext"));
        std::string bk_file = filename(p.gets("bk_base"), p.geti("digits"), mock, p.gets("bk_ext"));
        
        std::cout << "Processing mock: " << in_file << std::endl;
        
        size_t num_gals;
        densityField nden_gal(L, N, r_min);
        
        std::cout << "    Reading in and binning galaxies..." << std::endl;
        if (p.gets("fileType") == "Patchy" || p.gets("fileType") == "patchy") {
            num_gals = readPatchy(in_file, nden_gal, cosmo, red_lim, p.getd("P_FKP"), false);
        } else if (p.gets("fileType") == "QPM") {
            num_gals = readQPM(in_file, nden_gal, cosmo, red_lim, p.getd("P_FKP"), false, NofZ, acc);
        }
        
        std::cout << "    Ratio: " << double(num_gals)/double(num_rans);
        const double alpha = nden_gal.nbw()/nden_ran.nbw();
        const double shotnoise = nden_gal.nbw2() + alpha*alpha*nden_ran.nbw2();        
        
        std::cout << "    Calculating overdensity field..." << std::endl;
        for (int i = 0; i < N_tot; ++i) {
            delta[i][0] = nden_gal.at(i) - alpha*nden_ran.at(i);
            delta[i][1] = 0.0;
        }
        
        std::cout << "    Fourier transforming..." << std::endl;
        fftw_execute(dr2dk);
        
        int N_grid[4];
        N_grid[0] = 2*k_lim.y/dk.x + 1 - (int(2*k_lim.y/dk.x) % 2);
        N_grid[1] = 2*k_lim.y/dk.y + 1 - (int(2*k_lim.y/dk.y) % 2);
        N_grid[2] = 2*k_lim.y/dk.z + 1 - (int(2*k_lim.y/dk.z) % 2);
        N_grid[3] = N_grid[0]*N_grid[1]*N_grid[2];
        std::cout << "    Small cube dimension: (" << N_grid[0] << ", " << N_grid[1] << ", " << N_grid[2];
        std::cout << ")" << std::endl;
        
        gpuErrchk(cudaMemcpyToSymbol(d_Ngrid, &N_grid[0], 4*sizeof(int)));
        
        std::vector<int4> kvec;
        float4 *dk3d = new float4[N_grid[3]];
        
        for (int i = 0; i < N_grid[3]; ++i) {
            dk3d[i].x = 0.0;
            dk3d[i].y = 0.0;
            dk3d[i].z = 0.0;
            dk3d[i].w = 0.0;
        }
        
        std::vector<double> kxs = myfreq(N_grid[0], L.x);
        std::vector<double> kxb = fftfreq(N.x, L.x);
        std::vector<double> kys = myfreq(N_grid[1], L.y);
        std::vector<double> kyb = fftfreq(N.y, L.y);
        std::vector<double> kzs = myfreq(N_grid[2], L.z);
        std::vector<double> kzb = fftfreq(N.z, L.z);
        
        std::cout << "    Filling small cube for bispectrum calculation..." << std::endl;
        for (int i = 0; i < N_grid[0]; ++i) {
            int i2 = kMatch(kxs[i], kxb, L.x);
            for (int j = 0; j < N_grid[1]; ++j) {
                int j2 = kMatch(kys[j], kyb, L.y);
                for (int k = 0; k < N_grid[2]; ++k) {
                    float k_mag = sqrt(kxs[i]*kxs[i] + kys[j]*kys[j] + kzs[k]*kzs[k]);
                    int k2 = kMatch(kzs[k], kzb, L.z);
                    int dkindex = k2 + N.z*(j2 + N.y*i2);
                    int index = k + N_grid[2]*(j + N_grid[1]*i);
                    if (dkindex >= N_tot || dkindex < 0) {
                        std::cout << "ERROR: index out of range" << std::endl;
                        std::cout << "   dkindex = " << dkindex << std::endl;
                        std::cout << "     N_tot = " << N_tot << std::endl;
                        std::cout << "   (" << i2 << ", " << j2 << ", " << k2 << ")" << std::endl;
                        std::cout << "   (" << i << ", " << j << ", " << k << ")" << std::endl;
                        std::cout << "   (" << kxs[i] << ", " << kys[j] << ", " << kzs[k] << ")" << std::endl;
                        return 0;
                    }
                    if (index >= N_grid[3] || index < 0) {
                        std::cout << "ERROR: index out of range" << std::endl;
                        std::cout << "      index = " << index << std::endl;
                        std::cout << "   N_grid.w = " << N_grid[3] << std::endl;
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
                        int4 ktemp = {i - N_grid[0]/2, j - N_grid[1]/2, k - N_grid[2]/2, index};
                        kvec.push_back(ktemp);
                    }
                }
            }
        }
        std::cout << "    Total number of wave vectors in range: " << kvec.size() << std::endl;
        int num_k_vecs = kvec.size();
        gpuErrchk(cudaMemcpyToSymbol(d_N, &num_k_vecs, sizeof(int)));
        
        int4 *d_k;
        gpuErrchk(cudaMalloc((void **)&d_k, num_k_vecs*sizeof(int4)));
        
        float4 *d_dk3d;
        gpuErrchk(cudaMalloc((void **)&d_dk3d, N_grid[3]*sizeof(float4)));
               
        double *Bk = new double[totBins];
        double *d_Bk;
        gpuErrchk(cudaMalloc((void **)&d_Bk, totBins*sizeof(double)));
        
        unsigned int *Ntri = new unsigned int[totBins];
        unsigned int *d_Ntri;
        gpuErrchk(cudaMalloc((void **)&d_Ntri, totBins*sizeof(unsigned int)));
        
        for (int i = 0; i < totBins; ++i) {
            Bk[i] = 0.0;
            Ntri[i] = 0;
        }
        
        std::cout << "    Copying data to the GPU..." << std::endl;
        gpuErrchk(cudaMemcpy(d_Bk, Bk, totBins*sizeof(double), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_Ntri, Ntri, totBins*sizeof(unsigned int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_dk3d, dk3d, N_grid[3]*sizeof(float4), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_k, kvec.data(), num_k_vecs*sizeof(int4), cudaMemcpyHostToDevice));
        
        dim3 num_gpu_threads(p.geti("num_gpu_threads"), p.geti("num_gpu_threads"));
        dim3 num_blocks(ceil(num_k_vecs/p.getd("num_gpu_threads")),
                        ceil(num_k_vecs/p.getd("num_gpu_threads")));
        
        std::cout << "    Calculating bispectrum..." << std::endl;
        cudaEvent_t begin, end;
        float elapsedTime;
        cudaEventCreate(&begin);
        cudaEventRecord(begin, 0);
        calcBk<<<num_blocks, num_gpu_threads>>>(d_dk3d, d_k, d_Ntri, d_Bk);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        cudaEventCreate(&end);
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsedTime, begin, end);
        std::cout << "    Time to calculate bispectrum: " << elapsedTime << " ms" << std::endl;
        
        int numBlocks = ceil(totBins/1024.0);
        
        cudaEventCreate(&begin);
        cudaEventRecord(begin, 0);
        normBk<<<numBlocks, 1024>>>(d_Ntri, d_Bk, float(nden_gal.nb3w3()), totBins);
        cudaEventCreate(&end);
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsedTime, begin, end);
        std::cout << "Time to normalize bispectrum: " << elapsedTime << " ms" << std::endl;
        
        gpuErrchk(cudaMemcpy(Bk, d_Bk, totBins*sizeof(double), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(Ntri, d_Ntri, totBins*sizeof(unsigned int), cudaMemcpyDeviceToHost));
        
        std::ofstream fout(bk_file.c_str());
        for (int i = 0; i < numBins; ++i) {
            double k1 = k_lim.x + (i + 0.5)*binWidth;
            for (int j = 0; j < numBins; ++j) {
                double k2 = k_lim.x + (j + 0.5)*binWidth;
                for (int k = 0; k < numBins; ++k) {
                    double k3 = k_lim.x + (k + 0.5)*binWidth;
                    int bin = k + numBins*(j + numBins*i);
                    
                    fout << std::setprecision(3) << k1 << " " << k2 << " " << k3 << " " << std::setprecision(15) << Bk[bin] << " " << Ntri[bin] << "\n";
                }
            }
        }
        fout.close();
        
        delete[] Bk;
        delete[] Ntri;
        delete[] dk3d;
        cudaFree(d_k);
        cudaFree(d_dk3d);
        cudaFree(d_Bk);
        cudaFree(d_Ntri);
    }
    
    fftw_destroy_plan(dr2dk);
    delete[] delta;
    
    return 0;
}
