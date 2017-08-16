#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <omp.h>
#include <fftw3.h>
#include <harppi.h>
#include <tpods.h>
#include <constants.h>
#include "include/bispec.h"

std::vector<double> fft_freq(int N, double L) {
    std::vector<double> k;
    k.reserve(N);
    double dk = (2.0*pi)/L;
    for (int i = 0; i <= N/2; ++i)
        k.push_back(i*dk);
    for (int i = N/2 + 1; i < N; ++i)
        k.push_back((i - N)*dk);
    return k;
}

int main(int argc, char *argv[]) {
    parameters p(argv[1]);
    p.print();
    
    std::ifstream fin;
    std::ofstream fout;
    
    vec3<int> N = {p.geti("Nx"), p.geti("Ny"), p.geti("Nz")};
    vec3<double> L = {p.getd("Lx"), p.getd("Ly"), p.getd("Lz")};
    
    std::vector<double> kx = fft_freq(N.x, L.x);
    std::vector<double> ky = fft_freq(N.y, L.y);
    std::vector<double> kz = fft_freq(N.z, L.z);
    
    std::vector<fftw_complex> delta(N.x*N.y*N.z);
    std::vector<fftw_complex> shell(N.x*N.y*N.z);
    std::vector<fftw_complex> shell2(N.x*N.y*N.z);
    
    fftw_init_threads();
    
    fftw_import_wisdom_from_filename(p.gets("wisdom_file").c_str());
    fftw_plan_with_nthreads(omp_get_max_threads());
    fftw_plan shellTrans = fftw_plan_dft_3d(N.x, N.y, N.z, shell.data(), shell.data(), FFTW_BACKWARD, FFTW_MEASURE);
    fftw_export_wisdom_to_filename(p.gets("wisdom_file").c_str());
    
    double k_min = p.getd("k_val") - 0.5*p.getd("bin_width");
    double k_max = p.getd("k_val") + 0.5*p.getd("bin_width");
    
    std::cout << "Filling cube..." << std::endl;
    double start = omp_get_wtime();
//     #pragma omp parallel for
    for (int i = 0; i < N.x; ++i) {
        for (int j = 0; j < N.y; ++j) {
            for (int k = 0; k < N.z; ++k) {
                double k_mag = sqrt(kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k]);
                int index = k + N.z*(j + N.y*i);
                
                delta[index][0] = k_mag;
                delta[index][1] = k_mag;
            }
        }
    }
    std::cout << "Time to fill cube: " << omp_get_wtime() - start << " s" << std::endl;
    
    std::cout << "Getting shell..." << std::endl;
    start = omp_get_wtime();
    get_shell(delta, shell, p.getd("k_val"), p.getd("bin_width"), kx, ky, kz);
    std::cout << "Time to get shell: " << omp_get_wtime() - start << " s" << std::endl;
    
    std::cout << "Writing binary shell file..." << std::endl;
    start = omp_get_wtime();
    fout.open(p.gets("shell_file"), std::ios::out|std::ios::binary);
    fout.write((char *)shell.data(), N.x*N.y*N.z*sizeof(fftw_complex));
    fout.close();
    std::cout << "Time to write file: " << omp_get_wtime() - start << " s" << std::endl;
    
    std::cout << "Reading binary shell file..." << std::endl;
    start = omp_get_wtime();
    fin.open(p.gets("shell_file"), std::ios::in|std::ios::binary);
    fin.read((char *)shell2.data(), N.x*N.y*N.z*sizeof(fftw_complex));
    fin.close();
    std::cout << "Time to read file: " << omp_get_wtime() - start << " s" << std::endl;
    
    std::cout << "Transforming shell..." << std::endl;
    start = omp_get_wtime();
    fftw_execute(shellTrans);
    std::cout << "Time to transform..." << omp_get_wtime() - start << " s" << std::endl;
    
    int N_tot = N.x*N.y*N.z;
    std::cout << "Simulating grid multiplications..." << std::endl;
    start = omp_get_wtime();
    double sum = {0.0};
    for (int i = 0; i < N_tot; ++i) {
        sum += delta[i][0]*shell[i][0]*shell2[i][0] - delta[i][0]*shell[i][1]*shell2[i][1] - delta[i][1]*shell[i][0]*shell2[i][1] - delta[i][1]*shell[i][1]*shell2[i][2];
    }
    std::cout << "Time for sum: " << omp_get_wtime() - start << " s" << std::endl;
    
    return 0;
}
