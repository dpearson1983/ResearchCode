#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <cuda.h>
#include <vector_types.h>
#include <harppi.h>
#include <constants.h>

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

// __global__ void get_k_triplets(int4 *triplets, float3 *ks, int N

int main(int argc, char *argv[]) {
    parameters p(argv[1]);
    p.print();
    
    std::ifstream fin;
    std::ofstream fout;
    
    int3 N = {p.geti("Nx"), p.geti("Ny"), p.geti("Nz")};
    double3 L = {p.getd("Lx"), p.getd("Ly"), p.getd("Lz")};
    
    std::vector<double> kx = fft_freq(N.x, L.x);
    std::vector<double> ky = fft_freq(N.y, L.y);
    std::vector<double> kz = fft_freq(N.z, L.z);
    
    double k_min = p.getd("k_min");
    double k_max = p.getd("k_max");
    
    std::vector<double3> ks;
    
    int count = 0;
    for (int i = 0; i < N.x; ++i) {
        for (int j = 0; j < N.y; ++j) {
            for (int k = 0; k < N.z; ++k) {
                double k_mag = sqrt(kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k]);
                
                if (k_mag >= k_min && k_mag < k_max) {
                    count++;
                    double3 k_temp = {kx[i], ky[j], kz[k]};
                    ks.push_back(k_temp);
                }
            }
        }
    }
    
    int count_squared = 0;
    for (int i = 0; i < count; ++i) {
        for (int j = i; j < count; ++j) {
            double3 k3 = {-(ks[i].x + ks[j].x), -(ks[i].y + ks[j].y), -(ks[i].z + ks[j].z)};
            double k3_mag = sqrt(k3.x*k3.x + k3.y*k3.y + k3.z*k3.z);
            if (k3_mag >= k_min && k3_mag < k_max) count_squared++;
        }
    }
    
    std::cout << "Number of frequencies in range [" << k_min << " to " << k_max << "]: " << count << std::endl;
    std::cout << "Number of potential k's to check: " << count_squared << std::endl;
    std::cout << "Memory needed: " << count_squared*sizeof(int4)/1073741824.0 << " GiB" << std::endl;
    
    return 0;
}
