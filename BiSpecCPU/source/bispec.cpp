#include <vector>
#include <cmath>
#include <fftw3.h>
#include <tpods.h>

void get_shell(std::vector<fftw_complex> &delta, std::vector<fftw_complex> &shell, double k, 
               double bin_width, std::vector<double> &kx, std::vector<double> &ky, 
               std::vector<double> &kz) {
    vec3<int> N = {kx.size(), ky.size(), kz.size()};
    double k_min = k - 0.5*bin_width;
    double k_max = k + 0.5*bin_width;
    
    #pragma omp parallel for
    for (int i = 0; i < N.x; ++i) {
        for (int j = 0; j < N.y; ++j) {
            for (int k = 0; k < N.z; ++k) {
                double k_mag = sqrt(kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k]);
                int index = k + N.z*(j + N.y*i);
                if (k_mag >= k_min && k_mag < k_max) {
                    shell[index][0] = delta[index][0];
                    shell[index][1] = delta[index][1];
                } else {
                    shell[index][0] = 0.0;
                    shell[index][1] = 0.0;
                }
            }
        }
    }
}
