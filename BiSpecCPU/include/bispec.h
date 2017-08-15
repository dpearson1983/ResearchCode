#ifndef _BISPEC_H_
#define _BISPEC_H_

#include <vector>
#include <fftw3.h>
#include <tpods.h>

void get_shell(std::vector<fftw_complex> &dr3d, std::vector<fftw_complex> &shell, double k, 
               double bin_width, std::vector<double> &kx, std::vector<double> &ky, 
               std::vector<double> &kz);

#endif
