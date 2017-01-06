/* bispec.cpp
 * David W. Pearson
 * January 5, 2017
 * 
 * LICENSE: GPL v3
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>. 
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <fftw3.h>
#include <omp.h>
#include <tpods.h>
#include <constants.h>
#include "bispec.h"

template <typename T> void bispec<T>::get_shell(double *dk3d, double *dk3d_shell, int N_p, 
                                                int kBin) {
    for (int i = 0; i < N_p; ++i) {
        if (bispec<T>::kbins[i] == kBin) dk3d_shell[i] = dk3d[i];
        else dk3d_shell[i] = 0.0;
    }
}

template <typename T> void bispec<T>::get_shell(fftw_complex *dk3d, double *dk3d_shell, 
                                                vec3<int. N_grid, int kBin) {
    for (int i = 0; i < N_grid.x; ++i) {
        for (int j = 0; j < N_grid.y; ++j) {
            for (int k = 0; k <= N_grid.z/2; ++k) {
                int index1 = k + (N_grid.z/2 + 1)*(j + N_grid.y*i);
                int index2 = (2*k    ) + 2*(N_grid.z/2 + 1)*(j + N_grid.y*i);
                int index3 = (2*k + 1) + 2*(N_grid.z/2 + 1)*(j + N_grid.y*i);
                
                if (bispec<T>::kbins[index2] == kBin && bispec<T>::kbins[index3] == kBin) {
                    dk3d_shell[index2] = dk3d[index1][0];
                    dk3d_shell[index3] = dk3d[index1][1];
                } else {
                    dk3d_shell[index2] = 0.0;
                    dk3d_shell[index3] = 0.0;
                }
            }
        }
    }
}

template <typename T> bispec<T>::getks(int numKVals, vec2<double> k_lim) {
    double dk = (k_lim.y - l_lim.x)/double(numKVals);
    int count = 0;
    for (int i = 0; i < numKVals; ++i) {
        double ki = k_lim.x + (i + 0.5)*dk;
        for (int j = 0; j < numKVals; ++j) {
            double kj = k_lim.x + (j + 0.5)*dk;
            for (int l = 0; l < numKVals; ++l) {
                double kl = k_lim.x + (l + 0.5)*dk;
                vec3<double> temp = {ki, kj, kl};
                bispec<T>::ks.push_back(temp);
                bispec<T>::val.push_back(-pi);
            }
        }
    }
}

template <typename T> bispec<T>::bispec() {
    bispec<T>::N = 1;
}

template <typename T> bispec<T>::bispec(int numKVals, vec2<double> k_lim, int flags) {
    bispec<T>::val.reserve(numKVals*numKVals*numKVals);
    bispec<T>::ks.reserve(numKVals*numKVals*numKVals);
    bispec<T>::getks(numKVals, k_lim);
}

template <typename T> bispec<T>::calc(double *dk3d, vec3<int> N_grid, std::string fftwWisdom) {
    int N_p = N_grid.x*N_grid.y*2*(N_grid.z/2 + 1);
    
    double *dk_i = new double[N_p];
    double *dk_j = new double[N_p];
    double *dk_l = new double[N_p];
    
    fftw_init_threads();
    fftw_plan_with_nthread(omp_get_max_threads());
    fftw_import_wisdom_from_filename(fftwWisdom.c_str());
    fftw_plan dki2dri = fftw_plan_dft_c2r_3d(N_grid.x, N_grid.y, N_grid.z, 
                                             (fftw_complex *)dk_i, dk_i, FFTW_MEASURE);
    fftw_plan dkj2drj = fftw_plan_dft_c2r_3d(N_grid.x, N_grid.y, N_grid.z, 
                                             (fftw_complex *)dk_j, dk_j, FFTW_MEASURE);
    fftw_plan dkl2drl = fftw_plan_dft_c2r_3d(N_grid.x, N_grid.y, N_grid.z, 
                                             (fftw_complex *)dk_l, dk_l, FFTW_MEASURE);
    fftw_export_wisdom_to_filename(fftwWisdom.c_str());
    
    
}

template <typename T> bispec<T>::calc(fftw_complex *dk3d, vec3<int> N_grid, 
                                      std::string fftwWisdom) {
    int N_p = N_grid.x*N_grid.y*2*(N_grid.z/2 + 1);
    int N_k = N_grid.x*N_grid.y*(N_grid.z/2 + 1);
    
    double *dk_i = new double[N_p];
    double *dk_j = new double[N_p];
    double *dk_l = new double[N_p];
    
    fftw_init_threads();
    fftw_plan_with_nthread(omp_get_max_threads());
    fftw_import_wisdom_from_filename(fftwWisdom.c_str());
    fftw_plan dki2dri = fftw_plan_dft_c2r_3d(N_grid.x, N_grid.y, N_grid.z, 
                                             (fftw_complex *)dk_i, dk_i, FFTW_MEASURE);
    fftw_plan dkj2drj = fftw_plan_dft_c2r_3d(N_grid.x, N_grid.y, N_grid.z, 
                                             (fftw_complex *)dk_j, dk_j, FFTW_MEASURE);
    fftw_plan dkl2drl = fftw_plan_dft_c2r_3d(N_grid.x, N_grid.y, N_grid.z, 
                                             (fftw_complex *)dk_l, dk_l, FFTW_MEASURE);
    fftw_export_wisdom_to_filename(fftwWisdom.c_str());
    
}

template <typename T> bispec<T>::mapdrs(vec3<int> N_grid, int flags) {
    
}

template <typename T> bispec<T>::mapkbins(vec3<int> N_grid, vec2<double> k_lim, int flags) {
    
}

template <typename T> bispec<T>::setdrs(int index) {
    bispec<T>::drs.push_back(index);
}

template <typename T> bispec<T>::norm() {
    
}

template <typename T> bispec<T>::print(int flags) {
    
}

template <typename T> bispec<T>::write(std::string file, int flags) {
    
}

template class bispec<double>
template class bispec<float>
