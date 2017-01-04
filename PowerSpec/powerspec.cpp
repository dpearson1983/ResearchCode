/* powerspec.cpp
 * David W. Pearson
 * January 4, 2017
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
#include <cmath>
#include <vector>
#include <fftw3.h>
#include <omp.h>
#include <tpods.h>
#include "powerspec.h"

void fftfreq(double *kvec, int N, double L) {
    double dk = (2.0*pi)/L;
    for (int i = 0; i <= N/2; ++i)
        kvec[i] = i*dk;
    for (int i = N/2 + 1; i < N; ++i)
        kvec[i] = (i - N)*dk;
}

double gridCor(vec3<double> k, vec3<double> dr, int flags) {
    double sincx = sin(0.5*k.x*dr.x + 1E-17)/(0.5*k.x*dr.x + 1E-17);
    double sincy = sin(0.5*k.y*dr.y + 1E-17)/(0.5*k.y*dr.y + 1E-17);
    double sincz = sin(0.5*k.z*dr.z + 1E-17)/(0.5*k.z*dr.z + 1E-17);
    double prodsinc = sincx*sincy*sincz;
    
    if (flags & pkFlags::NGP) return 1.0/prodsinc;
    if (flags & pkFlags::CIC) return 1.0/(prodsinc*prodsinc);
}

void generateWisdom(vec3<int> N, std::string fftwWisdom, int flags) {
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
    if (flags & pkFlags::OUT_OF_PLACE) {
        double *in = new double[N.x*N.y*N.z];
        fftw_complex *out = new fftw_complex[N.x*N.y*(N.z/2 + 1)];
        fftw_import_wisdom_from_filename(fftwWisdom.c_str());
        fftw_plan test = fftw_plan_dft_r2c_3d(N.x, N.y, N.z, in, out, FFTW_MEASURE);
        fftw_export_wisdom_to_filename(fftwWisdom.c_str());
        fftw_destroy_plan(test);
        delete[] in;
        delete[] out;
    } else {
        double *in = new double[N.x*N.y*2*(N.z/2 + 1)];
        fftw_import_wisdom_from_filename(fftwWisdom.c_str());
        fftw_plan test = fftw_plan_dft_r2c_3d(N.x, N.y, N.z, in, (fftw_complex *)in, 
                                              FFTW_MEASURE);
        fftw_export_wisdom_to_filename(fftwWisdom.c_str());

template <typename T> powerspec<T>::powerspec() {
    powerspec<T>::N = 1;
}

// The preferred initialization method for the powerspec class. By default, it will set
// up for a calculation of the power spectrum monopole with optional flags specifying 
// whether or not to setup for the calculation of the quadrupole and/or hexadecapole.
template <typename T> powerspec<T>::powerspec(int numKVals, vec2<double> k_lim, int flags) {
    double dk = (k_lim.y - k_lim.x)/double(numKVals);
    powerspec<T>::k.reserve(numKVals);
    powerspec<T>::N_k.reserve(numKVals);
    powerspec<T>::mono.reserve(numKVals);
    if (flags & pkFlags::QUAD) powerspec<T>::quad.reserve(numKVals);
    if (flags & pkFlags::HEXA) powerspec<T>::hexa.reserve(numKVals);
    for (int i = 0; i < numKVals; ++i) {
        powerspec<T>::k[i] = k_lim.x + (i + 0.5)*dk;
        powerspec<T>::mono[i] = 0.0;
        powerspec<T>::N_k[i] = 0;
        if (flags & pkFlags::QUAD) powerspec<T>::quad[i] = 0.0;
        if (flags & pkFlags::HEXA) powerspec<T>::hexa[i] = 0.0;
    }
}

template <typename T> void powerspec<T>::calc(double *dr3d, vec3<double> L, 
                                         vec3<int> N_grid, vec2<double> k_lim, 
                                         double shotnoise, std::string fftwWisdom, 
                                         int flags) {
    generateWisdom(N_grid, fftwWisdom, flags);
    bool inPlace = true;
    if (flags & pkFlags::OUT_OF_PLACE) inPlace = false;
    
    if (inPlace) {
        if (flags & pkFlags::HEXA) {
            std::cout << "Hexadecapole not currently implemented." << std::endl;
        }
        
        if (flags & pkFlags::QUAD) {
            // TODO: Figure out the best way of implementing this.
            // calcQuad(double *dr3d, other parameters);
            std::cout << "Quadrupole not currently implemented." << std::endl;
        }
        
        fftw_init_threads();
        fftw_import_wisdom_from_filename(fftwWisdom.c_str());
        fftw_plan_with_nthreads(omp_get_max_threads());
        fftw_plan dr3d2dk3d = fftw_plan_dft_r2c_3d(N_grid.x, N_grid.y, N_grid.z, dr3d,
                                                   (fftw_complex *)dr3d, FFTW_WISDOM_ONLY);
        
        fftw_execute(dr3d2dk3d);
        fftw_destroy_plan(dr3d2dk3d);
        
        // Call binning function
}

template <typename T> void powerspec<T>::disc_cor(std::string file, int flags) {
    double k;
    std::ifstream fin;
    fin.open(file.c_str(), std::ios::in);
    for (int i = 0; i < powerspec<T>::N; ++i) {
        fin >> k;
        if (flags & pkFlags::MONO) {
            double mono_cor;
            fin >> mono_cor;
            powerspec<T>::mono[i] -= mono_cor;
        }
        if (flags & pkFlags::QUAD) {
            double quad_cor;
            fin >> quad_cor;
            powerspec<T>::quad[i] -= quad_cor;
        }
        if (flags & pkFlags::HEXA) {
            double hexa_cor;
            fin >> hexa_cor;
            powerspec<T>::hexa[i] -= hexa_cor;
        }
    }
    fin.close();
}

template <typename T> void powerspec<T>::norm(double gal_nbsqwsq, int flags) {
    for (int i = 0; i < powerspec<T>::N; ++i) {
        powerspec<T>::mono[i] /= (gal_nbsqwsq*powerspec<T>::N_k[i]);
        if (flags & pkFlags::QUAD) {
            powerspec<T>::quad[i] /= (gal_nbsqwsq*powerspec<T>::N_k[i]);
        }
        if (flags & pkFlags::HEXA) {
            powerspec<T>::hexa[i] /= (gal_nbsqwsq*powerspec<T>::N_k[i]);
        }
    }
}

template <typename T> void powerspec<T>::print(int flags) {
    int width = std::numeric_limits<T>::digits10 + 6;
    if (flags & pkFlags::HEADER) {
        std::cout << std::setw(10) << "k";
        if (flags &pkFlags::MONO) std::cout << std::setw(width) << "P_0(k)";
        if (flags &pkFlags::QUAD) std::cout << std::setw(width) << "P_2(k)";
        if (flags &pkFlags::HEXA) std::cout << std::setw(width) << "P_4(k)";
        std::cout << std::endl;
    }
    for (int i = 0; i < powerspec<T>::N; ++i) {
        std::cout << std::setw(10) << powerspec<T>::k[i];
        if (flags &pkFlags::MONO) std::cout << std::setw(width) << powerspec<T>::mono[i];
        if (flags &pkFlags::QUAD) std::cout << std::setw(width) << powerspec<T>::quad[i];
        if (flags &pkFlags::HEXA) std::cout << std::setw(width) << powerspec<T>::hexa[i];
        std::cout << std::endl;
    }
}

template <typename T> void powerspec<T>::print() {
    powerspec<T>::print(pkFlags::HEADER|pkFlags::MONO);
}

template <typename T> void write(std::string file, int flags) {
    
}

template class powerspec<double>;
template class powerspec<float>;
