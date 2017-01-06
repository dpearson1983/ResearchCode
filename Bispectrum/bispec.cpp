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

template <typename T> bispec<T>::calc(fftw_complex *dk3d) {
    
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
