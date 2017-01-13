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
#include <cmath>
#include <vector>
#include <string>
#include <limits>
#include <fftw3.h>
#include <omp.h>
#include <tpods.h>
#include <constants.h>
#include <powerspec.h>
#include "bispec.h"

template <typename T> std::vector<T> bispec<T>::fftFreq(int N, double L) {
    double dk = (2.0*pi)/L;
    std::vector<T> k;
    k.reserve(N);
    for (int i = 0; i <= N/2; ++i)
        k[i] = i*dk;
    for (int i = N/2 + 1; i < N; ++i)
        k[i] = (i - N)*dk;
    return k;
}

template <typename T> void bispec<T>::get_shell(double *dk3d, double *dk3d_shell, 
                                                vec3<int> N_grid, int kBin) {
    int N_p = N_grid.x*N_grid.y*2*(N_grid.z/2 + 1);
    for (int i = 0; i < N_p; ++i) {
        if (bispec<T>::kbins[i] == kBin) dk3d_shell[i] = dk3d[i];
        else dk3d_shell[i] = 0.0;
    }
}

template <typename T> void bispec<T>::get_shell(fftw_complex *dk3d, double *dk3d_shell, 
                                                vec3<int> N_grid, int kBin) {
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

template <typename T> void bispec<T>::getks(int numKVals, vec2<double> k_lim) {
    double dk = (k_lim.y - k_lim.x)/double(numKVals);
    int count = 0;
    for (int i = 0; i < numKVals; ++i) {
        double ki = k_lim.x + (i + 0.5)*dk;
        for (int j = 0; j < numKVals; ++j) {
            double kj = k_lim.x + (j + 0.5)*dk;
            for (int l = 0; l < numKVals; ++l) {
                double kl = k_lim.x + (l + 0.5)*dk;
                vec3<T> temp = {ki, kj, kl};
                bispec<T>::ks.push_back(temp);
                bispec<T>::val.push_back(-pi);
            }
        }
    }
}

template <typename T> void bispec<T>::getks(vec3<double> L, vec3<int> N_grid, vec2<double> k_lim) {
    std::vector<T> kx = fftFreq(N_grid.x, L.x);
    std::vector<T> ky = fftFreq(N_grid.y, L.y);
    std::vector<T> kz = fftFreq(N_grid.z, L.z);
    
    for (int i = 0; i < N_grid.x; ++i) {
        for (int j = 0; j < N_grid.y; ++j) {
            for (int l = 0; l < N_grid.z; ++l) {
                double k = sqrt(kx[i]*kx[i] + ky[j]*ky[j] + kz[l]*kz[l]);
                
                if (k >= k_lim.x && k <= k_lim.y) {
                    int index = l + (N_grid.z/2 + 1)*(j + N_grid.y*i);
                    vec3<T> kvec = {kx[i], ky[j], kz[l]};
                    bispec<T>::ks.push_back(kvec);
                    bispec<T>::kbins.push_back(index);
                }
            }
        }
    }
}

template <typename T> void bispec<T>::setdrs(std::vector<int> vals) {
    int num = vals.size();
    bispec<T>::drs.reserve(num);
    for (int i = 0; i < num; ++i)
        bispec<T>::drs[i] = vals[i];
}    

template <typename T> void bispec<T>::mapdrs(vec3<int> N_grid, int flags) {
    int N_r = N_grid.x*N_grid.y*N_grid.z;
    bispec<T>::drs.reserve(N_r);
    if (flags & bkFlags::IN_PLACE) {
        for (int i = 0; i < N_grid.x; ++i) {
            for (int j = 0; j < N_grid.y; ++j) {
                for (int l = 0; l < N_grid.z; ++l) {
                    int index = l + 2*(N_grid.z/2 + 1)*(j + N_grid.y*i);
                    bispec<T>::drs.push_back(index);
                }
            }
        }
    }
    
    if (flags & bkFlags::OUT_OF_PLACE) {
        for (int i = 0; i < N_r; ++i) {
            bispec<T>::drs.push_back(i);
        }
    }
}

template <typename T> void bispec<T>::mapkbins(vec3<int> N_grid, vec3<double> dk, 
                                          vec2<double> k_lim, int flags) {
    double delta_k = (k_lim.y - k_lim.x)/double(bispec<T>::N);
    int N_p = N_grid.x*N_grid.y*2*(N_grid.z/2 + 1);
    bispec<T>::kbins.reserve(N_p);
    for (int i = 0; i < N_grid.x; ++i) {
        double kx = i*dk.x;
        for (int j = 0; j < N_grid.y; ++j) {
            double ky = j*dk.y;
            for (int k = 0; k <= N_grid.z/2; ++k) {
                double kz = k*dk.z;
                double k_tot = sqrt(kx*kx + ky*ky + kz*kz);
                int kBin = (k_tot - k_lim.x)/delta_k;
                bispec<T>::kbins[(2*k    ) + 2*(N_grid.z/2 + 1)*(j + N_grid.y*i)] = kBin;
                bispec<T>::kbins[(2*k + 1) + 2*(N_grid.z/2 + 1)*(j + N_grid.y*i)] = kBin;
            }
        }
    }
}

template <typename T> bispec<T>::bispec() {
    bispec<T>::N = 1;
}

template <typename T> bispec<T>::bispec(int numKVals, vec3<double> L, vec3<int> N_grid, 
                                        vec2<double> k_lim, int flags) {
    vec3<double> dk = {(2.0*pi)/L.x, (2.0*pi)/L.y, (2.0*pi)/L.z};
    bispec<T>::val.reserve(numKVals*numKVals*numKVals);
    bispec<T>::ks.reserve(numKVals*numKVals*numKVals);
    if (flags & bkFlags::BRUTE_CALC) {
        bispec<T>::getks(L, N_grid, k_lim);
        bispec<T>::N = bispec<T>::ks.size();
    } else {
        bispec<T>::getks(numKVals, k_lim);
        bispec<T>::N = numKVals;
        bispec<T>::mapdrs(N_grid, flags);
        bispec<T>::mapkbins(N_grid, dk, k_lim, flags);
    }
}

template <typename T> bispec<T>::bispec(int numKVals, vec3<double> L, vec3<int> N_grid, 
                                        vec2<double> k_lim, std::vector<int> vals, 
                                        int flags) {
    vec3<double> dk = {(2.0*pi)/L.x, (2.0*pi)/L.y, (2.0*pi)/L.z};
    bispec<T>::val.reserve(numKVals*numKVals*numKVals);
    bispec<T>::ks.reserve(numKVals*numKVals*numKVals);
    bispec<T>::getks(numKVals, k_lim);
    bispec<T>::N = numKVals;
    bispec<T>::setdrs(vals);
    bispec<T>::mapkbins(N_grid, dk, k_lim, flags);
}

template <typename T> void bispec<T>::calc(double *dk3d, vec3<int> N_grid, 
                                           vec2<double> k_lim, double V_f, 
                                           std::string fftwWisdom, powerspec<T> Pk, 
                                           double nbar, double norm) {
    int N_p = N_grid.x*N_grid.y*2*(N_grid.z/2 + 1);
    int N_r = N_grid.x*N_grid.y*N_grid.z;
    double Ncube = double(N_r)*double(N_r)*double(N_r);
    double delta_k = (k_lim.y - k_lim.x)/double(bispec<T>::N);
    int numdrs = bispec<T>::drs.capacity();
    std::cout << "Number of dr grid points: " << numdrs << std::endl;
    
    double *dk_i = new double[N_p];
    double *dk_j = new double[N_p];
    double *dk_l = new double[N_p];
    
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
    fftw_import_wisdom_from_filename(fftwWisdom.c_str());
    fftw_plan dki2dri = fftw_plan_dft_c2r_3d(N_grid.x, N_grid.y, N_grid.z, 
                                             (fftw_complex *)dk_i, dk_i, FFTW_MEASURE);
    fftw_plan dkj2drj = fftw_plan_dft_c2r_3d(N_grid.x, N_grid.y, N_grid.z, 
                                             (fftw_complex *)dk_j, dk_j, FFTW_MEASURE);
    fftw_plan dkl2drl = fftw_plan_dft_c2r_3d(N_grid.x, N_grid.y, N_grid.z, 
                                             (fftw_complex *)dk_l, dk_l, FFTW_MEASURE);
    fftw_export_wisdom_to_filename(fftwWisdom.c_str());
    
    //nbar = 1.0/nbar;
    
    double coeff = 1.0/(8.0*pi*pi*pi*pi);
    double delta_k_cube = delta_k*delta_k*delta_k;
    for (int i = 0; i < bispec<T>::N; ++i) {
        double k_i = k_lim.x + (i + 0.5)*delta_k;
        int k_iBin = (k_i - k_lim.x)/delta_k;
        get_shell(dk3d, dk_i, N_grid, k_iBin);
        fftw_execute(dki2dri);
        for (int j = i; j < bispec<T>::N; ++j) {
            double k_j = k_lim.x + (j + 0.5)*delta_k;
            if (j != i) {
                int k_jBin = (k_j - k_lim.x)/delta_k;
                get_shell(dk3d, dk_j, N_grid, k_jBin);
                fftw_execute(dkj2drj);
            } else {
                #pragma omp parallel for
                for (int q = 0; q < N_p; ++q)
                    dk_j[q] = dk_i[q];
            }
            //int stop = (k_i + k_j - k_lim.x)/delta_k;
            //stop = std::min(stop, bispec<T>::N);
            for (int l = j; l < bispec<T>::N; ++l) {
                double k_l = k_lim.x + (l + 0.5)*delta_k;
                if (l != j) {
                    int k_lBin = (k_l - k_lim.x)/delta_k;
                    get_shell(dk3d, dk_l, N_grid, k_lBin);
                    fftw_execute(dkl2drl);
                } else {
                    #pragma omp parallel for
                    for (int q = 0; q < N_p; ++q)
                        dk_l[q] = dk_j[q];
                }
                
                double sum = 0.0;
                #pragma omp parallel for reduction(+:sum)
                for (int q = 0; q < numdrs; ++q) {
                    sum += dk_i[bispec<T>::drs[q]]
                          *dk_j[bispec<T>::drs[q]]
                          *dk_l[bispec<T>::drs[q]];
                }
                double V = V_f/(coeff*k_i*k_j*k_l*delta_k_cube);
                double shotnoise = (Pk.get(i, pkFlags::MONO) + Pk.get(j, pkFlags::MONO) + 
                       Pk.get(l, pkFlags::MONO))/nbar + 1.0/(nbar*nbar);
                sum *= V;
                sum /= (Ncube*norm);
                //sum -= shotnoise;
                bispec<T>::val[l + bispec<T>::N*(j + bispec<T>::N*i)] = sum;
                bispec<T>::val[l + bispec<T>::N*(i + bispec<T>::N*j)] = sum;
                bispec<T>::val[j + bispec<T>::N*(l + bispec<T>::N*i)] = sum;
                bispec<T>::val[j + bispec<T>::N*(i + bispec<T>::N*l)] = sum;
                bispec<T>::val[i + bispec<T>::N*(j + bispec<T>::N*l)] = sum;
                bispec<T>::val[i + bispec<T>::N*(l + bispec<T>::N*j)] = sum;
                
                std::cout << k_i << " " << k_j << " " << k_l << " " << sum << " " << shotnoise << std::endl;
            }
        }
    }
    
    fftw_destroy_plan(dki2dri);
    fftw_destroy_plan(dkj2drj);
    fftw_destroy_plan(dkl2drl);
}

template <typename T> void bispec<T>::calc(fftw_complex *dk3d, vec3<int> N_grid, 
                                      vec2<double> k_lim, double V_f, 
                                      std::string fftwWisdom, powerspec<T> Pk, double nbar, double norm) {
    int N_p = N_grid.x*N_grid.y*2*(N_grid.z/2 + 1);
    int N_r = N_grid.x*N_grid.y*N_grid.z;
    double Ncube = double(N_r)*double(N_r)*double(N_r);
    double delta_k = (k_lim.y - k_lim.x)/double(bispec<T>::N);
    int numdrs = bispec<T>::drs.size();
    
    double *dk_i = new double[N_p];
    double *dk_j = new double[N_p];
    double *dk_l = new double[N_p];
    
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
    fftw_import_wisdom_from_filename(fftwWisdom.c_str());
    fftw_plan dki2dri = fftw_plan_dft_c2r_3d(N_grid.x, N_grid.y, N_grid.z, 
                                             (fftw_complex *)dk_i, dk_i, FFTW_MEASURE);
    fftw_plan dkj2drj = fftw_plan_dft_c2r_3d(N_grid.x, N_grid.y, N_grid.z, 
                                             (fftw_complex *)dk_j, dk_j, FFTW_MEASURE);
    fftw_plan dkl2drl = fftw_plan_dft_c2r_3d(N_grid.x, N_grid.y, N_grid.z, 
                                             (fftw_complex *)dk_l, dk_l, FFTW_MEASURE);
    fftw_export_wisdom_to_filename(fftwWisdom.c_str());
    
    double coeff = 1.0/(8.0*pi*pi*pi*pi);
    double delta_k_cube = delta_k*delta_k*delta_k;
    for (int i = 0; i < bispec<T>::N; ++i) {
        double k_i = k_lim.x + (i + 0.5)*delta_k;
        int k_iBin = (k_i - k_lim.x)/delta_k;
        get_shell(dk3d, dk_i, N_grid, k_iBin);
        fftw_execute(dki2dri);
        for (int j = i; j < bispec<T>::N; ++j) {
            double k_j = k_lim.x + (j + 0.5)*delta_k;
            if (j != i) {
                int k_jBin = (k_j - k_lim.x)/delta_k;
                get_shell(dk3d, dk_j, N_grid, k_jBin);
                fftw_execute(dkj2drj);
            } else {
                #pragma omp parallel for
                for (int q = 0; q < N_p; ++q)
                    dk_j[q] = dk_i[q];
            }
            int stop = (k_i + k_j - k_lim.x)/delta_k;
            stop = std::min(stop, bispec<T>::N);
            for (int l = j; l < stop; ++l) {
                double k_l = k_lim.x + (l + 0.5)*delta_k;
                if (l != j) {
                    int k_lBin = (k_l - k_lim.x)/delta_k;
                    get_shell(dk3d, dk_l, N_grid, k_lBin);
                    fftw_execute(dkl2drl);
                } else {
                    #pragma omp parallel for
                    for (int q = 0; q < N_p; ++q)
                        dk_l[q] = dk_j[q];
                }
                
                double sum = 0.0;
                #pragma omp parallel for reduction(+:sum)
                for (int q = 0; q < numdrs; ++q) {
                    sum += dk_i[bispec<T>::drs[q]]
                          *dk_j[bispec<T>::drs[q]]
                          *dk_l[bispec<T>::drs[q]];
                }
                double V = V_f/(coeff*k_i*k_j*k_l*delta_k_cube);
                sum /= (Ncube*norm);
                sum *= V;
                sum -= ((Pk.get(i, pkFlags::MONO) + Pk.get(j, pkFlags::MONO) + 
                       Pk.get(l, pkFlags::MONO))/nbar + 1.0/(nbar*nbar));
                bispec<T>::val[l + bispec<T>::N*(j + bispec<T>::N*i)] = sum;
                bispec<T>::val[l + bispec<T>::N*(i + bispec<T>::N*j)] = sum;
                bispec<T>::val[j + bispec<T>::N*(l + bispec<T>::N*i)] = sum;
                bispec<T>::val[j + bispec<T>::N*(i + bispec<T>::N*l)] = sum;
                bispec<T>::val[i + bispec<T>::N*(j + bispec<T>::N*l)] = sum;
                bispec<T>::val[i + bispec<T>::N*(l + bispec<T>::N*j)] = sum;
            }
        }
    }
    
    fftw_destroy_plan(dki2dri);
    fftw_destroy_plan(dkj2drj);
    fftw_destroy_plan(dkl2drl);
}

// This function will be a brute force calculation of the bispectrum. This is being included as a 
// way of getting data to test the other calculation algorithms which seem not to be working at this
// time. In order to make the algorithm as optimal as possible, the funtion will need:
//      fftw_complex *dk3d - The delta(k) box
//      vec3<T> ks - The k vectors associated with grid points of interest in dk3d
//      vec3<int> drs - The indices of the grid points of interest
// As this function will use data members meant for the other calculation methods in a different maner,
// it will be necessary to call a custom initialization for use with this function.
template <typename T> void bispec<T>::bruteCalc(fftw_complex *dk3d, vec3<double> L, vec3<int> N_grid, 
                                                vec2<double> k_lim, double nbar, 
                                                double norm) {
    std::cout << "This is a work in progress." << std::endl;
    
    std::ofstream fout;
    
    std::cout << "ks.size() = " << bispec<T>::ks.size() << std::endl;
    std::cout << "ks.capacity() = " << bispec<T>::ks.capacity() << std::endl;
    
    int n = bispec<T>::kbins.size();
    
    int numRepeats = 0;
    
    std::vector<vec3<T>> k3s;
   
    for (int i = 0; i < n; ++i) {
        for(int j = i; j < n; ++j) {
            vec3<T> k3 = {(-1.0)*bispec<T>::ks[i].x + (-1.0)*bispec<T>::ks[j].x,
                          (-1.0)*bispec<T>::ks[i].y + (-1.0)*bispec<T>::ks[j].y,
                          (-1.0)*bispec<T>::ks[i].z + (-1.0)*bispec<T>::ks[j].z};
            if (k3.x == 0.0) k3.x = 0;
            if (k3.y == 0.0) k3.y = 0;
            if (k3.z == 0.0) k3.z = 0;
            k3s.push_back(k3);
//             int num = k3s.size();
//             for (int l = 0; l < num; ++l) {
//                 if (k3s[i] == k3) ++numRepeats;
//             }
        }
    }
    
    std::cout << "Repeated k3's: " << numRepeats << std::endl;
}

template <typename T> void bispec<T>::norm() {
    std::cout << "bispec<T>::norm - This function currently does nothing.\n";
    std::cout << "    Sorry you wasted your time by calling it...\n";
}

template <typename T> void bispec<T>::print() {
    int N_tot = bispec<T>::N*bispec<T>::N*bispec<T>::N;
    
    std::cout.precision(std::numeric_limits<T>::digits10);
    for (int i = 0; i < N_tot; ++i) {
        if (bispec<T>::val[i] != -pi) {
            std::cout << bispec<T>::ks[i].x << " " << bispec<T>::ks[i].y << " " << 
                    bispec<T>::ks[i].z << " " << bispec<T>::val[i] << "\n";
        }
    }
}

template <typename T> void bispec<T>::write(std::string file) {
    int N_tot = bispec<T>::N*bispec<T>::N*bispec<T>::N;
    
    std::ofstream fout;
    fout.open(file.c_str(), std::ios::out);
    fout.precision(std::numeric_limits<T>::digits10);
    for (int i = 0; i < N_tot; ++i) {
        if (bispec<T>::val[i] != -pi) {
            fout << bispec<T>::ks[i].x << " " << bispec<T>::ks[i].y << " " << 
                    bispec<T>::ks[i].z << " " << bispec<T>::val[i] << "\n";
        }
    }
    fout.close();
}

template class bispec<double>;
template class bispec<float>;
