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
#include <fstream>
#include <limits>
#include <cmath>
#include <vector>
#include <gsl/gsl_spline.h>
#include <fftw3.h>
#include <omp.h>
#include <tpods.h>
#include <constants.h>
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
    }
}

void transformDelta(double *dr3d, fftw_complex *dk3d, std::string fftwWisdom, vec3<int> N) {
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
    fftw_import_wisdom_from_filename(fftwWisdom.c_str());
    fftw_plan dr3d2dk3d = fftw_plan_dft_r2c_3d(N.x, N.y, N.z, dr3d, dk3d, FFTW_WISDOM_ONLY);
    fftw_execute(dr3d2dk3d);
    fftw_destroy_plan(dr3d2dk3d);
}

void transformDelta(double *dr3d, std::string fftwWisdom, vec3<int> N) {
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
    fftw_import_wisdom_from_filename(fftwWisdom.c_str());
    fftw_plan dr3d2dk3d = fftw_plan_dft_r2c_3d(N.x, N.y, N.z, dr3d, (fftw_complex *)dr3d,
                                               FFTW_WISDOM_ONLY);
    fftw_execute(dr3d2dk3d);
    fftw_destroy_plan(dr3d2dk3d);
}

template <typename T> void powerspec<T>::binFreq(fftw_complex A_0, int bin, double grid_cor, 
                                                 double shotnoise) {
    powerspec<T>::mono[bin] += (A_0[0]*A_0[0] + A_0[1]*A_0[1] - shotnoise)*grid_cor*grid_cor;
}

template <typename T> void powerspec<T>::freqBin(fftw_complex *A_0, vec3<double> L, 
                                                 vec3<int> N, double shotnoise, 
                                                 vec2<double> k_lim, int flags) {
    double *kx = new double[N.x];
    double *ky = new double[N.y];
    double *kz = new double[N.z];
    fftfreq(kx, N.x, L.x);
    fftfreq(ky, N.y, L.y);
    fftfreq(kz, N.z, L.z);
    int maxInd = N.x*N.y*(N.z/2 + 1);
    double binWidth = (k_lim.y - k_lim.x)/double(powerspec<T>::N);
    vec3<double> dr = {L.x/double(N.x), L.y/double(N.y), L.z/double(N.z)};
    vec3<int> Nstop = {ceil(k_lim.y/kx[1]), ceil(k_lim.y/ky[1]), ceil(k_lim.y/kz[1])};
    std::cout << Nstop.x << " " << Nstop.y << " " << Nstop.z << std::endl;
    for (int i = 0; i < Nstop.x; ++i) {
        int i2 = N.x - i;
        for (int j = 0; j < Nstop.y; ++j) {
            int j2 = N.y - j;
            for (int l = 0; l < Nstop.z; ++l) {
                double k_tot = sqrt(kx[i]*kx[i] + ky[j]*ky[j] + kz[l]*kz[l]);
                
                if (k_tot >= k_lim.x && k_tot <= k_lim.y) {
                    int bin = (k_tot - k_lim.x)/binWidth;
                    if (bin >= powerspec<T>::N) {
                        std::cout << "ERROR: bin out of range." << std::endl;
                        std::cout << "    bin = " << bin << std::endl;
                    }
                    double grid_cor = 1.0;
                    if (flags & pkFlags::GRID_COR) {
                        vec3<double> kvec = {kx[i], ky[j], kz[l]};
                        grid_cor = gridCor(kvec, dr, flags);
                    }
                    
                    if (i != 0 && j != 0) {
                        int index1 = l + (N.z/2 + 1)*(j  + N.y*i );
                        int index2 = l + (N.z/2 + 1)*(j2 + N.y*i );
                        int index3 = l + (N.z/2 + 1)*(j  + N.y*i2);
                        int index4 = l + (N.z/2 + 1)*(j2 + N.y*i2);
                        if (index1 >= maxInd || index2 >= maxInd || index3 >= maxInd || index4 >= maxInd) {
                            std::cout << "ERROR: An index is out of range." << std::endl;
                            std::cout << "    maxInd = " << maxInd << std::endl;
                            std::cout << "    index1 = " << index1 << std::endl;
                            std::cout << "    index2 = " << index2 << std::endl;
                            std::cout << "    index3 = " << index3 << std::endl;
                            std::cout << "    index4 = " << index4 << std::endl;
                        }
                        binFreq(A_0[index1], bin, grid_cor, shotnoise);
                        binFreq(A_0[index2], bin, grid_cor, shotnoise);
                        binFreq(A_0[index3], bin, grid_cor, shotnoise);
                        binFreq(A_0[index4], bin, grid_cor, shotnoise);
                        powerspec<T>::N_k[bin] += 4;
                    } else if (i != 0 && j == 0) {
                        int index1 = l + (N.z/2 + 1)*(j  + N.y*i );
                        int index2 = l + (N.z/2 + 1)*(j  + N.y*i2);
                        if (index1 >= maxInd || index2 >= maxInd) {
                            std::cout << "ERROR: An index is out of range." << std::endl;
                            std::cout << "    maxInd = " << maxInd << std::endl;
                            std::cout << "    index1 = " << index1 << std::endl;
                            std::cout << "    index2 = " << index2 << std::endl;
                        }
                        binFreq(A_0[index1], bin, grid_cor, shotnoise);
                        binFreq(A_0[index2], bin, grid_cor, shotnoise);
                        powerspec<T>::N_k[bin] += 2;
                    } else if (i == 0 && j != 0) {
                        int index1 = l + (N.z/2 + 1)*(j  + N.y*i );
                        int index2 = l + (N.z/2 + 1)*(j2 + N.y*i );
                        if (index1 >= maxInd || index2 >= maxInd) {
                            std::cout << "ERROR: An index is out of range." << std::endl;
                            std::cout << "    maxInd = " << maxInd << std::endl;
                            std::cout << "    index1 = " << index1 << std::endl;
                            std::cout << "    index2 = " << index2 << std::endl;
                        }
                        binFreq(A_0[index1], bin, grid_cor, shotnoise);
                        binFreq(A_0[index2], bin, grid_cor, shotnoise);
                        powerspec<T>::N_k[bin] += 2;
                    } else {
                        int index1 = l + (N.z/2 + 1)*(j  + N.y*i );
                        if (index1 >= maxInd) {
                            std::cout << "ERROR: An index is out of range." << std::endl;
                            std::cout << "    maxInd = " << maxInd << std::endl;
                            std::cout << "    index1 = " << index1 << std::endl;
                        }
                        binFreq(A_0[index1], bin, grid_cor, shotnoise);
                        powerspec<T>::N_k[bin] += 1;
                    }
                }
            }
        }
    }
    
    delete[] kx;
    delete[] ky;
    delete[] kz;
}

template <typename T> powerspec<T>::powerspec() {
    powerspec<T>::N = 1;
}

// The preferred initialization method for the powerspec class. By default, it will set
// up for a calculation of the power spectrum monopole with optional flags specifying 
// whether or not to setup for the calculation of the quadrupole and/or hexadecapole.
template <typename T> powerspec<T>::powerspec(int numKVals, vec2<double> k_lim, int flags) {
    double dk = (k_lim.y - k_lim.x)/double(numKVals);
    powerspec<T>::N = numKVals;
    powerspec<T>::k.reserve(numKVals);
    powerspec<T>::mono.reserve(numKVals);
    powerspec<T>::N_k.reserve(numKVals);
    powerspec<T>::quad.reserve(numKVals);
    powerspec<T>::hexa.reserve(numKVals);
    for (int i = 0; i < numKVals; ++i) {
        powerspec<T>::k.push_back(k_lim.x + (i + 0.5)*dk);
        powerspec<T>::mono.push_back(0.0);
        powerspec<T>::N_k.push_back(0);
        powerspec<T>::quad.push_back(0.0);
        powerspec<T>::hexa.push_back(0.0);
    }
}

template <typename T> void powerspec<T>::calc(double *dr3d, vec3<double> L, 
                                         vec3<int> N_grid, vec2<double> k_lim, 
                                         double shotnoise, std::string fftwWisdom, 
                                         int flags) {
    std::cout << "Generating wisdom..." << std::endl;
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
        
        std::cout << "Transforming delta..." << std::endl;
        transformDelta(dr3d, fftwWisdom, N_grid);
        std::cout << "Binning frequencies..." << std::endl;
        powerspec<T>::freqBin((fftw_complex *)dr3d, L, N_grid, shotnoise, k_lim, flags);
    } else {
        if (flags & pkFlags::HEXA) {
            std::cout << "Hexadecapole not currently implemented." << std::endl;
        }
        
        if (flags & pkFlags::QUAD) {
            // TODO: Figure out the best way of implementing this.
            // calcQuad(double *dr3d, other parameters);
            std::cout << "Quadrupole not currently implemented." << std::endl;
        }
        
        fftw_complex *dk3d = new fftw_complex[N_grid.x*N_grid.y*(N_grid.z/2 + 1)];
        std::cout << "Transforming delta..." << std::endl;
        transformDelta(dr3d, dk3d, fftwWisdom, N_grid);
        // Call binning function
        std::cout << "Binning frequencies..." << std::endl;
        powerspec<T>::freqBin(dk3d, L, N_grid, shotnoise, k_lim, flags);
        delete[] dk3d;
    }
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
        std::cout << gal_nbsqwsq << " " << powerspec<T>::N_k[i] << std::endl;
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

template <typename T> void powerspec<T>::writeFile(std::string file, int flags) {
    std::ofstream fout;
    fout.open(file.c_str(), std::ios::out);
    for (int i = 0; i < powerspec<T>::N; ++i) {
        fout << powerspec<T>::k[i] << " " << powerspec<T>::mono[i] << "\n";
    }
    if (flags & pkFlags::QUAD) {
        for (int i = 0; i < powerspec<T>::N; ++i) {
            fout << powerspec<T>::k[i] << " " << powerspec<T>::quad[i] << "\n";
        }
    }
    if (flags & pkFlags::HEXA) {
        for (int i = 0; i < powerspec<T>::N; ++i) {
            fout << powerspec<T>::k[i] << " " << powerspec<T>::hexa[i] << "\n";
        }
    }
    fout.close();
}

template <typename T> void initSplines(int flags) {
    double dk = powerspec<T>::k[1] - powerspec<T>::k[0];
    std::vector<T> power;
    power.reserve(powerspec<T>::N + 2);
    std::vector<T> k_sp;
    k_sp.reserve(powerspec<T>::N + 2);
    if (flags & pkFlags::MONO) {
        for (int i = 1; i <= powerspec<T>::N; ++i) {
            k_sp[i] = powerspec<T>::k[i - 1];
            power[i] = powerspec<T>::mono[i - 1];
        }
        k_sp[0] = powerspec<T>::k[0] - 0.5*dk;
        k_sp[powerspec<T>::N + 1] = powerspec<T>::k[powerspec<T>::N - 1] + 0.5*dk;
        power[0] = powerspec<T>::mono[0] - 0.5*(powerspec<T>::mono[1] - powerspec<T>::mono[0]);
        power[powerspec<T>::N + 1] = powerspec<T>::mono[powerspec<T>::N - 1] + 
                                0.5*(powerspec<T>::mono[powerspec<T>::N - 1] - powerspec<T>::mono[powerspec<T>::N - 2]);
        powerspec<T>::P_0 = gsl_spline_alloc(gsl_interp_cspline, powerspec<T>::N + 2);
        gsl_spline_init(powerspec<T>::P_0, &k_sp[0], &power[0], powerspec<T>::N + 2);
    }
    
    if (flags & pkFlags::QUAD) {
        for (int i = 1; i <= powerspec<T>::N; ++i) {
            k_sp[i] = powerspec<T>::k[i - 1];
            power[i] = powerspec<T>::quad[i - 1];
        }
        k_sp[0] = powerspec<T>::k[0] - 0.5*dk;
        k_sp[powerspec<T>::N + 1] = powerspec<T>::k[powerspec<T>::N - 1] + 0.5*dk;
        power[0] = powerspec<T>::quad[0] - 0.5*(powerspec<T>::quad[1] - powerspec<T>::quad[0]);
        power[powerspec<T>::N + 1] = powerspec<T>::quad[powerspec<T>::N - 1] + 
                                0.5*(powerspec<T>::quad[powerspec<T>::N - 1] - powerspec<T>::quad[powerspec<T>::N - 2]);
        powerspec<T>::P_2 = gsl_spline_alloc(gsl_interp_cspline, powerspec<T>::N + 2);
        gsl_spline_init(powerspec<T>::P_2, &k_sp[0], &power[0], powerspec<T>::N + 2);
    }
    
    if (flags & pkFlags::HEXA) {
        for (int i = 1; i <= powerspec<T>::N; ++i) {
            k_sp[i] = powerspec<T>::k[i - 1];
            power[i] = powerspec<T>::hexa[i - 1];
        }
        k_sp[0] = powerspec<T>::k[0] - 0.5*dk;
        k_sp[powerspec<T>::N + 1] = powerspec<T>::k[powerspec<T>::N - 1] + 0.5*dk;
        power[0] = powerspec<T>::hexa[0] - 0.5*(powerspec<T>::hexa[1] - powerspec<T>::hexa[0]);
        power[powerspec<T>::N + 1] = powerspec<T>::hexa[powerspec<T>::N - 1] + 
                                0.5*(powerspec<T>::hexa[powerspec<T>::N - 1] - powerspec<T>::hexa[powerspec<T>::N - 2]);
        powerspec<T>::P_4 = gsl_spline_alloc(gsl_interp_cspline, powerspec<T>::N + 2);
        gsl_spline_init(powerspec<T>::P_4, &k_sp[0], &power[0], powerspec<T>::N + 2);
    }
}

template <typename T> T powerspec<T>::get(int index, int flags) {
    if (flags & pkFlags::MONO) return powerspec<T>::mono[index];
    if (flags & pkFlags::QUAD) return powerspec<T>::quad[index];
    if (flags & pkFlags::HEXA) return powerspec<T>::hexa[index];
}

template <typename T> T powerspec<T>::get(double k, int flags) {
    if (flags & pkFlags::MONO) return gsl_spline_eval(powerspec::P_0, k, powerspec<T>::acc_0);
    if (flags & pkFlags::QUAD) return gsl_spline_eval(powerspec::P_2, k, powerspec<T>::acc_2);
    if (flags & pkFlags::HEXA) return gsl_spline_eval(powerspec::P_4, k, powerspec<T>::acc_4);
}

template <typename T> void cleanUp(int flags) {
    if (flags & pkFlags::MONO) {
        gsl_spline_free(powerspec<T>::P_0);
        gsl_interp_accel_free(powerspec<T>::acc_0);
    }
    if (flags & pkFlags::QUAD) {
        gsl_spline_free(powerspec<T>::P_2);
        gsl_interp_accel_free(powerspec<T>::acc_2);
    }
    if (flags & pkFlags::HEXA) {
        gsl_spline_free(powerspec<T>::P_4);
        gsl_interp_accel_free(powerspec<T>::acc_4);
    }
}

template class powerspec<double>;
template class powerspec<float>;
