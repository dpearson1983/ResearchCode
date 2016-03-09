/*
 * LNKNLogs-2Tracer.cpp
 * David W. Pearson & Lado Samushia
 * 2/19/2016
 * 
 * This code will generate lognormal mock for two tracers in various redshift bins. It
 * requires a CAMB file for each redshift bin and a file containing the central redshift of
 * each bin (should match the redshift of the CAMB file), the length of the side of the
 * cube, the numbers of each tracer desired (the actual numbers will vary around that
 * value), the bias of both tracers, and the growth rate.
 * 
 * Compile with:
 * g++ -std=c++11 -lgsl -lgslcblas -lfftw -lrfftw -lfftw_threads -lrfftw_threads -lm -fopenmp -march=native -mtune=native -O3 -o LNKNLogs-2Tracer LNKNLogs-2Tracer.cpp lognormal.o
 */


#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>
#include <cmath>
#include <rfftw_threads.h>
#include <omp.h>
#include "pods.h"
#include "lognormal.h"

const int3 N = {512, 512, 512};
const long int N_im = N.x*N.y*(N.z/2 + 1);
const long int N_tot = N.x*N.y*N.z;
const std::string infofile = "RedshiftBinInfo.dat";
const std::string cambbase = "camb_56260000_matterpower_z";
const std::string tracer1base = "LRG";
const std::string tracer2base = "ELG";
const std::string ext = ".dat";

const int numMocks = 1;
const int startNum = 1;
const int numCores = omp_get_max_threads();

// This function dynamically names files in the format "filebase####.fileext". The . should 
// be part of the fileext string.
std::string filename(std::string filebase, double z, int filenum, std::string fileext) {
    std::string file; // Declare a string that will be returned
    
    std::stringstream ss; // Create a string stream to put the pieces together
    ss << filebase << "-z" << z << "-" << std::setw(4) << std::setfill('0') << filenum << fileext; // Put pieces together
    file = ss.str(); // Save the filename to the string
    
    return file; // Return the string
}

int main() {
    std::cout << "Preparing to generate mocks for 2-tracers in multiple redshift bins...\n";
    std::ifstream infoin;
    std::ifstream fin;
    
    fftw_threads_init();
    rfftwnd_plan dp_c2r;
    rfftwnd_plan dp_r2c;
    
    std::cout << "    Creating Fourier transform plans...\n";
    dp_c2r = rfftw3d_create_plan(N.x, N.y, N.z, FFTW_COMPLEX_TO_REAL, FFTW_MEASURE);
    dp_r2c = rfftw3d_create_plan(N.x, N.y, N.z, FFTW_REAL_TO_COMPLEX, FFTW_MEASURE);
    
    int numZBins = 0;
    infoin.open(infofile.c_str(), std::ios::in);
    infoin >> numZBins;
    for (int redBin = 0; redBin < numZBins; ++redBin) {
        std::cout << "    Processing redshift bin " << redBin+1 << " of " << numZBins << "...\n";
        double z;
        double3 L;
        int2 numGals;
        double2 b;
        double f;
        
        infoin >> z >> L.x >> numGals.x >> numGals.y >> b.x >> b.y >> f;
        
        L.y = L.x;
        L.z = L.x;
        
        double V = L.x*L.y*L.z;
        double2 nbar = {double(numGals.x)/V, double(numGals.y)/V};
        
        std::vector< double3 > vk(N.x);
        fftfreq3d(&vk[0], N, L);
        
        std::stringstream ss;
        ss << cambbase << z << ext;
        std::string cambfile = ss.str();
        
        std::vector< double > kin;
        std::vector< double > Pin;
        
        std::cout << "      Reading in CAMB power spectrum...\n";
        fin.open(cambfile.c_str(), std::ios::in);
        while (!fin.eof()) {
            double ktemp;
            double Ptemp;
            
            fin >> ktemp >> Ptemp;
            if (!fin.eof()) {
                kin.push_back(ktemp);
                Pin.push_back(Ptemp);
            }
        }
        fin.close();
        
        std::cout << "      Getting the power at the grid points...\n";
        double *Pk = new double[N_im];
        Pk_CAMB(N, &vk[0], &kin[0], &Pin[0], Pin.size(), Pk);
        
        fftw_complex *dk3di1 = new fftw_complex[N_im];
        fftw_complex *dk3di2 = new fftw_complex[N_im];
        fftw_real *dr3di1 = new fftw_real[N_tot];
        fftw_real *dr3di2 = new fftw_real[N_tot];
#pragma omp parallel for
        for (int i = 0; i < N_tot; ++i) {
            dr3di1[i] = 0.0;
            dr3di2[i] = 0.0;
            if (i < N_im) {
                dk3di1[i].re = 0.0;
                dk3di1[i].im = 0.0;
                dk3di2[i].re = 0.0;
                dk3di2[i].im = 0.0;
            }
        }
        
        std::cout << "      Biasing and adding anisotropy...\n";
        Genddk_2Tracer(N, L, &vk[0], b, f, Pk, dk3di1, dk3di2);
        std::cout << "      Performing initial inverse Fourier transforms...\n";
        rfftwnd_threads_one_complex_to_real(numCores, dp_c2r, dk3di1, dr3di1);
        rfftwnd_threads_one_complex_to_real(numCores, dp_c2r, dk3di2, dr3di2);
        
        std::cout << "      Taking the natural log...\n";
        for (int i = 0; i < N_tot; ++i) {
            dr3di1[i] = log(1.0+dr3di1[i]);
            dr3di2[i] = log(1.0+dr3di2[i]);
            if (i < N_im) {
                dk3di1[i].re = 0.0;
                dk3di1[i].im = 0.0;
                dk3di2[i].re = 0.0;
                dk3di2[i].im = 0.0;
            }
        }
        
        std::cout << "      Performing initial forward Fourier transforms...\n";
        rfftwnd_threads_one_real_to_complex(numCores, dp_r2c, dr3di1, dk3di1);
        rfftwnd_threads_one_real_to_complex(numCores, dp_r2c, dr3di2, dk3di2);
        
        std::vector< double > ratio;
        ratio.reserve(N_im);
        
        std::cout << "      Normalizing and finding the ratio for the tracers...\n";
        for (int i = 0; i < N_im; ++i) {
            dk3di1[i].re /= N_tot;
            dk3di1[i].im /= N_tot;
            
            dk3di2[i].re /= N_tot;
            dk3di2[i].im /= N_tot;
            
            ratio[i] = dk3di2[i].re/dk3di1[i].re;
        }
        
        delete[] dr3di1;
        delete[] dr3di2;
        delete[] dk3di2;
        delete[] Pk;
        
        std::cout << "    Beginning mock generation...\n";
        for (int mock = startNum; mock <= numMocks; ++mock) {
            double startTime = omp_get_wtime();
            string2 outfiles;
            
            outfiles.x = filename(tracer1base, z, mock, ext);
            outfiles.y = filename(tracer2base, z, mock, ext);
            
            std::cout << "      Creating mocks " << outfiles.x << " and " << outfiles.y << "\n";
            
            fftw_complex *dk3d1 = new fftw_complex[N_im];
            fftw_complex *dk3d2 = new fftw_complex[N_im];
            fftw_real *dr3d1 = new fftw_real[N_tot];
            fftw_real *dr3d2 = new fftw_real[N_tot];
            
#pragma omp parallel for
            for (int i = 0; i < N_tot; ++i) {
                dr3d1[i] = 0.0;
                dr3d2[i] = 0.0;
                if (i < N_im) {
                    dk3d1[i].re = 0.0;
                    dk3d1[i].im = 0.0;
                    dk3d2[i].re = 0.0;
                    dk3d2[i].im = 0.0;
                }
            }
            
            std::cout << "      Setting up for the inverse Fourier transforms...\n";
            Sampdk_2Tracer(N, &vk[0], dk3di1, dk3d1, dk3d2, &ratio[0]);
            
            std::cout << "      Performing inverse Fourier transforms...\n";
            rfftwnd_threads_one_complex_to_real(numCores, dp_c2r, dk3d1, dr3d1);
            rfftwnd_threads_one_complex_to_real(numCores, dp_c2r, dk3d2, dr3d2);
            
            double mean1 = 0.0;
            double mean2 = 0.0;
            double2 variance = {0.0, 0.0};
            
            std::cout << "      Finding the means...\n";
            for (int i = 0; i < N_tot; ++i) {
                mean1 += dr3d1[i]/N_tot;
                mean2 += dr3d2[i]/N_tot;
            }
            
            std::cout << "      Mean1 = " << mean1 << "\n";
            std::cout << "      Mean2 = " << mean2 << "\n";
            
            std::cout << "      Calculating variance...\n";
            for (int i = 0; i < N_tot; ++i) {
                dr3d1[i] -= mean1;
                dr3d2[i] -= mean2;
                
                variance.x += (dr3d1[i]*dr3d1[i])/double(N_tot-1.0);
                variance.y += (dr3d2[i]*dr3d2[i])/double(N_tot-1.0);
            }
            
            std::cout << "      Poisson sampling...\n";
            Genddr_2Tracer(N, L, nbar, outfiles, variance, dr3d1, dr3d2);
            
            delete[] dr3d1;
            delete[] dr3d2;
            delete[] dk3d1;
            delete[] dk3d2;
            
            std::cout << "      Time to generate mock: " << omp_get_wtime()-startTime << " s\n";
        }
        delete[] dk3di1;
    }
    infoin.close();
    
    rfftwnd_destroy_plan(dp_c2r);
    rfftwnd_destroy_plan(dp_r2c);
    
    return 0;
}