#ifndef _POWERSPEC_H_
#define _POWERSPEC_H_

#include <vector>
#include <string>
#include <fftw3.h>
#include <tpods.h>

struct pkFlags{
    enum value{
        MONO         =   0x1,
        QUAD         =   0x2,
        HEXA         =   0x4,
        FF           =   0x8,
        BS           =  0x10,
        GRID_COR     =  0x20,
        NGP          =  0x40,
        CIC          =  0x80,
        LIST         = 0x100,
        TABLE        = 0x200,
        HEADER       = 0x400,
        OUT_OF_PLACE = 0x800
    };
};

template <typename T> class powerspec{
    std::vector<T> mono, quad, hexa, k; // Storage for multipoles and frequencies
    std::vector<int> N_k; // Storage for number of frequencies in each bin
    int N; // Number of frequency bins
    
    void binFreq(fftw_complex A_0, int bin, double grid_cor, double shotnoise);
    
    void freqBin(fftw_complex *A_0, vec3<double> L, vec3<int> N, double shotnoise, 
                 vec2<double> k_lim, int flags);
    
    public:
        powerspec(); // Need to add setter functions for when this is invoked
        
        powerspec(int numKVals, vec2<double> k_lim, int flags = 0);
        
        void calc(double *dr3d, vec3<double> L, vec3<int> N_grid, vec2<double> k_lim, 
                  double shotnoise, std::string fftwWisdom, int flags);
        
        void disc_cor(std::string file, int flags);
        
        void norm(double gal_nbsqwsq, int flags);
        
        void print(int flags);
        
        void print();
        
        void write(std::string file, int flags);
        
};

#endif
