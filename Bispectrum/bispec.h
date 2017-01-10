#ifndef _BISPEC_H_
#define _BISPEC_H_

#include <vector>
#include <string>
#include <fftw3.h>
#include <tpods.h>

struct bkFlags{
    enum value{
        IN_PLACE     = 0x1,
        OUT_OF_PLACE = 0x2,
        HEADER       = 0x4,
        TABLE        = 0x8
    };
};

template <typename T> class bispec{
    std::vector<T> val;
    std::vector<vec3<T>> ks;
    std::vector<int> drs, kbins;
    int N;
    
    void get_shell(double *dk3d, double *dk3d_shell, vec3<int> N_grid, int kBin);
    
    void get_shell(fftw_complex *dk3d, double *dk3d_shell, vec3<int> N_grid, int kBin);
    
    void getks(int numKVals, vec2<double> k_lim);
    
    void mapdrs(vec3<int> N_grid, int flags);
        
    void mapkbins(vec3<int> N_grid, vec3<double> dk, vec2<double> k_lim, int flags);
    
    public:
        
        bispec();
        
        bispec(int numKVals, vec3<double> L, vec3<int> N_grid, vec2<double> k_lim, 
               int flags = 0);
        
        void calc(double *dk3d, vec3<int> N_grid, vec2<double> k_lim, double V_f,
                  std::string fftwWisdom, powerspec<T> Pk, double nbar, double norm);
        
        void calc(fftw_complex *dk3d, vec3<int> N_grid, vec2<double> k_lim, double V_f,
                  std::string fftwWisdom, powerspec<T> Pk, double nbar, double norm);
        
        void norm();
        
        void print();
        
        void write(std::string file);
        
};

#endif
