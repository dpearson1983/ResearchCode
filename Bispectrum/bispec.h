#ifndef _BISPEC_H_
#define _BISPEC_H_

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
    
    void getks(int numKVals, vec2<double> k_lim);
    
    public:
        
        bispec();
        
        bispec(int numKVals, vec2<double> k_lim, int flags = 0);
        
        void calc(fftw_complex *dk3d);
        
        void mapdrs(vec3<int> N_grid, int flags);
        
        void mapkbins(vec3<int> N_grid, vec2<double> k_lim, int flags);
        
        void setdrs(int index);
        
        void norm();
        
        void print(int flags);
        
        void write(std::sting file, int flags);
        
};

#endif
