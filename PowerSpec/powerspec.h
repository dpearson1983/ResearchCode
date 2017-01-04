#ifndef _POWERSPEC_H_
#define _POWERSPEC_H_

#include <vector>
#include <string>
#include <fftw3.h>
#include <tpods.h>

struct pkFlags{
    enum value{
        MONO        = 0x001,
        QUAD_FF     = 0x002,
        QUAD_BS     = 0x004,
        HEXA_FF     = 0x008,
        HEXA_BS     = 0x010,
        GRID_COR    = 0x020,
        NGP         = 0x040,
        CIC         = 0x080,
        LIST        = 0x100,
        TABLE       = 0x200,
        HEADER      = 0x400
    };
};

template <typename T> class powerspec{
    std::vector<T> mono, quad, hexa, k;
    int N;
    
    public:
        powerspec();
        
        powerspec(int numKVals);
        
        void calc(fftw_complex *dk3d, vec3<double> L, vec2<double> k_lim, double shotnoise, 
                  int flags);
        
        void disc_cor(std::string file, int flags);
        
        void norm(double gal_nbsqwsq);
        
        void print(int flags);
        
        void print();
        
        void write(std::string file, int flags);
        
};

#endif
