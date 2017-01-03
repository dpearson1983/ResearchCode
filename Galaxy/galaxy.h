#ifndef _GALAXY_H_
#define _GALAXY_H_

#include <gsl/gsl_integration.h>

struct galFlags{
    enum value{
        FKP_WEIGHT      = 0x01,
        PVP_WEIGHT      = 0x02,
        PSG_WEIGHT      = 0x04,
        UNWWEIGHTED     = 0x08,
        INPUT_WEIGHT    = 0x10,
        NGP             = 0x20,
        CIC             = 0x40
    };
};        

template <typename T> struct vec3{
    T x, y, z;
};

template <typename T> class galaxy{
    T ra, dec, red, x, y, z, nbar, bias, w;
    
    double wFKP(double P_FKP);
    
    double wPVP(double P_PVP);
    
    double wPSG(double P_PSG);
    
    public:
        
        void cartesian(double Omega_M, double Omega_L, gsl_integration_workspace *w);
        
        void equatorial(double Omega_M, double Omega_L, gsl_integration_workspace *w);
        
        vec3<double> bin(double *nden, vec3<double> L, vec3<int> N, vec3<double> r_min, double P_w, int flags);
        
        vec3<double> rMax(vec3<double> r_max);
        
        vec3<double> rMin(vec3<double> r_min);
        
};
        
#endif
