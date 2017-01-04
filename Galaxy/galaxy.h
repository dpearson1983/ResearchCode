#ifndef _GALAXY_H_
#define _GALAXY_H_

#include <iostream>
#include <cmath>
#include <constants.h>
#include <gsl/gsl_integration.h>
#include <tpods.h>

struct galFlags{
    enum value{
        FKP_WEIGHT      = 0x01,
        PVP_WEIGHT      = 0x02,
        PSG_WEIGHT      = 0x04,
        UNWEIGHTED      = 0x08,
        INPUT_WEIGHT    = 0x10,
        NGP             = 0x20,
        CIC             = 0x40
    };
};

template <typename T> class galaxy{
    T ra, dec, red, x, y, z, nbar, bias, w;
    
    T wFKP(T P_FKP);
    
    T wPVP(T P_PVP);
    
    T wPSG(T P_PSG);
    
    public:
        galaxy();
        
        galaxy(T RA, T DEC, T RED, T X, T Y, T Z, T NBAR, T BIAS, T W);
        
        void cartesian(double Omega_M, double Omega_L, gsl_integration_workspace *w);
        
        void equatorial(double Omega_M, double Omega_L, gsl_integration_workspace *w);
        
        void bin(double *nden, vec3<double> L, vec3<int> N, vec3<double> r_min, vec3<double> &gal_nbw, double P_w, int flags);
        
        void rMax(vec3<double> &r_max);
        
        void rMin(vec3<double> &r_min);
        
};
        
#endif
