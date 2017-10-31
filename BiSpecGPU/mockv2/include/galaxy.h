#ifndef _GALAXY_H_
#define _GALAXY_H_

#include <gsl/gsl_integration.h>
#include "cosmology.h"
#include "tpods.h"

struct galFlags{
    enum value{
        FKP_WEIGHT      = 0x01,
        PVP_WEIGHT      = 0x02,
        PSG_WEIGHT      = 0x04,
        UNWEIGHTED      = 0x08,
        INPUT_WEIGHT    = 0x10
    };
};

class galaxy{
    double ra, dec, red, mass, n, b, w, w_rf, w_cp, P_FKP;
    
    public:
        galaxy();
        
        galaxy(double RA, double DEC, double RED, double M = 0.0, double N = 0.0, double B = 1.0, 
               double W = 1.0, double WRF = 1.0, double WCP = 1.0, double PFKP = 20000.0);
        
        void initialize(double RA, double DEC, double RED, double M = 0.0, double N = 0.0, double B = 1.0, 
                        double W = 1.0, double WRF = 1.0, double WCP = 1.0, , double PFKP = 20000.0);
        
        vec3<double> cartesian(cosmology cos, gsl_integration_workspace *w_gsl);
        
        vec3<double> equatorial();
        
        double W(int flags = 0);
        
        double N();
        
};

#endif
