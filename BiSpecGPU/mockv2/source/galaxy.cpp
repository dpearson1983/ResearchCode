#include <sstream>
#include <gsl/gsl_integration.h>
#include "../include/tpods.h"
#include "../include/cosmology.h"
#include "../include/galaxy.h"
#include "../include/constants.h"

struct intparams{
    double O_m, O_L;
};

// This defines the function to be integrated by GSL when locating the distance to a
// galaxy.
double f(double x, void *params) {
    intparams p = *(intparams *)params;
    double ff = c/(100.0*sqrt(p.O_m*(1.0+x)*(1.0+x)*(1.0+x)+p.O_L));
    return ff;
}

// Below is the function which will take a redshift in convert it into a distance.
// This will be needed when calculating the three dimensional positions of the
// galaxy and random catalogs.
double rz(double red, double O_m, double O_L, gsl_integration_workspace *w_gsl) {
    double error;
    double D;
    intparams p;
    p.O_m = O_m;
    p.O_L = O_L;
    gsl_function F;
    F.function = &f;
    F.params = &p;
    gsl_integration_qags(&F, 0.0, red, 1e-6, 1e-6, 10000000, w_gsl, &D, &error);
    
    return D;
}

galaxy::galaxy() {
    galaxy::ra = 0.0;
    galaxy::dec = 0.0;
    galaxy::red = 0.0;
}

galaxy::galaxy(double RA, double DEC, double RED, double M, double N, double B, double W, double WRF, 
               double WCP, double PFKP) {
    galaxy::initialize(RA, DEC, RED, M, N, B, W, WRF, WCP, PFKP);
}

void galaxy::initialize(double RA, double DEC, double RED, double M, double N, double B, double W, 
                        double WRF, double WCP, double PFKP) {
    galaxy::ra = RA;
    galaxy::dec = DEC;
    galaxy::red = RED;
    galaxy::mass = M;
    galaxy::n = N;
    galaxy::b = B;
    galaxy::w = W;
    galaxy::w_rf = WRF;
    galaxy::w_cp = WCP;
    galaxy::P_FKP = PFKP;
}

vec3<double> galaxy::cartesian(cosmology cosmo, gsl_integration_workspace *w_gsl) {
    double r = rz(galaxy::red, cosmo.Omega_M(), cosmo.Omega_L(), w_gsl);
    vec3<double> cart = {double(r*cos(galaxy::dec*pi/180.0)*cos(galaxy::ra*pi/180.0)),
                         double(r*cos(galaxy::dec*pi/180.0)*sin(galaxy::ra*pi/180.0)),
                         double(r*sin(galaxy::dec*pi/180.0))};
    return cart;
}

vec3<double> galaxy::equatorial() {
    vec3<double> equa = {galaxy::ra, galaxy::dec, galaxy::red};
    return equa;
}

double galaxy::W(int flags) {
    if (flags & galFlags::FKP_WEIGHT) {
        return (1.0/(1.0 + galaxy::n*galaxy::P_FKP))*(galaxy::w_rf + galaxy::w_cp - 1.0);
    } else if (flags & galFlags::PVP_WEIGHT) {
        std::stringstream message;
        message << "PVP weighting not yet implemented." << std::endl;
        throw std::runtime_error(message.str());
    } else if (flags & galFlags::UNWEIGHTED) {
        return 1.0;
    } else if (flags & galFlags::INPUT_WEIGHT) {
        return galaxy::w*(galaxy::w_rf + galaxy::w_cp - 1.0);
    } else {
        return galaxy::w;
    }
}

double galaxy::N() {
    return galaxy::n;
}
