#include "../include/cosmology.h"
#include <cmath>
#include <gsl/gsl_integration.h>

#define c 299792.458

struct intParams{
    double OmM, OmL;
};

double cosmology::E(double z) {
    return sqrt((1.0 + z)*(1.0 + z)*(1.0 + z)*cosmology::Om_M + cosmology::Om_L);
}

double cosmology::E_inv(double z, void *params) {
    intParams p = *(intParams *)params;
    return 1.0/sqrt((1.0 + z)*(1.0 + z)*(1.0 + z)*p.OmM + p.OmL);
}

cosmology::cosmology(double H_0, double OmegaM, double OmegaL, double Omegab, double Omegac, double Tau,
                     double TCMB) {
    cosmology::Om_M = OmegaM;
    cosmology::Om_L = OmegaL;
    cosmology::Om_b = Omegab;
    cosmology::Om_c = Omegac;
    cosmology::tau = Tau;
    cosmology::T_CMB = TCMB;
    cosmology::h = H_0/100.0;
}

double cosmology::Omega_M() {
    return cosmology::Om_M;
}

double cosmology::Omega_L() {
    return cosmology::Om_L;
}

double cosmology::Omega_bh2() {
    return cosmology::Om_b*cosmology::h*cosmology::h;
}

double cosmology::Omega_ch2() {
    return cosmology::Om_c*cosmology::h*cosmology::h;
}

double cosmology::h_param() {
    return cosmology::h;
}

double cosmology::H0() {
    return cosmology::h*100.0;
}

double cosmology::H(double z) {
    return cosmology::H0()*cosmology::E(z);
}

double cosmology::D_A(double z) {
    double D, error;
    intParams p;
    p.OmM = cosmology::Om_M;
    p.OmL = cosmology::Om_L;
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000000);
    gsl_function F;
    F.function = &cosmology::E_inv;
    F.params = &p;
    gsl_integration_qags(&F, 0.0, z, 1E-6, 1E-6, 1000000, w, &D, &error);
    D *= (c/((1 + z)*cosmology::H0()));
    return D;
}

double cosmology::D_V(double z) {
    double D_ang = cosmology::D_A(z);
    double H_z = cosmology::H(z);
    double D = pow((c*z*(1.0 + z)*(1.0 + z)*D_ang*D_ang)/H_z, 1.0/3.0);
    return D;
}
