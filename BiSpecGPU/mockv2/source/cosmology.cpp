#include "../include/cosmology.h"

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
