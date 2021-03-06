#ifndef _COSMOLOGY_H_
#define _COSMOLOGY_H_

class cosmology{
    double Om_M, Om_L, Om_b, Om_c, tau, T_CMB, h;
    
    public:
        cosmology(double H_0 = 70.0, double OmegaM = 0.3, double OmegaL = 0.7, double Omegab = 0.04, 
                  double Omegac = 0.26, double Tau = 0.066, double TCMB = 2.718);
        
        double Omega_M();
        
        double Omega_L();
        
        double Omega_bh2();
        
        double Omega_ch2();
        
        double h_param();
        
        double H0();

};

#endif
