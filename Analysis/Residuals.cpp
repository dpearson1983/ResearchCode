#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>
#include <cmath>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>

struct Pk {
    double k, P;
};

int main() {
    double b = 2.0;
    double f = 0.537244398679;
    
    std::ifstream fin;
    std::ofstream fout;
    
    std::vector< Pk > InputPower;
    int numKModes = 0;
    
    std::cout << "Reading input power file...\n";
    fin.open("camb_90763888_matterpower_z0.57.dat",std::ios::in);
    while (!fin.eof()) {
        Pk Input_temp;
        fin >> Input_temp.k >> Input_temp.P;
        
        if (!fin.eof()) {
            InputPower.push_back(Input_temp);
            ++numKModes;
        }
    }
    fin.close();
    
    double *kvals = new double[numKModes];
    double *InPow = new double[numKModes];
    
    for (int i = 0; i < numKModes; ++i) {
        kvals[i] = InputPower[i].k;
        InPow[i] = InputPower[i].P;
    }
    
    gsl_spline *Power = gsl_spline_alloc(gsl_interp_cspline, numKModes);
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    
    gsl_spline_init(Power, kvals, InPow, numKModes);
    
    double mono_scale = b*b + 0.66666666666667*b*f + 0.2*f*f;
    double quad_scale = 1.333333333333*b*f + 0.571428571429*f*f;
    
    fin.open("Avg_Pk_0020.dat",std::ios::in);
    fout.open("Avg_Pk_0020_Residuals.dat",std::ios::out);
    
    for (int i = 0; i < 64; ++i) {
        double k = 0.0;
        double P0 = 0.0;
        double P0_err = 0.0;
        double P2 = 0.0;
        double P2_err = 0.0;
        double ratio = 0.0;
        double ratio_err = 0.0;
        fin >> k >> P0 >> P0_err >> P2 >> P2_err >> ratio >> ratio_err;
        
        double P = gsl_spline_eval(Power, k, acc);
        
        double P0_exp = mono_scale*P;
        double P2_exp = quad_scale*P;
        
        fout << k << " " << (P0-P0_exp)/P0_exp << " " << P0_err/P0_exp << " " << (P2-P2_exp)/P2_exp << " " << P2_err/P2_exp << "\n";
    }
    
    fin.close();
    fout.close();
    
    delete[] kvals;
    delete[] InPow;
    
    gsl_spline_free(Power);
    gsl_interp_accel_free(acc);
    
    return 0;
}
    
    