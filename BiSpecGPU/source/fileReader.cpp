#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <CCfits/CCfits>
#include <gsl/gsl_integration.h>
#include <tpods.h>
#include <galaxy.h>
#include "../include/fileReader.h"

void initArray(double *array, int N) {
    for (int i = 0; i < N; ++i)
        array[i] = 0.0;
}

double readFits(std::string file, std::vector<std::string> hdus, int hduTableNum, 
              double *&nden, double res, vec3<double> &L, vec3<int> &N, vec3<double> &r_min, 
              vec3<double> &pk_nbw, vec3<double> &bk_nbw, double P_w, int flags, double Omega_M, 
              double Omega_L, double z_min, double z_max) {
    std::cout << "Reading from file " << file << std::endl;
    
    double nbar = 0.0;
    
    std::auto_ptr<CCfits::FITS> pInfile(new CCfits::FITS(file, CCfits::Read,hdus,false));
    
    CCfits::ExtHDU &table = pInfile->extension(hdus[hduTableNum]);
    long start = 1L;
    long end = table.rows();
    std::vector<double> ra;
    std::vector<double> dec;
    std::vector<double> z;
    std::vector<double> nz;
    std::vector<double> weight;
    
    table.column("RA").read(ra, start, end);
    table.column("DEC").read(dec, start, end);
    table.column("Z").read(z, start, end);
    table.column("NZ").read(nz, start, end);
    table.column("WEIGHT_FKP").read(weight, start, end);
    
    int numGals = ra.size();
    std::cout << "numRans = " << numGals << std::endl;
    std::vector< galaxy<double> > gals;
    vec3<double> r_max = {-10000.0, -10000.0, -100000.0};
    r_min.x = 1000000.0;
    r_min.y = 1000000.0;
    r_min.z = 1000000.0;
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(10000000);
//     std::ofstream fout;
//     fout.open("nbarvsz.dat", std::ios::out);
//     fout.precision(15);
    for (int i = 0; i < numGals; ++i) {
        if (z[i] <= z_max && z[i] >= z_min) {
            galaxy<double> gal(ra[i], dec[i], z[i], 0.0, 0.0, 0.0, nz[i], 0.0, weight[i]);
            gal.cartesian(Omega_M, Omega_L, w);
            gal.rMax(r_max);
            gal.rMin(r_min);
            gals.push_back(gal);
            nbar += weight[i]*nz[i];
//             fout << z[i] << " " << nz[i] << "\n";
        }
    }
//     fout.close();
    gsl_integration_workspace_free(w);
    
    L.x = r_max.x - r_min.x;
    L.y = r_max.y - r_min.y;
    L.z = r_max.z - r_min.z;
    
    std::cout.precision(15);
    std::cout << "x_min = " << r_min.x << std::endl;
    std::cout << "y_min = " << r_min.y << std::endl;
    std::cout << "z_min = " << r_min.z << std::endl;
    std::cout << "x_max = " << r_max.x << std::endl;
    std::cout << "y_max = " << r_max.y << std::endl;
    std::cout << "z_max = " << r_max.z << std::endl;
    
    std::cout << "Minimum box dimensions: " << L.x << ", " << L.y << ", " << L.z << std::endl;
    
    N.x = int(pow(2,int(log2(L.x/res)) + 1));
    N.y = int(pow(2,int(log2(L.y/res)) + 1));
    N.z = int(pow(2,int(log2(L.z/res)) + 1));
    
    int N_tot = N.x*N.y*N.z;
    nden = new double[N_tot];
    initArray(nden, N_tot);
    
    r_min.x -= (N.x*res - L.x)/2.0;
    r_min.y -= (N.y*res - L.y)/2.0;
    r_min.z -= (N.z*res - L.z)/2.0;
    
    std::cout.precision(15);
    std::cout << "x_min = " << r_min.x << std::endl;
    std::cout << "y_min = " << r_min.y << std::endl;
    std::cout << "z_min = " << r_min.z << std::endl;
    
    L.x = N.x*res;
    L.y = N.y*res;
    L.z = N.z*res;
    
    for (int i = 0; i < numGals; ++i)
        gals[i].bin(nden, L, N, r_min, pk_nbw, bk_nbw, P_w, flags);
    
    nbar /= pk_nbw.x;
    
    return nbar;
}

double readFits(std::string file, std::vector<std::string> hdus, int hduTableNum, 
              double *&nden, vec3<double> L, vec3<int> N, vec3<double> r_min, 
              vec3<double> &pk_nbw, vec3<double> &bk_nbw, double P_w, int flags, double Omega_M, 
              double Omega_L, double z_min, double z_max) {
    std::cout << "Reading from file " << file << std::endl;
    
    double nbar = 0.0;
    
    std::auto_ptr<CCfits::FITS> pInfile(new CCfits::FITS(file, CCfits::Read,hdus,false));
    
    CCfits::ExtHDU &table = pInfile->extension(hdus[hduTableNum]);
    long start = 1L;
    long end = table.rows();
    std::vector<double> ra;
    std::vector<double> dec;
    std::vector<float> z;
    std::vector<float> nz;
    std::vector<float> weight;
    
    table.column("RA").read(ra, start, end);
    table.column("DEC").read(dec, start, end);
    table.column("Z").read(z, start, end);
    table.column("NZ").read(nz, start, end);
    table.column("WEIGHT_FKP").read(weight, start, end);
    
    int N_tot = N.x*N.y*N.z;
    nden = new double[N_tot];
    initArray(nden, N_tot);
    
    int numGals = ra.size();
    std::cout << "numGals = " << numGals << std::endl;
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(10000000);
    for (int i = 0; i < numGals; ++i) {
        if (z[i] <= z_max && z[i] >= z_min) {
            galaxy<double> gal(ra[i], dec[i], z[i], 0.0, 0.0, 0.0, nz[i], 0.0, weight[i]);
            gal.cartesian(Omega_M, Omega_L, w);
            gal.bin(nden, L, N, r_min, pk_nbw, bk_nbw, P_w, flags);
            nbar += weight[i]*nz[i];
        }
    }
    gsl_integration_workspace_free(w);
    
    nbar /= pk_nbw.x;
    
    return nbar;
}
