#include <iostream>
#include <cmath>
#include <fstream>
#include <cstring>
#include <sstream>
#include <cstdlib>
#include <random>
#include <fftw3.h>
#include <omp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>
// #include <boost/iostreams/device/mapped_file.hpp>
#include "pods.h"
#include "lognormal.h"

double gridCorCIC(double kx, double ky, double kz, double3 binSize) {    
    double ax = (kx*binSize.x)/2.0 + 1E-17;
    double ay = (ky*binSize.y)/2.0 + 1E-17;
    double az = (kz*binSize.z)/2.0 + 1E-17;
    
    double sincx = sin(ax)/ax;
    double sincy = sin(ay)/ay;
    double sincz = sin(az)/az;
    double prodsinc = sincx*sincy*sincz;
    
    double grid_cor = 1.0/(prodsinc*prodsinc);
    
    return grid_cor;
}
       
double cloudInCell(double *data, double3 r, int3 N, double3 L, double3 dL) {
    int3 ngp = {int(r.x/dL.x), int(r.y/dL.y), int(r.z/dL.z)};
    double3 r_ngp = {(ngp.x + 0.5)*dL.x, (ngp.y + 0.5)*dL.y, (ngp.z + 0.5)*dL.z};
    double3 dr = {r.x-r_ngp.x, r.y-r_ngp.y, r.z-r_ngp.z};
    int3 shift = {int(dr.x/fabs(dr.x)), int(dr.y/fabs(dr.y)), int(dr.z/fabs(dr.z))};
    
    if (ngp.x+shift.x < 0) shift.x = N.x-1;
    if (ngp.y+shift.y < 0) shift.y = N.y-1;
    if (ngp.z+shift.z < 0) shift.z = N.z-1;
    if (ngp.x+shift.x == N.x) shift.x = 1-N.x;
    if (ngp.y+shift.y == N.y) shift.y = 1-N.y;
    if (ngp.z+shift.z == N.z) shift.z = 1-N.z;
    
    dr.x = fabs(dr.x);
    dr.y = fabs(dr.y);
    dr.z = fabs(dr.z);
    
    int index = ngp.z + N.z*(ngp.y + N.y*ngp.x);
    double V = dL.x*dL.y*dL.z;
    double V1 = dr.x*dr.y*dr.z;
    double V2 = (dL.x-dr.x)*dr.y*dr.z;
    double V3 = dr.x*(dL.y-dr.y)*dr.z;
    double V4 = dr.x*dr.y*(dL.z-dr.z);
    double V5 = (dL.x-dr.x)*(dL.y-dr.y)*dr.z;
    double V6 = (dL.x-dr.x)*dr.y*(dL.z-dr.z);
    double V7 = dr.x*(dL.y-dr.y)*(dL.z-dr.z);
    double V8 = (dL.x-dr.x)*(dL.y-dr.y)*(dL.z-dr.z);
    
    double result = data[index]*V8 + data[index+shift.x*N.z*N.y]*V7 
                  + data[index+shift.y*N.z]*V6 + data[index+shift.z]*V5 
                  + data[index+shift.y*N.z+shift.x*N.z*N.y]*V4
                  + data[index+shift.z+shift.x*N.z*N.y]*V3 
                  + data[index+shift.z+shift.y*N.z]*V2
                  + data[index+shift.z+shift.y*N.z+shift.x*N.z*N.y]*V1;
    result /= V;
    
    return result;
}

double3 cloudInCell3(double *vx, double *vy, double *vz, double3 r, int3 N, double3 L,
                     double3 dL) {
    int3 ngp = {int(r.x/dL.x), int(r.y/dL.y), int(r.z/dL.z)};
    double3 r_ngp = {ngp.x*dL.x, ngp.y*dL.y, ngp.z*dL.z};
    double3 dr = {r.x-r_ngp.x, r.y-r_ngp.y, r.z-r_ngp.z};
    int3 shift = {int(dr.x/fabs(dr.x)), int(dr.y/fabs(dr.y)), int(dr.z/fabs(dr.z))};
    
    if (ngp.x+shift.x < 0) shift.x = N.x-1;
    if (ngp.y+shift.y < 0) shift.y = N.y-1;
    if (ngp.z+shift.z < 0) shift.z = N.z-1;
    if (ngp.x+shift.x == N.x) shift.x = 1-N.x;
    if (ngp.y+shift.y == N.y) shift.y = 1-N.y;
    if (ngp.z+shift.z == N.z) shift.z = 1-N.z;
    
    dr.x = fabs(dr.x);
    dr.y = fabs(dr.y);
    dr.z = fabs(dr.z);
    
    int index = ngp.z + N.z*(ngp.y + N.y*ngp.x);
    double V = dL.x*dL.y*dL.z;
    double V1 = dr.x*dr.y*dr.z;
    double V2 = (dL.x-dr.x)*dr.y*dr.z;
    double V3 = dr.x*(dL.y-dr.y)*dr.z;
    double V4 = dr.x*dr.y*(dL.z-dr.z);
    double V5 = (dL.x-dr.x)*(dL.y-dr.y)*dr.z;
    double V6 = (dL.x-dr.x)*dr.y*(dL.z-dr.z);
    double V7 = dr.x*(dL.y-dr.y)*(dL.z-dr.z);
    double V8 = (dL.x-dr.x)*(dL.y-dr.y)*(dL.z-dr.z);
    
    double3 result;
    result.x = vx[index]*V8 
             + vx[index+shift.x*N.z*N.y]*V7 
             + vx[index+shift.y*N.z]*V6 
             + vx[index+shift.z]*V5 
             + vx[index+shift.y*N.z+shift.x*N.z*N.y]*V4
             + vx[index+shift.z+shift.x*N.z*N.y]*V3 
             + vx[index+shift.z+shift.y*N.z]*V2
             + vx[index+shift.z+shift.y*N.z+shift.x*N.z*N.y]*V1;
    result.y = vy[index]*V8 
             + vy[index+shift.x*N.z*N.y]*V7 
             + vy[index+shift.y*N.z]*V6 
             + vy[index+shift.z]*V5 
             + vy[index+shift.y*N.z+shift.x*N.z*N.y]*V4
             + vy[index+shift.z+shift.x*N.z*N.y]*V3 
             + vy[index+shift.z+shift.y*N.z]*V2
             + vy[index+shift.z+shift.y*N.z+shift.x*N.z*N.y]*V1;
    result.z = vz[index]*V8 
             + vz[index+shift.x*N.z*N.y]*V7 
             + vz[index+shift.y*N.z]*V6 
             + vz[index+shift.z]*V5 
             + vz[index+shift.y*N.z+shift.x*N.z*N.y]*V4
             + vz[index+shift.z+shift.x*N.z*N.y]*V3 
             + vz[index+shift.z+shift.y*N.z]*V2
             + vz[index+shift.z+shift.y*N.z+shift.x*N.z*N.y]*V1;
    result.x /= V;
    result.y /= V;
    result.z /= V;
    
    return result;
}

void Gendk(int3 N, double3 L, double b, double *kval, double *Pk, int numKVals, 
           fftw_complex *dk3d) {
    gsl_spline *Power = gsl_spline_alloc(gsl_interp_cspline, numKVals);
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    
    gsl_spline_init(Power, kval, Pk, numKVals);
    
    double3 dk;
    dk.x = (2.0*pi)/L.x;
    dk.y = (2.0*pi)/L.y;
    dk.z = (2.0*pi)/L.z;
    
    for (int i = 0; i < N.x; ++i) {
        double kx = double(i - ((i - 1)/(N.x/2))*N.x)*dk.x;
        
        for (int j = 0; j < N.y; ++j) {
            double ky = double(j - ((j - 1)/(N.y/2))*N.y)*dk.y;
            
            for (int k = 0; k <= N.z/2; ++k) {
                double kz = k*dk.z;
                
                int index = k + (N.z/2 + 1)*(j + N.y*i);
                
                double k_tot = sqrt(kx*kx+ky*ky+kz*kz);
                double mu = kx/k_tot;
                
                if (k_tot != 0) {
                    double P = b*b*gsl_spline_eval(Power, k_tot, acc);
                    
                    dk3d[index][0] = P;
                    dk3d[index][1] = 0.0;
                } else {
                    dk3d[index][0] = 0.0;
                    dk3d[index][1] = 0.0;
                }
                
            }
        }
    }
    
    gsl_spline_free(Power);
    gsl_interp_accel_free(acc);
}

void Smpdk(int3 N, double3 L, double b, double h, double f, std::string dk3difile, 
           fftw_complex *dk3d, fftw_complex *vk3dx, fftw_complex *vk3dy, fftw_complex *vk3dz) {
    std::mt19937_64 generator;
    std::random_device seeder;
    generator.seed(seeder());
    std::normal_distribution<double> distribution(0.0, 1.0);
    
    std::ifstream fin;
    
    double3 dk;
    dk.x = 2.0*pi/L.x;
    dk.y = 2.0*pi/L.y;
    dk.z = 2.0*pi/L.z;
    
    double3 dr = {L.x/double(N.x), L.y/double(N.y), L.z/double(N.z)};
    
    double coeff = 1.0;
    
    fin.open(dk3difile.c_str(), std::ios::in|std::ios::binary);
    for (int i = 0; i < N.x; ++i) {
        double kx = (i - ((i - 1)/(N.x/2))*N.x)*dk.x;
        int i2 = (2*N.x - i) % N.x;
        
        for (int j = 0; j < N.y; ++j) {
            double ky = (j - ((j - 1)/(N.y/2))*N.y)*dk.y;
            int j2 = (2*N.y - j) % N.y;
            
            for (int k = 0; k <= N.z/2; ++k) {
                double kz = k*dk.z;
                
                int index1 = k + (N.z/2 + 1)*(j + N.y*i);
                int index2 = k + (N.z/2 + 1)*(j2 + N.y*i2);
                
                double k_tot = kx*kx + ky*ky + kz*kz;
                //double grid_cor = gridCorCIC(kx, ky, kz, dr);
                
                fftw_complex dk3di;
                fin.read((char *) &dk3di, sizeof(fftw_complex));
                double Power = dk3di[0];
                
                double k_invsq;
                if (k_tot > 0) k_invsq = 1.0/(k_tot);
                else k_invsq = 0.0;
                
                if ((i == 0 || i == N.x/2) && (j == 0 || j == N.y/2) && (k == 0 || k == N.z/2)){
                    dk3d[index1][0] = distribution(generator)*sqrt(Power*dk3di[1]);
                    dk3d[index1][1] = 0.0;
                    
                    vk3dx[index1][1] = 0.0;
                    vk3dy[index1][1] = 0.0;
                    vk3dz[index1][1] = 0.0;
                    
                    vk3dx[index1][0] = 0.0;
                    vk3dy[index1][0] = 0.0;
                    vk3dz[index1][0] = 0.0;
                } else if (k == 0 || k == N.z/2) {
                    dk3d[index1][0] = distribution(generator)*sqrt(Power/2.0);
                    dk3d[index1][1] = distribution(generator)*sqrt(Power/2.0);
                    
                    vk3dx[index1][1] = k_invsq*kx*dk3d[index1][0];
                    vk3dx[index1][0] = -k_invsq*kx*dk3d[index1][1];
                    
                    vk3dy[index1][1] = k_invsq*ky*dk3d[index1][0];
                    vk3dy[index1][0] = -k_invsq*ky*dk3d[index1][1];
                    
                    vk3dz[index1][1] = k_invsq*kz*dk3d[index1][0];
                    vk3dz[index1][0] = -k_invsq*kz*dk3d[index1][1];
                    
                    vk3dx[index2][1] = -vk3dx[index1][1];
                    vk3dx[index2][0] = vk3dx[index1][0];
                    
                    vk3dy[index2][1] = -vk3dy[index1][1];
                    vk3dy[index2][0] = vk3dy[index1][0];
                    
                    vk3dz[index2][1] = -vk3dz[index1][1];
                    vk3dz[index2][0] = vk3dz[index1][0];
                    
                    dk3d[index1][0] *= sqrt(dk3di[1]);
                    dk3d[index1][1] *= sqrt(dk3di[1]);
                    
                    dk3d[index2][0] = dk3d[index1][0];
                    dk3d[index2][1] = -dk3d[index1][1];
                } else {
                    dk3d[index1][0] = distribution(generator)*sqrt(Power/2.0);
                    dk3d[index1][1] = distribution(generator)*sqrt(Power/2.0);
                    
                    vk3dx[index1][1] = k_invsq*kx*dk3d[index1][0];
                    vk3dx[index1][0] = -k_invsq*kx*dk3d[index1][1];
                    
                    vk3dy[index1][1] = k_invsq*ky*dk3d[index1][0];
                    vk3dy[index1][0] = -k_invsq*ky*dk3d[index1][1];
                    
                    vk3dz[index1][1] = k_invsq*kz*dk3d[index1][0];
                    vk3dz[index1][0] = -k_invsq*kz*dk3d[index1][1];
                    
                    dk3d[index1][0] *= sqrt(dk3di[1]);
                    dk3d[index1][1] *= sqrt(dk3di[1]);
                }
            }
        }
    }
    fin.close();
}

void Gendr(int3 N, double3 L, double *nbar, int numTracers, std::string file, double variance,
           double *dr3d, double *vr3dx, double *vr3dy, double *vr3dz, 
           std::vector< double > b, double bias) {
    std::mt19937_64 generator;
    std::random_device seeder;
    generator.seed(seeder());
    std::uniform_real_distribution<double> pos(0.0, 1.0);
    
    std::ofstream fout; // Output filestream
    
    double maxden = 0.0;
    double3 dL = {L.x/double(N.x), L.y/double(N.y), L.z/double(N.z)};
    double *n = new double[numTracers];
    for (int i = 0; i < numTracers; ++i) {
        n[i] = nbar[i]*dL.x*dL.y*dL.z;
    }
    
    std::vector<galaxy> gals;
    fout.open(file.c_str(),std::ios::out|std::ios::binary); // Open output file
    //fout.precision(15); // Set the number of digits to output
    
    // Loop through all the grid points, assign galaxies uniform random positions within
    // each cell.
    for (int i = 0; i < N.x; ++i) {
        double xmin = i*dL.x;
        for (int j = 0; j < N.y; ++j) {
            double ymin = j*dL.y;
            for (int k = 0; k < N.z; ++k) {
                double zmin = k*dL.z;
                
                int index = k + N.z*(j+N.y*i); // Calculate grid index
                
                // Initialize the Poisson distribution with the value of the matter field
                for (int tracer = 0; tracer < numTracers; ++tracer) {
                    //double ratio = b[tracer];
                    double density = n[tracer]*exp(dr3d[index]-variance/2.0);
                    //density = pow(density,ratio);
                    if (density > maxden) maxden = density;
                    std::poisson_distribution<int> distribution(density);
                    int numGal = distribution(generator); // Randomly Poisson sample
                    
                    // Randomly generate positions for numGal galaxies within the cell
                    for (int gal = 0; gal < numGal; ++gal) {
                        double3 r = {xmin + pos(generator)*dL.x,
                                     ymin + pos(generator)*dL.y,
                                     zmin + pos(generator)*dL.z};
                        double3 v = cloudInCell3(vr3dx, vr3dy, vr3dz, r, N, L, dL);
                        galaxy temp = {r.x, r.y, r.z, v.x, v.y, v.z, b[tracer]};
                        
                        gals.push_back(temp);
                        
//                         fout << r.x << " " << r.y << " " << r.z << " " 
//                              << v.x << " " << v.y << " " << v.z << " " 
//                              << b[tracer] << "\n";
                    }
                }
            }
        }
    }
    fout.write((char *) &gals[0], gals.size()*sizeof(galaxy));
    fout.close(); // Close file
    delete[] n;
    std::cout << "    Maximum Density = " << maxden << "\n";
}

void Gendr_interp(int3 N, double3 L, double *nbar, int numTracers, std::string file,
                  double variance, double *dr3d, double *vr3dx, double *vr3dy, 
                  double *vr3dz, std::vector< double > b, double bias) {
    std::mt19937_64 gen;
    std::random_device seeder;
    gen.seed(seeder());
    
    std::ofstream fout;
    
    std::uniform_real_distribution<double> pos(0.0, 1.0);
    
    double3 dL = {L.x/double(N.x), L.y/double(N.y), L.z/double(N.z)};
    
    fout.open(file.c_str(),std::ios::out);
    fout.precision(15);
    int *numGals = new int[numTracers];
    for (int i = 0; i < numTracers; ++i) {
        numGals[i] = nbar[i]*L.x*L.y*L.z;
        int gals = 0;
        double ratio = b[i]/bias;
        while (gals < numGals[i]) {
            double3 trial = {pos(gen)*L.x, pos(gen)*L.y, pos(gen)*L.z};
            
            double dr3d_cic = cloudInCell(dr3d, trial, N, L, dL);
            
            double density = nbar[i]*dL.x*dL.y*dL.z*exp(dr3d_cic - variance/2.0);
            density = pow(density,ratio);
            double prob = exp(-density);
            double test = pos(gen);
            
            if (prob < test) {
                double vr3dx_cic = cloudInCell(vr3dx, trial, N, L, dL);
                double vr3dy_cic = cloudInCell(vr3dy, trial, N, L, dL);
                double vr3dz_cic = cloudInCell(vr3dz, trial, N, L, dL);
                
                fout << trial.x << " " << trial.y << " " << trial.z << " " << vr3dx_cic
                     << " " << vr3dy_cic << " " << vr3dz_cic << " " << b[i] << "\n";
                
                ++gals;
            }
        }
    }
    fout.close();
    delete[] numGals;
}
