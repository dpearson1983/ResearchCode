// NewSimPKCode v1
// David W. Pearson, Ph.D.
// December 4, 2015
//
// This code is intended to calculate the power spectrum of simulated galaxy catalogs for which
// the exact positions and velocites of galaxies are known. In an effort to keep the code as 
// general as possible, in anticipation of having to run on many different simulations and 
// calculate both the monopole and quadrupole of the power spectrum, this code will have many
// things factored out as functions which are called in the main body of code. This way, by
// changing a few parameters at the top of the source file, the code can be easily ran on 
// different simulations.
//
// Compile with:
// g++ -lfftw -lrfftw -lfftw_threads -lrfftw_threads -lm -fopenmp -march=native -mtune=native -O2 NewSimPKCode.cpp -o NewSimPKCode

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>
#include <cmath>
#include <omp.h>
#include <algorithm>
#include <rfftw_threads.h>

using namespace std;

// Below are some useful constants and various parameters the code needs defined in order to 
// run. These are intended to be moved to a parameter file eliminating the need to recompile the
// code each time you need to change one of them, such as the file name base, FFT size, etc.
const double c = 299792.458; // Speed of light in units km s^-1
const double H_0 = 100.0; // Hubble constant in units of h km s^-1 Mpc^-1
const double Omega_L = 0.726; // Dark energy density parameter
const double Omega_M = 0.274; // Matter density parameter
const double pi = 3.14159265359; // Mmmmmmm, Pi!
const double h = 0.7;

// Below are parameters that control the resolution of the FFT.
const unsigned int N_x = 1024; // Size of the FFT in x
const unsigned int N_y = 1024; // Size of the FFT in y
const unsigned int N_z = 1024; // Size of the FFT in z
const unsigned int N_tot = N_x*N_y*N_z; // Size of FFT arrays

const double L_x = 2048.0;
const double L_y = 2048.0;
const double L_z = 2048.0;

const int NB = 64;
const double k_min = 0.0;
const double k_max = 0.512;
const double binWidth = (k_max-k_min)/((double)NB);

const int numDyn = 1;
const int numKOut = 64;
const int numMocks = 20;

const int startNum = 1;

const int numCores = omp_get_max_threads();

const string galbase = "galaxies";
const string ranbase = "randoms";
const string galinfobase = "GalaxyNum";
const string pkbase = "Power_Spectrum";
const string dynpkbase = "FKP_Pk_Vals";
const string extdat = ".dat";
const string extbin = ".bin";

struct int4{
    int x, y, z, w;
};

struct int8{
    int cic1, cic2, cic3, cic4, cic5, cic6, cic7, cic8;
};

struct double3{
    double x, y, z;
};

void fftfreq(double *kvec, int N, double L) {
    double dk = 1.0/L;
    
    for (int i = 0; i <= N/2; ++i) {
        kvec[i] = i*dk;
    }
    for (int i = N/2+1; i < N; ++i) {
        kvec[i] = (i-N)*dk;
    }
}

string filename(string filebase, string fileext) {
    string file;
    
    stringstream ss;
    ss << filebase << fileext;
    file = ss.str();
    
    return file;
}

string filenumber(string filebase, int filenum, string fileext) {
    string file;
    
    stringstream ss;
    ss << filebase << setw(4) << setfill('0') << filenum << fileext;
    file = ss.str();
    
    return file;
}

double FKPWeight(int w, double nbar, double P_0) {
    double w_fkp = w/(1.0+nbar*P_0);
    
    return w_fkp;
}

double PVPWeight(int w, double nbar, double P_0, double bias, double nbias) {
    double w_pvp = (w*bias*P_0)/(1.0+nbias*P_0);
    
    return w_pvp;
}

void PowerFF (double kx, double ky, double kz, double k, int i, int j, int l, int bin,      
              fftw_complex *A_0, double *P0, double *N0, double *P2, double grid_cor) {
    double A_0r = A_0[l + (N_z/2 + 1)*(j+N_y*i)].re;
    double A_0i = A_0[l + (N_z/2 + 1)*(j+N_y*i)].im;
    
    if (k != 0) {
        double mu = ky/k;
        P0[bin] += (A_0r*A_0r+A_0i*A_0i)*grid_cor*grid_cor;
        P2[bin] += 2.5*(A_0r*A_0r+A_0i*A_0i)*(3.0*mu*mu - 1.0)*grid_cor*grid_cor;
        ++N0[bin];
    } else {
        P0[bin] += (A_0r*A_0r+A_0i*A_0i)*grid_cor*grid_cor;
        P2[bin] += 0.0;
        ++N0[bin];
    }
}
    

void PowerSpec(double kx, double ky, double kz, double k, int i, int j, int l, int bin, 
               double *A_0, double *B_xx, double *B_yy, double *B_zz, double *B_xy, 
               double *B_yz, double *B_xz, double *P0, double *P2r, double *P2i, 
               double *P2, double *N0, double grid_cor)
{
    double A_0r = A_0[(2*l    ) + 2*(N_z/2 + 1)*(j+N_y*i)];
    double A_0i = A_0[(2*l + 1) + 2*(N_z/2 + 1)*(j+N_y*i)];
    double B_xxr = B_xx[(2*l    ) + 2*(N_z/2 + 1)*(j+N_y*i)];
    double B_xxi = B_xx[(2*l + 1) + 2*(N_z/2 + 1)*(j+N_y*i)];
    double B_yyr = B_yy[(2*l    ) + 2*(N_z/2 + 1)*(j+N_y*i)];
    double B_yyi = B_yy[(2*l + 1) + 2*(N_z/2 + 1)*(j+N_y*i)];
    double B_zzr = B_zz[(2*l    ) + 2*(N_z/2 + 1)*(j+N_y*i)];
    double B_zzi = B_zz[(2*l + 1) + 2*(N_z/2 + 1)*(j+N_y*i)];
    double B_xyr = B_xy[(2*l    ) + 2*(N_z/2 + 1)*(j+N_y*i)];
    double B_xyi = B_xy[(2*l + 1) + 2*(N_z/2 + 1)*(j+N_y*i)];
    double B_yzr = B_yz[(2*l    ) + 2*(N_z/2 + 1)*(j+N_y*i)];
    double B_yzi = B_yz[(2*l + 1) + 2*(N_z/2 + 1)*(j+N_y*i)];
    double B_xzr = B_xz[(2*l    ) + 2*(N_z/2 + 1)*(j+N_y*i)];
    double B_xzi = B_xz[(2*l + 1) + 2*(N_z/2 + 1)*(j+N_y*i)];
    
    P0[bin] += (A_0r*A_0r+A_0i*A_0i)*grid_cor*grid_cor;
    ++N0[bin];
    
    if (k != 0) {
        double A_2r = (kx*kx*B_xxr+ky*ky*B_yyr+kz*kz*B_zzr+ 2*kx*ky*B_xyr+ 2*ky*kz*B_yzr+ 
                       2*kx*kz*B_xzr)/(k*k);
        double A_2i = (kx*kx*B_xxi+ky*ky*B_yyi+kz*kz*B_zzi+ 2*kx*ky*B_xyi+ 2*ky*kz*B_yzi+ 
                       2*kx*kz*B_xzi)/(k*k);
        
        double mu = kz/k;
        P2[bin] += ((A_0r*A_0r+A_0i*A_0i)*((3*mu*mu-1))*0.5)*grid_cor*grid_cor;
        P2r[bin] += (3*A_0r*A_2r+3*A_0i*A_2i-A_0r*A_0r-A_0i*A_0i)*grid_cor*grid_cor;
        P2i[bin] += 3*(A_0i*A_2r-A_0r*A_2i)*grid_cor*grid_cor;
    }
}

void NGP(double3 pos, double3 rmin, double3 binSize, int Nz, int Ny, double *nden) {
    int4 ijk = {0, 0, 0, 0};
    
    ijk.x = (pos.x-rmin.x)/(binSize.x);
    ijk.y = (pos.y-rmin.y)/(binSize.y);
    ijk.z = (pos.z-rmin.z)/(binSize.z);
    ijk.w = ijk.z + 2*(Nz/2 + 1)*(ijk.y+Ny*ijk.x);
    
    ++nden[ijk.w];
}

void CIC(double3 pos, double3 rmin, double3 binSize, int Nz, int Ny, double *nden) {
    int4 ngp = {0, 0, 0, 0};
    int8 gps = {0, 0, 0, 0, 0, 0, 0, 0};
    double3 r_c = {0.0, 0.0, 0.0};
    double V = binSize.x*binSize.y*binSize.z;
    
    ngp.x = (pos.x-rmin.x)/(binSize.x);
    ngp.y = (pos.y-rmin.y)/(binSize.y);
    ngp.z = (pos.z-rmin.z)/(binSize.z);
    ngp.w = ngp.z + N_z*(ngp.y+Ny*ngp.x);
    
    r_c.x = (ngp.x+0.5)*binSize.x+rmin.x;
    r_c.y = (ngp.y+0.5)*binSize.y+rmin.y;
    r_c.z = (ngp.z+0.5)*binSize.z+rmin.z;
    
    double dx = pos.x-r_c.x;
    double dy = pos.y-r_c.y;
    double dz = pos.z-r_c.z;
    
    int xshift = int(dx/abs(dx));
    int yshift = int(dy/abs(dy));
    int zshift = int(dz/abs(dz));
    
    gps.cic1 = ngp.w; // Nearest grid point
    gps.cic2 = ngp.z + Nz*(ngp.y+Ny*(ngp.x+xshift)); // x-shift only
    gps.cic3 = ngp.z + Nz*((ngp.y+yshift)+Ny*ngp.x); // y-shift only
    gps.cic4 = (ngp.z+zshift) + Nz*(ngp.y+Ny*ngp.x); // z-shift only
    gps.cic5 = ngp.z + Nz*((ngp.y+yshift)+Ny*(ngp.x+xshift)); // x and y shift
    gps.cic6 = (ngp.z+zshift) + Nz*((ngp.y+yshift)+Ny*ngp.x); // y and z shift
    gps.cic7 = (ngp.z+zshift) + Nz*(ngp.y+Ny*(ngp.x+xshift)); // x and z shift
    gps.cic8 = (ngp.z+zshift) + Nz*((ngp.y+yshift)+Ny*(ngp.x+xshift)); // x, y and z shift
    
    dx = abs(dx);
    dy = abs(dy);
    dz = abs(dz);
    
    if (gps.cic1 < N_tot && gps.cic1 >= 0) nden[gps.cic1] += (binSize.x-dx)*(binSize.y-dy)*(binSize.z-dz)/V;
    if (gps.cic2 < N_tot && gps.cic2 >= 0) nden[gps.cic2] += dx*(binSize.y-dy)*(binSize.z-dz)/V;
    if (gps.cic3 < N_tot && gps.cic3 >= 0) nden[gps.cic3] += (binSize.x-dx)*dy*(binSize.z-dz)/V;
    if (gps.cic4 < N_tot && gps.cic4 >= 0) nden[gps.cic4] += (binSize.x-dx)*(binSize.y-dy)*dz/V;
    if (gps.cic5 < N_tot && gps.cic5 >= 0) nden[gps.cic5] += dx*dy*(binSize.z-dz)/V;
    if (gps.cic6 < N_tot && gps.cic6 >= 0) nden[gps.cic6] += (binSize.x-dx)*dy*dz/V;
    if (gps.cic7 < N_tot && gps.cic7 >= 0) nden[gps.cic7] += dx*(binSize.y-dy)*dz/V;
    if (gps.cic8 < N_tot && gps.cic8 >= 0) nden[gps.cic8] += dx*dy*dz/V;
}

int main() {
    ofstream fout;
    ifstream fin;
    ifstream galinfoin;
    
    omp_set_num_threads(numCores);
    
    fftw_threads_init();
    rfftwnd_plan dp_r2c;
    dp_r2c = rfftw3d_create_plan(N_x, N_y, N_z,FFTW_REAL_TO_COMPLEX, FFTW_MEASURE);
    
    double *kvec = new double[N_x];
    
    fftfreq(kvec, N_x, L_x);
    
    string galinfofile = filename(galinfobase, extdat);
    
    galinfoin.open(galinfofile.c_str(),ios::in);
    for (int mock = 0; mock < numMocks; ++mock) {
        double begin_mock = omp_get_wtime();
        
        string galfile;
        //string ranfile = filename(ranbase, extdat);
        
        int numGal = 0;
        //int numRan = 0;
        
        galinfoin >> numGal >> galfile;
        
        cout << "Reading in " << numGal << " galaxies from file " << galfile << "\n";
        
        //double3 r_min = {-512.0, -512.0, 1024.0};
        //double3 r_max = {512.0, 512.0, 2048.0};
        double3 r_min = {0.0, 0.0, 0.0};
        double3 r_max = {L_x, L_y, L_z};
        double3 dr = {0.0, 0.0, 0.0};
        
        cout << "Reading in galaxies file... ";
        cout.flush();
        fin.open(galfile.c_str(),ios::in);
        double3 *r_g = new double3[numGal];
        for (int i = 0; i < numGal; ++i) {
            fin >> r_g[i].x >> r_g[i].y >> r_g[i].z;
            //cout << r_g[i].x << ", " << r_g[i].y << ", " << r_g[i].z << "\n";
            //r_g[i].x -= 512.0;
            //r_g[i].y -= 512.0;
            //r_g[i].z += 2048.0;
            
            //if (r_g[i].x < r_min.x) {r_min.x = r_g[i].x;}
            //if (r_g[i].x > r_max.x) {r_max.x = r_g[i].x;}
            //if (r_g[i].y < r_min.y) {r_min.y = r_g[i].y;}
            //if (r_g[i].y > r_max.y) {r_max.y = r_g[i].y;}
            //if (r_g[i].z < r_min.z) {r_min.z = r_g[i].z;}
            //if (r_g[i].z > r_max.z) {r_max.z = r_g[i].z;}
        }
        fin.close();
        cout << "Done!\n";
        
        dr.x = (r_max.x-r_min.x)/N_x;
        dr.y = (r_max.y-r_min.y)/N_y;
        dr.z = (r_max.z-r_min.z)/N_z;
        
        double V = (r_max.x-r_min.x)*(r_max.y-r_min.y)*(r_max.z-r_min.z);
        double nbar = double(numGal)/V;
        
        cout << "nbar = " << nbar << "\n";
        
        int4 *indicies = new int4[numGal];
        double *nden = new double[N_tot];
        double *delta = new double[N_tot];
        fftw_complex *deltak = new fftw_complex[N_x*N_y*(N_z/2 + 1)];
        
        for (int i = 0; i < N_tot; ++i) {
            nden[i] = 0.0;
            delta[i] = 0.0;
            if (i < N_x*N_y*(N_z/2 + 1)) {
                deltak[i].re = 0.0;
                deltak[i].im = 0.0;
            }
        }
//         double *B_xx = new double[N_tot];
//         double *B_yy = new double[N_tot];
//         double *B_zz = new double[N_tot];
//         double *B_xy = new double[N_tot];
//         double *B_yz = new double[N_tot];
//         double *B_xz = new double[N_tot];
//         double3 *r = new double3[N_tot];
        
//         for (int i = 0; i < N_x; ++i) {
//             for (int j = 0; j < N_y; ++j) {
//                 for (int l = 0; l < N_z; ++l) {
//                         int ind = l + 2*(N_z/2 + 1)*(j+N_y*i);
//                         double3 rhat = {0.0, 0.0, 0.0};
//                         rhat.x = i*dr.x+r_min.x;
//                         rhat.y = j*dr.y+r_min.y;
//                         rhat.z = l*dr.z+r_min.z;
//                         
//                         double rmag = sqrt(rhat.x*rhat.x+rhat.y*rhat.y+rhat.z*rhat.z);
//                         
//                         rhat.x /= rmag;
//                         rhat.y /= rmag;
//                         rhat.z /= rmag;
//                         
//                         r[ind].x = rhat.x;
//                         r[ind].y = rhat.y;
//                         r[ind].z = rhat.z;
//                 }
//             }
//         }
        
        
        cout << "Calculating indicies... ";
        cout.flush();
        for (int i = 0; i < numGal; ++i) {
            CIC(r_g[i], r_min, dr, N_z, N_y, nden);
//             double rsqinv = 1/(r_g[i].x*r_g[i].x+r_g[i].y*r_g[i].y+r_g[i].z*r_g[i].z);
//             indicies[i] = index(r_g[i], r_min, dr, N_z, N_y);
//             if (indicies[i].w < N_tot) {
//                 ++nden[indicies[i].w];
//                 B_xx[indicies[i].w] += r_g[i].x*r_g[i].x*rsqinv;
//                 B_yy[indicies[i].w] += r_g[i].y*r_g[i].y*rsqinv; 
//                 B_zz[indicies[i].w] += r_g[i].z*r_g[i].z*rsqinv;
//                 B_xy[indicies[i].w] += r_g[i].x*r_g[i].y*rsqinv;
//                 B_yz[indicies[i].w] += r_g[i].y*r_g[i].z*rsqinv;
//                 B_xz[indicies[i].w] += r_g[i].x*r_g[i].z*rsqinv;
//             }             
            
        }
        cout << "Done!\n";
        
        cout << "Binning galaxies and finding density contrast... ";
        cout.flush();
#pragma omp parallel for
        for (int i = 0; i < N_tot; i++) {
            double n_g = nbar*dr.x*dr.y*dr.z;
            nden[i] /= n_g;
            delta[i] = nden[i] - 1.0;
//             B_xx[i] = r[i].x*r[i].x*delta[i];
//             B_yy[i] = r[i].y*r[i].y*delta[i];
//             B_zz[i] = r[i].z*r[i].z*delta[i];
//             B_xy[i] = r[i].x*r[i].y*delta[i];
//             B_yz[i] = r[i].y*r[i].z*delta[i];
//             B_xz[i] = r[i].x*r[i].z*delta[i];
//             B_xx[i] /= n_g;
//             B_yy[i] /= n_g;
//             B_zz[i] /= n_g;
//             B_xy[i] /= n_g;
//             B_yz[i] /= n_g;
//             B_xz[i] /= n_g;
//             B_xx[i] -= 1;
//             B_yy[i] -= 1;
//             B_zz[i] -= 1;
//             B_xy[i] -= 1;
//             B_yz[i] -= 1;
//             B_xz[i] -= 1;
        }
        cout << "Done!\n";
        
        cout << "Performing FFTs... ";
        cout.flush();
        double fftTime = omp_get_wtime();
        rfftwnd_threads_one_real_to_complex(numCores,dp_r2c,(fftw_real*)delta,deltak);
//         rfftwnd_threads_one_real_to_complex(numCores,dp_r2c,(fftw_real*)B_xx,NULL);
//         rfftwnd_threads_one_real_to_complex(numCores,dp_r2c,(fftw_real*)B_yy,NULL);
//         rfftwnd_threads_one_real_to_complex(numCores,dp_r2c,(fftw_real*)B_zz,NULL);
//         rfftwnd_threads_one_real_to_complex(numCores,dp_r2c,(fftw_real*)B_xy,NULL);
//         rfftwnd_threads_one_real_to_complex(numCores,dp_r2c,(fftw_real*)B_yz,NULL);
//         rfftwnd_threads_one_real_to_complex(numCores,dp_r2c,(fftw_real*)B_xz,NULL);
        cout << "Done!\n";
        cout << "Time for FFTs: " << omp_get_wtime()-fftTime << " s\n";
        
        double nyx = pi/dr.x;
        double nyy = pi/dr.y;
        double nyz = pi/dr.z;
        
        double minNy = nyx;
        if (nyy < minNy) {minNy = nyy;}
        if (nyz < minNy) {minNy = nyz;}
        
        double *P0 = new double[NB];
        double *P2 = new double[NB];
        double *Nk = new double[NB];
        
        for (int i = 0; i < NB; ++i) {
            P0[i] = 0.0;
            P2[i] = 0.0;
            Nk[i] = 0.0;
        }
        
        //double kx = 0.0, ky = 0.0, kz = 0.0;
        
        cout << "Binning Frequencies... ";
        cout.flush();
        
        int BinnedFreq = 0;
        
        for (int i = 0; i <= N_x/4; ++i) {
            
            double kx = 2.0*pi*kvec[i];
            
            for (int j = 0; j <= N_y/4; ++j) {
                
                double ky = 2.0*pi*kvec[j];
                
                for (int l = 0; l <= N_z/4; ++l) {
                    
                    double kz = 2.0*pi*kvec[l];
                    
                    double k = sqrt(kx*kx+ky*ky+kz*kz);
                    int bin = (k-k_min)/binWidth;
                    
                    if (bin >= 0 && bin < NB) {
                        double ax = (kx*dr.x)/2.0 + 1E-17;
                        double ay = (ky*dr.y)/2.0 + 1E-17;
                        double az = (kz*dr.z)/2.0 + 1E-17;
                        
                        double sincx = sin(ax)/ax;
                        double sincy = sin(ay)/ay;
                        double sincz = sin(az)/az;
                        
                        double grid_cor = 1.0/((sincx*sincy*sincz)*(sincx*sincy*sincz));
                        
//                         PowerFF(kx, ky, kz, k, i, j, l, bin, deltak, P0, Nk, P2, grid_cor);
//                         BinnedFreq++;
                        
                        if (i != 0 && j != 0) {
                            PowerFF(kx, ky, kz, k, i, j, l, bin, deltak, P0, Nk, P2, grid_cor);
                            
                            PowerFF(-kx, ky, kz, k, N_x-i, j, l, bin, deltak, P0, Nk, P2, grid_cor);
                            
                            PowerFF(-kx, -ky, kz, k, N_x-i, N_y-j, l, bin, deltak, P0, Nk, P2, grid_cor);
                            
                            PowerFF(kx, -ky, kz, k, i, N_y-j, l, bin, deltak, P0, Nk, P2, grid_cor);
                            BinnedFreq += 4;
                        } else if(i != 0 && j == 0) {
                            PowerFF(kx, ky, kz, k, i, j, l, bin, deltak, P0, Nk, P2, grid_cor);
                            
                            PowerFF(-kx, ky, kz, k, N_x-i, j, l, bin, deltak, P0, Nk, P2, grid_cor);
                            BinnedFreq += 2;
                        } else if(i == 0 && j != 0) {
                            PowerFF(kx, ky, kz, k, i, j, l, bin, deltak, P0, Nk, P2, grid_cor);
                            
                            PowerFF(kx, -ky, kz, k, i, N_y-j, l, bin, deltak, P0, Nk, P2, grid_cor);
                            BinnedFreq += 2;
                        } else if(i == 0 && j == 0) {
                            PowerFF(kx, ky, kz, k, i, j, l, bin, deltak, P0, Nk, P2, grid_cor);
                            BinnedFreq += 1;
                        }
                    }
                }
            }
        }
        cout << "Done!\n";
        cout << "Number of binned frequencies: " << BinnedFreq << "\n";
        
        fout.open("NumKmodes.dat",ios::out);
        for (int i = 0; i < NB; ++i) {
            if(Nk[i] > 0) {
                fout << k_min+(i+0.5)*binWidth << " " << Nk[i] << "\n";
                //P0[i] /= Nk[i];
                //double num = V;
                double den = pow(N_x,6);
                P0[i] /= (Nk[i]);
                P0[i] *= V/den;
                P0[i] -= 1.0/nbar;
                P2[i] /= (Nk[i]);
                P2[i] *= V/den;
                //P2[i] -= 1/nbar;
                //if (i == 0) {Nk[i]--;}
                //P2r[i] *= ((5*num)/(2*Nk[i]*den));
                //P2i[i] *= ((5*num)/(2*Nk[i]*den));
                //P2r[i] -= 1/nbar;
                //P2i[i] -= 1/nbar;
            }
        }
        fout.close();
        
        string pkfile = filenumber(pkbase, mock+startNum, extdat);
        fout.open(pkfile.c_str(),ios::out);
        for (int i = 0; i < numKOut; ++i) {
            double kp = k_min+(i+0.5)*binWidth;
            fout << kp << " " << P0[i] << " " << P2[i] << "\n"; // << P2i[i] << " " << P2[i] << "\n";
        }
        fout.close();
        
        delete[] indicies;
        delete[] nden;
        delete[] delta;
        delete[] P0;
        delete[] deltak;
        delete[] r_g;
//         delete[] P2r;
//         delete[] P2i;
        delete[] Nk;
//         delete[] B_xx;
//         delete[] B_yy;
//         delete[] B_zz;
//         delete[] B_xy;
//         delete[] B_yz;
//         delete[] B_xz;
//         delete[] r;
        delete[] P2;
        
        cout << "Time to process mock: " << omp_get_wtime()-begin_mock << " seconds\n";
    }
    galinfoin.close();
    rfftwnd_destroy_plan(dp_r2c);
    
    delete[] kvec;
    
    return 0;
}