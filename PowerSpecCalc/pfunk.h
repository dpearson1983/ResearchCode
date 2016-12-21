#ifndef _PFUNK_H_
#define _PFUNK_H_

#include <pods.h>
#include <fftw3.h>
#include <gsl/gsl_integration.h>

double rz(double red, double O_m, double O_L, gsl_integration_workspace *w);

double r2z(double r, double O_m, double O_L, double tolerance, gsl_integration_workspace *w);

void initArray(double *array, long int N);

void initArray(fftw_complex *array, long int N);

void initArray(int *array, long int N);

void initArray(double *dr, fftw_complex *dk, long int N_r, long int N_k);

void equatorial2cartesian(galaxy *gals, int N, double O_m, double O_L, gsl_integration_workspace *w);

void equatorial2cartesian(galaxyf *gals, double3 *pos, int N, double O_m, double O_L, gsl_integration_workspace *w);

void cartesian2equatorial(galaxy *gals, int N, double O_m, double O_L, gsl_integration_workspace *w);

double binNGP(galaxy *gals, double *nden, double3 dr, int numGals, int3 N);

double binNGP(galaxy *gals, double *nden, double3 dr, int numGals, int3 N, double3 r_obs);

double binCIC(galaxy *gals, double *nden, double3 dr, int numGals, int3 N);

double binCIC(galaxy *gals, double *nden, double3 dr, int numGals, int3 N, double3 r_obs);

double3 wbinCIC(galaxy *gals, double *nden, double3 dr, int numGals, int3 N, double3 rmin, double *redshift, double *nbar, int nVals, double Pk_FKP);

double3 wbinCIC(galaxyf *gals, double *nden, double3 dr, int numGals, int3 N, double3 rmin, double *redshift, double *nbar, int nVals, double Pk_FKP, double Omega_M, double Omega_L);

double3 wbinNGP(galaxyf *gals, double *nden, double3 dr, int numGals, int3 N, double3 rmin, double *redshift, double *nbar, int nVals, double Pk_FKP, double Omega_M, double Omega_L);

void freqBinMono(fftw_complex *dk, double *P_0, int *N_k, int3 N, double3 L, double shotnoise, 
                 double k_min, double k_max, int kBins, bool corr, int type);

void freqBinPP(fftw_complex *dk3d, double *P_0, double *P_2, double *P_2shot, int *N_k, int3 N, 
               double3 L, double shotnoise, double k_min, double k_max, int kBins, bool corr, 
               int type);

void normalizePk(double *P_0, double *P_2, double *P_2shot, int *N_k, double gal_nbsqwsq, int N);

void correct_discreteness(std::string cor_file, double *P_0, double *P_2, int kbins);

void calcB_ij(double *B_ij, double *nden_gal, double *nden_ran, double alpha, double3 dr,
               double3 rmin, double3 robs, int3 N, std::string ij);

void accumulateA_2(fftw_complex *A_2, fftw_complex *B_ij, int3 N, double3 L, std::string ij);

void freqBinBS(fftw_complex *A_0, fftw_complex *A_2, double *P_0, double *P_2, int *N_k, int3 N,
               double3 L, double shotnoise, double kmin, double kmax, int kBins, bool corr, int type);

void normalizePk(double *P_0, double *P_2, int *N_k, double gal_nbsqwsq, int N);

void normalizePk(double *P_0, int *N_k, double gal_nbsqwsq, int N);

void displayFFT(fftw_complex *dk);

#endif
