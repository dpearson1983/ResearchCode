#ifndef _LOGNORMAL_H_
#define _LOGNORMAL_H_

void fftfreq1d(double *k, int N, double L);

void fftfreq2d(double2 *k, int2 N, double2 L);

void fftfreq3d(double3 *k, int3 N, double3 L);

void Pk_CAMB(int3 N, double3 *vk, double *k_mag, double *P, int numKVals, double *Pk);

void Genddk(int3 N, double3 L, double3 *vk, double b, double f, double *Pk,
            fftw_complex *dk3d);

void Genddk_2Tracer(int3 N, double3 L, double3 *vk, double2 b, double f, double *Pk,
                    fftw_complex *dk3d1, fftw_complex *dk3d2);

void Sampdk(int3 N, double3 *vk, fftw_complex *dk3di, fftw_complex *dk3d);

void Sampdk_2Tracer(int3 N, double3 *vk, fftw_complex *dk3di, fftw_complex *dk3d1,
                    fftw_complex *dk3d2, double *ratio);

void Genddr(int3 N, double3 L, double nbar, std::string file, double variance,
            fftw_real *dr3d);

void Genddr_2Tracer(int3 N, double3 L, double2 nbar, string2 file, double2 variance,
                    fftw_real *dr3d1, fftw_real *dr3d2);

#endif