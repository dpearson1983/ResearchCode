#ifndef _LOGNORMAL_H_
#define _LOGNORMAL_H_

double gridCorCIC(double kx, double ky, double kz, double3 binSize);

double cloudInCell(double *data, double3 r, int3 N, double3 L, double3 dL);

void Gendk(int3 N, double3 L, double b, double *kval, double *Pk, int numKVals, 
           fftw_complex *dk3d);

void Smpdk(int3 N, double3 L, double b, double h, double f, std::string dk3difile, 
           fftw_complex *dk3d, fftw_complex *vk3dx, fftw_complex *vk3dy, fftw_complex *vk3dz);

void Gendr(int3 N, double3 L, double *nbar, int numTracers, std::string file, double variance,
           double *dr3d, double *vr3dx, double *vr3dy, double *vr3dz, 
           std::vector< double > b, double bias);

void Gendr_interp(int3 N, double3 L, double *nbar, int numTracers, std::string file,
                  double variance, double *dr3d, double *vr3dx, double *vr3dy, double *vr3dz,
                  std::vector< double > b, double bias);

#endif
