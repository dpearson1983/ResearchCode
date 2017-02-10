#ifndef _FILEREADER_H_
#define _FILEREADER_H_

void initArray(double *array, int N);

double readFits(std::string file, std::vector<std::string> hdus, int hduTableNum, 
              double *&nden, double res, vec3<double> &L, vec3<int> &N, vec3<double> &r_min, 
              vec3<double> &pk_nbw, vec3<double> &bk_nbw, double P_w, int flags, double Omega_M, 
              double Omega_L, double z_min, double z_max);

double readFits(std::string file, std::vector<std::string> hdus, int hduTableNum, 
              double *&nden, vec3<double> L, vec3<int> N, vec3<double> r_min, 
              vec3<double> &pk_nbw, vec3<double> &bk_nbw, double P_w, int flags, double Omega_M, 
              double Omega_L, double z_min, double z_max);

#endif
