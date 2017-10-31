#ifndef _FILE_READER_H_
#define _FILE_READER_H_

#include <vector>
#include <string>

size_t readPatchy(std::string file, std::vector<double> &nden, double red_min, double red_max, double P_FKP,
                double Omega_M, double Omega_L, vec3<double> r_min, vec3<double> L, vec3<int> N,
                vec3<double> &pk_nbw, vec3<double> &bk_nbw, bool randoms);

size_t readQPM(std::string file, std::vector<double> &nden, double red_min, double red_max, double P_FKP,
               double Omega_M, double Omega_L, vec3<double> r_min, vec3<double> L, vec3<int> N,
               vec3<double> &pk_nbw, vec3<double> &bk_nbw, bool randoms);

void readFits(std::string file, std::vector<double> &nden, std::vector<std::string> cols, bool randoms);

#endif
