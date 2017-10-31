#ifndef _FILE_READER_H_
#define _FILE_READER_H_

#include <vector>
#include <string>
#include "density_field.h"
#include "galaxy.h"
#include "cosmology.h"
#include "tpods.h"

size_t readPatchy(std::string file, desityField &nden, cosmology &cos, vec2<double> red_lim, double P_FKP, 
                  bool randoms);

size_t readQPM(std::string file,desityField &nden, cosmology &cos, vec2<double> red_lim, double P_FKP, 
               bool randoms, gsl_spline *NofZ, gsl_interp_accel *acc);

void readFits(std::string file, std::vector<double> &nden, std::vector<std::string> cols, bool randoms);

#endif
