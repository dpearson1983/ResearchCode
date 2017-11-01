#ifndef _ADD_FUNCS_H_
#define _ADD_FUNCS_H_

#include <vector>
#include <string>
#include <sstream>
#include <constants.h>
#include "tpods.h"

std::vector<double> fftfreq(int N, double L);

std::vector<double> myfreq(int N, double L);

double gridCorCIC(vec3<double> k, vec3<double> dr);

int kMatch(double ks, std::vector<double> &kb, double L);

std::string filename(std::string base, int digits, int num, std::string ext);

#endif
