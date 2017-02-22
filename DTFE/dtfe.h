#ifndef _DTFE_H_
#define _DTFE_H_

#include <galaxy.h>
#include <tpods.h>

void interpDTFE(std::vector<galaxy<double>> &gals, vec3<double> r_min, vec3<double> L, vec3<int> N, 
                double *nden);

#endif
