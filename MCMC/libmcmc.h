#ifndef _LIBMCMC_H_
#define _LIBMCMC_H_

#include <string>

double chisqCalc(double *data, double *model, gsl_matrix *Psi, int N);

double likelihood(double chisq, double detPsi);

void readCov(std::string covfile, int N, gsl_matrix *cov, std::string format);

void calcPsi(gsl_matrix *cov, gsl_matrix *Psi, double *detPsi, int N, int samples);

#endif
