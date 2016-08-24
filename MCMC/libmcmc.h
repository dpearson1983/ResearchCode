#ifndef _LIBMCMC_H_
#define _LIBMCMC_H_

#include <string>
#include <gsl/gsl_matrix.h>
#include <vector>

double chisqCalc(std::vector<double> data, std::vector<double> model, std::vector<double> Psi, 
                 int N);

double likelihood(double chisq, double detPsi);

void readCov(std::string covfile, int N, gsl_matrix *cov, std::string format);

void readData(std::string file, int numVals, double *data, double *xvals);

void calcPsi(gsl_matrix *cov, gsl_matrix *Psi, double *detPsi, int N, int samples);

double varianceCalc(std::vector<double> data, std::vector<double> modParams,
                    std::vector<double> paramMins, std::vector<double> paramMaxs, 
                    std::vector<bool> limitParams, std::vector<double> xvals, int N, 
                    int numParams, std::vector<double> Psi, double detPsi);

#endif
