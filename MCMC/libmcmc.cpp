#include <fstream>
#include <string>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_permutation.h>

void ModelVals(double *model, double *modparams, int numParams, int N, double xmin, 
               double xbinWidth);

double chisqCalc(double *data, double *model, gsl_matrix *Psi, int N) {
    double result = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int j = i; j < N; ++j) {
            result += (data[i]-model[i])*gsl_matrix_get(Psi, i, j)*(data[j]-model(j));
        }
    }
    return result;
}

double likelihood(double chisq, double detPsi) {
    double likelihood = (detPsi/sqrt(2.0*pi))*exp(-0.5*chisq);
    return likelihood;
} 

void readCov(std::string covfile, int N, gsl_matrix *cov, std::string format) {
    std::ifstream fin;
    fin.open(covfile.c_str(), std::ios::in);
    
    std::string form;
    form.push_back(std::tolower(format[0]));
    form.push_back(std::tolower(format[1]));
    if (form == "sq") {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double val;
                fin >> val;
                gsl_matrix_set(cov, i, j, val);
            }
        }
    }
    if (form == "ij") {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int itemp, jtemp;
                double val;
                fin >> itemp >> jtemp >> val;
                gsl_matrix_set(cov, itemp, jtemp, val);
            }
        }
    }
    fin.close();
}

void calcPsi(gsl_matrix *cov, gsl_matrix *Psi, double *detPsi, int N, int samples) {
    gsl_permutation *perm = gsl_permutation_alloc(N);
    gsl_permutation *permLU = gsl_permutation_alloc(N);
    gsl_matrix *PsiLU = gsl_matrix_alloc(N,N);
    int s, sLU;
    gsl_linalg_LU_decomp(cov, perm, &s);
    gsl_linalg_LU_invert(cov, perm, Psi);
    
    double D = (double(N) + 1.0)/(double(samples) - 1.0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double val = (1.0 - D)*gsl_matrix_get(Psi, i, j);
            gsl_matrix_set(Psi, i, j, val);
            gsl_matrix_set(PsiLU, i, j, val);
        }
    }
    
    gsl_linalg_LU_decomp(PsiLU, permLU, &sLU);
    detPsi[0] = gsl_linalg_LU_det(PsiLU, sLU);
    
    gsl_matrix_free(PsiLU);
    gsl_permutation_free(perm);
    gsl_permutation_free(permLU);
}

double varianceCalc(double *data, double *modParams, double *paramMins, double *paramMaxs, 
                    bool *limitParams, int N, int numParams, gsl_matrix *Psi, double detPsi) {
    
    
