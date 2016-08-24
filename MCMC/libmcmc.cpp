#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <model.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_permutation.h>

long double pi = acos(-1.0);

double chisqCalc(std::vector<double> data, std::vector<double> model, std::vector<double> Psi, 
                 int N) {
    double result = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int j = i; j < N; ++j) {
            result += (data[i]-model[i])*Psi[j+N*i]*(data[j]-model[j]);
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

void readData(std::string file, int numVals, double *data, double *xvals) {
    std::ifstream fin;
    
    fin.open(file.c_str(), std::ios::in);
    for (int i = 0; i < numVals; ++i) {
        fin >> xvals[i] >> data[i];
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

double varianceCalc(std::vector<double> data, std::vector<double> modParams,
                    std::vector<double> paramMins, std::vector<double> paramMaxs, 
                    std::vector<bool> limitParams, std::vector<double> xvals, int N, 
                    int numParams, std::vector<double> Psi, double detPsi) {
    double variance = 0.1;
    double acceptance = 1.0;
    
    std::cout << "         Declaring random generator stuff..." << std::endl;
    std::random_device seeder;
    std::mt19937_64 gen(seeder());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    std::cout << "         Declaring some vectors..." << std::endl;
    std::vector<double> currentVals(numParams);
    std::vector<double> vars(numParams);
    std::vector<double> model(N);
    
    std::cout << "         Calculating initial model, chi^2 and likelihood..." << std::endl;
    ModelVals(model, modParams, numParams, N, xvals);
    double chisq_initial = chisqCalc(data, model, Psi, N);
    double L_initial = likelihood(chisq_initial, detPsi);
    
    std::cout << "         Tuning..." << std::endl;
    while (acceptance >= 0.235 || acceptance <= 0.233) {
        int accept = 0;
        for (int i = 0; i < numParams; ++i) {
            currentVals[i] = modParams[i];
            vars[i] = modParams[i]*variance;
        }
        double Li = L_initial;
        std::vector<double> trialParams(numParams);
        for (int i = 0; i < 1000; ++i) {
            for (int param = 0; param < numParams; ++param) {
                if (limitParams[param]) {
                    if (currentVals[param]+vars[param] > paramMaxs[param]) {
                        double center = paramMaxs[param]-vars[param];
                        trialParams[param] = center+vars[param]*dist(gen);
                    } else if (currentVals[param]-vars[param] < paramMins[param]) {
                        double center = paramMins[param]+vars[param];
                        trialParams[param] = center+vars[param]*dist(gen);
                    } else {
                        trialParams[param] = currentVals[param]+vars[param]*dist(gen);
                    }
                } else {
                    trialParams[param] = currentVals[param]+vars[param]*dist(gen);
                }
            }
            
            ModelVals(model, trialParams, numParams, N, xvals);
            double chisq = chisqCalc(data, model, Psi, N);
            double L = likelihood(chisq, detPsi);
            double ratio = L/Li;
            double test = (dist(gen) + 1.0)/2.0;
            
            if (ratio > test) {
                for (int param = 0; param < numParams; ++param) {
                    currentVals[param] = trialParams[param];
                }
                Li = L;
                ++accept;
            }
        }
        
        acceptance = double(accept)/1000.0;
        std::cout << "      Acceptance ratio =";
        std::cout.width(15);
        std::cout << acceptance << "\r";
        std::cout.flush();
        if (acceptance >= 0.235) {
            variance *= 1.01;
        }
        if (acceptance <= 0.233) {
            variance *= 0.99;
        }
    }
    std::cout << std::endl;
    
    return variance;
}
