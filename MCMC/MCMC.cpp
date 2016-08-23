#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <omp.h>
#include <harppi.h>
#include <libmcmc.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_permutation.h>
#include "model.h"

std::string filename(std::string filebase, int digits, int filenum, std::string fileext) {
    std::stringstream file;
    file << filebase << std::setw(digits) << std::setfill('0') << filenum << fileext;
    return file.str();
}

int main(int argc, char *argv[]) {
    if (argc == 1) {
        std::cout << "ERROR: Expected parameter file.\n";
        std::cout << "       Usage: " << argv[0] << " ParameterFile.params\n";
        std::cout << "       Note that the parameter file can have any extension\n";
        std::cout << "       so long as it is a plain text file formated as specified\n";
        std::cout << "       libharppi documentation." << std::endl;
        return 0;
    }
    parameters p;
    p.readParams(argv[1]);
    
    std::ifstream fin;
    std::ofstream fout;
    
    gsl_matrix *cov = gsl_matrix_alloc(p.geti("numVals"), p.geti("numVals"));
    gsl_matrix *Psi = gsl_matrix_alloc(p.geti("numVals"), p.gets("numVals"));
    double detPsi;
    
    readCov(p.gets("covFile"), p.geti("numVals"), cov, p.gets("covFormat"));
    calcPsi(cov, Psi, &detPsi, p.geti("numVals"), p.geti("numMeas"));
    std::vector<double> PsiVec(p.geti("numVals")*p.geti("numVals"));
    for (int i = 0; i < p.geti("numVals"); ++i) {
        for (int j = 0; j < p.geti("numVals"); ++j) {
            PsiVec[j+p.geti("numVals")*i] = gsl_matrix_get(Psi, i, j);
        }
    }
    
    gsl_matrix_free(cov);
    gsl_matrix_free(Psi);
    
    for (int file = p.geti("startNum"); file < p.geti("numFiles")+p.geti("startNum"); ++file) {
        std::string infile = filename(p.gets("inBase"), p.geti("digits"), file, p.gets("ext"));
        std::string outfile = filename(p.gets("outBase"), p.geti("digits"), file, p.gets("ext"));
        
        std::vector< double > modParams;
        std::vector< std::string > paramNames;
        std::vector< bool > limitParams;
        std::vector< double > paramMins;
        std::vector< double > paramMaxs;
        for (int i = 0; i < p.geti("numParams"); ++i) {
            modParams.push_back(p.getd("paramVals", i));
            paramNames.push_back(p.gets("paramNames", i));
            if (p.checkParam("limitParams")) {
                limitParams.push_back(p.getb("limitParams", i));
                paramMins.push_back(p.getd("paramMins", i));
                paramMaxs.push_back(p.getd("paramMins", i));
            }
        }
        
        std::vector< double > data(p.geti("numVals"));
        std::vector< double > xvals(p.geti("numVals"));
        readData(infile, p.geti("numVals"), &data[0], &xvals[0]);
        
        double variance = variaceCalc(data, modParams, paramMins, paramMaxs, limitParams,
                                      xvals, p.geti("numVals"), p.geti("numParams"), PsiVec,
                                      detPsi);
        
        std::vector< std::vector< std::vector<double> > > reals(p.geti("numThreads"));
        std::vector< std::vector<double> > averages(p.geti("numThreads"));
        std::vector< std::vector<double> > stdevs(p.geti("numThreads"));
        std::vector< int > draws(p.geti("numThreads"));
        
        int numDraws = p.geti("numDraws");
        int numParams = p.geti("numParams");
        int numVals = p.geti("numVals");
        int numThreads = p.geti("numThreads");
        std::vector< double > vars(numParams);
        for (int i = 0; i < numParams; ++i)
            vars[i] = modParams[i]*variance;
        
        for (int i = 0; i < numThreads; ++i) {
            ++draws[i];
            for (int param = 0; param < numParams; ++param) {
                averages[i].push_back(modParams[param]);
                stdevs[i].push_back(0.0);
            }
        }
        
        omp_set_num_threads(numThreads);
        bool converged = false;
        double criteria = p.getd("convergenceCriteria");
        int loop = 0;
        while (!converged) {
            #pragma omp parallel
            {
                std::random_device seeder;
                std::mt19937_64 gen(seeder());
                std::uniform_real_distribution<double> dist(-1.0,1.0);
                int tid = omp_get_thread_num();
                std::vector< double > currentParams(numParams);
                std::vector< double > trialParams(numParams);
                std::vector< double > model(numVals);
                if (loop != 0) {
                    int lastReal = loop*numDraws - 1;
                    for (int i = 0; i < numParams; ++i)
                        currentParams[i] = reals[tid][lastReal][i];
                } else {
                    for (int i = 0; i < numParams; ++i)
                        currentParams[i] = modParams[i];
                }
                ModelVals(&model[0], currentParams, numParasm, numVals, xvals);
                double chisq_initial = chisqCalc(data, model, PsiVec, numVals);
                double L_initial = likelihood(chisq_initial, detPsi);
                for (int draw = 0; draw < numDraws; ++draw) {
                    ++draws[tid];
                    for (int param = 0; param < numParams; ++param) {
                        if (limitParams[param]) {
                            if (currentParams[param]+vars[param] > paramMaxs[param]) {
                                double center = paramMaxs[param]-vars[param];
                                trialParams[param] = center+vars[param]*dist(gen);
                            } else if (currentParams[param]-vars[param] < paramMins[param]) {
                                double center = paramMins[param]+vars[param];
                                trialParams[param] = center+vars[param]*dist(gen);
                            } else {
                                trialParams[param] = currentParams[param]+vars[param]*dist(gen);
                            }
                        } else {
                            trialParams[param] = currentParams[param]+vars[param]*dist(gen);
                        }
                    }
                    
                    ModelVals(&model[0], trialParams, numParams, numVals, xvals);
                    double chisq = chisqCalc(data, model, PsiVec, numVals);
                    double L = likelihood(chisq, detPsi);
                    double ratio = L/L_initial;
                    double test = (dist(gen) + 1.0)/2.0;
                    
                    if (ratio > test) {
                        L_initial = L;
                        for (int param = 0; param < numParams, ++param)
                            currentParams[i] = trialParams[i];
                    }
                    
                    reals[tid].push_back(currentParams);
                    for (int param = 0; param < numParams; ++param) {
                        double avg = currentParams[param]/double(draws[tid]) + ((double(draws[tid]) - 1.0)/double(draws[tid]))*averages[tid][param];
                        stdevs[tid][param] += (currentParams[param]-averages[tid][param])*(currentParams[param]-avg);
                        averages[tid][param] = avg;
                    }
                }
            }
            // Test for convergence here
            std::vector<double> avgavg(numParams);
            std::vector<double> varavg(numParams);
            for (int i = 0; i < numThreads; ++i) {
                for (int param = 0; param < numParams; ++i) {
                    avgavg[param] += averages[i][param]/double(numThreads);
                }
            }
            
            for (int i = 0; i < numThreads; ++i) {
                for (int param = 0; param < numParams; ++i) {
                    varavg[param] += ((averages[i][param]-avgavg[param])*(averages[i][param]-avgavg[param]))/(Nthreads - 1.0);
                }
            }
            
            int paramconv = 0;
            for (int param = 0; param < numParams; ++param) {
                if (varavg[param] < criteria) ++paramconv;
            }
            if (paramconv == numParams) converged = true;
            
            ++loop;
        }
        
        std::vector<double> finalParams(numParams);
        std::vector<double> finalSigmas(numParams);
        long int totalDraws = 0;
        for (int i = 0; i < numThreads; ++i) {
            totalDraws += draws[i];
            for (int param = 0; param < numParams; ++param) {
                finalParams[param] += averages[i][param]/numParams;
                finalSigmas[param] += stdevs[i][param]/(numParams*(draws[i] - 1.0));
            }
        }
        
        std::vector<double> covariance(numParams*numParams);
        for (int i = 0; i < numParams; ++i) {
            for (int j = i; j < numParams; ++j) {
                for (int tid = 0; tid < numThreads; ++tid) {
                    for (int draw = 0; draw < draws[tid]; ++draw) {
                        covariance[j+i*numParams] += ((reals[tid][draw][i]-finalParams[i])*(reals[tid][draw][j]-finalParams[j]))/(totalDraws - 1.0);
                    }
                }
            }
        }
        
        std::vector<double> model(numVals);
        ModelVals(&model[0], finalParams, numParams, numVals, xvals);
        double chisq = chisqCalc(data, model, PsiVec, numVals);
        
        fout.open(outfile.c_str(), std::ios::out);
        fout.precision(15);
        fout << "Best fitting chi^2 = " << chisq << "\n";
        for (int i = 0; i < numParams; ++i) {
            fout << paramNames[i] << " " << finalParams[i] << " " << sqrt(finalSigmas[i]);
            fout << " " << sqrt(covariance[i+numParams*i]) << "\n";
        }
        fout.close();
        
        if (p.getb("covarianceOut")) {
            std::string covfile = filename(p.gets("covBase"), p.geti("digits"), file, p.gets("ext"));
            fout.open(covfile.c_str(), std::ios::out);
            fout.precision(15);
            for (int i = 0; i < numParams; ++i) {
                for (int j = i; j < numParams; ++j) {
                    
                }
            }
        }
        
        if (p.getb("realsOut")) {
            std::string realsfile = filename(p.gets("realsBase"), p.geti("digits"), file, p.gets("ext"));
        }
        
    }
    
    return 0;
}
