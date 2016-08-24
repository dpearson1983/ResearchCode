#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <random>
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
#include <model.h>

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
    p.print();
    
    std::ifstream fin;
    std::ofstream fout;
    std::ofstream sout;
    
    std::cout << "Creating matrix for covaraince and inverse..." << std::endl;
    gsl_matrix *cov = gsl_matrix_alloc(p.geti("numVals"), p.geti("numVals"));
    gsl_matrix *Psi = gsl_matrix_alloc(p.geti("numVals"), p.geti("numVals"));
    double detPsi;
    
    std::cout << "Reading in covaraince matrix..." << std::endl;
    readCov(p.gets("covFile"), p.geti("numVals"), cov, p.gets("covFormat"));
    std::cout << "Calculating inverse covariance..." << std::endl;
    calcPsi(cov, Psi, &detPsi, p.geti("numVals"), p.geti("numMeas"));
    std::cout << "Transfering to vector..." << std::endl;
    std::vector<double> PsiVec(p.geti("numVals")*p.geti("numVals"));
    for (int i = 0; i < p.geti("numVals"); ++i) {
        for (int j = 0; j < p.geti("numVals"); ++j) {
            PsiVec[j+p.geti("numVals")*i] = gsl_matrix_get(Psi, i, j);
        }
    }
    
    std::cout << "Freeing gsl matrices..." << std::endl;
    gsl_matrix_free(cov);
    gsl_matrix_free(Psi);
    
    std::cout << "Starting to process " << p.geti("numFiles") << " files..." << std::endl;
    sout.open(p.gets("summaryFile").c_str(), std::ios::out);
    for (int file = p.geti("startNum"); file < p.geti("numFiles")+p.geti("startNum"); ++file) {
        std::string infile = filename(p.gets("inBase"), p.geti("digits"), file, p.gets("ext"));
        std::string outfile = filename(p.gets("outBase"), p.geti("digits"), file, p.gets("ext"));
        
        std::cout << "    Processing " << infile << "..." << std::endl;
        
        std::cout << "      Copying parameter info to convenient storage..." << std::endl;
        std::vector< double > modParams(p.geti("numParams"));
        std::vector< std::string > paramNames(p.geti("numParams"));
        std::vector< bool > limitParams(p.geti("numParams"));
        std::vector< double > paramMins(p.geti("numParams"));
        std::vector< double > paramMaxs(p.geti("numParams"));
        for (int i = 0; i < p.geti("numParams"); ++i) {
            modParams[i] = p.getd("paramVals", i);
            paramNames[i] = p.gets("paramNames", i);
            if (p.checkParam("limitParams")) {
                limitParams[i] = p.getb("limitParams", i);
                paramMins[i] = p.getd("paramMins", i);
                paramMaxs[i] = p.getd("paramMaxs", i);
            } else {
                limitParams[i] = false;
                paramMins[i] = -100.0;
                paramMaxs[i] = 100.0;
            }
        }
        
        std::cout << "      Creating data objects..." << std::endl;
        std::vector< double > data(p.geti("numVals"));
        std::vector< double > xvals(p.geti("numVals"));
        std::cout << "      Reading in data..." << std::endl;
        readData(infile, p.geti("numVals"), &data[0], &xvals[0]);
        
        std::cout << "      Tuning acceptance ratio..." << std::endl;
        double variance = varianceCalc(data, modParams, paramMins, paramMaxs, limitParams,
                                      xvals, p.geti("numVals"), p.geti("numParams"), PsiVec,
                                      detPsi);
        std::cout << "      variance = " << variance << std::endl;
        std::cout << "      Declaring objects to store MCMC info..." << std::endl;
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
        
        std::cout << "      MCMC running chains until convergence..." << std::endl;
        omp_set_num_threads(numThreads);
        bool converged = false;
        double criteria = p.getd("convergenceCriteria");
        int loop = 0;
        while (!converged) {
            std::cout << "         .";
            std::cout.flush();
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
                ModelVals(model, currentParams, numParams, numVals, xvals);
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
                    
                    ModelVals(model, trialParams, numParams, numVals, xvals);
                    double chisq = chisqCalc(data, model, PsiVec, numVals);
                    double L = likelihood(chisq, detPsi);
                    double ratio = L/L_initial;
                    double test = (dist(gen) + 1.0)/2.0;
                    
                    if (ratio > test) {
                        L_initial = L;
                        for (int param = 0; param < numParams; ++param)
                            currentParams[param] = trialParams[param];
                    }
                    
                    reals[tid].push_back(currentParams);
                    for (int param = 0; param < numParams; ++param) {
                        double avg = currentParams[param]/double(draws[tid]) + ((double(draws[tid]) - 1.0)/double(draws[tid]))*averages[tid][param];
                        stdevs[tid][param] += (currentParams[param]-averages[tid][param])*(currentParams[param]-avg);
                        averages[tid][param] = avg;
                    }
                }
            }
            std::cout << ".";
            std::cout.flush();
            // Test for convergence here
            std::vector<double> avgavg(numParams);
            std::vector<double> varavg(numParams);
            for (int i = 0; i < numThreads; ++i) {
                for (int param = 0; param < numParams; ++param) {
                    avgavg[param] += averages[i][param]/double(numThreads);
                }
            }
            
            std::cout << ".";
            std::cout.flush();
            for (int i = 0; i < numThreads; ++i) {
                for (int param = 0; param < numParams; ++param) {
                    varavg[param] += ((averages[i][param]-avgavg[param])*(averages[i][param]-avgavg[param]))/(numThreads - 1.0);
                }
            }
            
            int paramconv = 0;
            std::cout << ".";
            std::cout.flush();
            for (int param = 0; param < numParams; ++param) {
                if (varavg[param] < criteria) ++paramconv;
            }
            if (paramconv == numParams) converged = true;
            
            std::cout << ".\r";
            std::cout.flush();
            ++loop;
        }
        
        std::cout << "      Calculating best fitting values of " << numParams << " parameters..." << std::endl;
        std::vector<double> finalParams(numParams);
        std::vector<double> finalSigmas(numParams);
        long int totalDraws = 0;
        for (int i = 0; i < numThreads; ++i) {
            totalDraws += draws[i];
            for (int param = 0; param < numParams; ++param) {
                finalParams[param] += averages[i][param]/numThreads;
                finalSigmas[param] += stdevs[i][param]/(numThreads*(draws[i] - 1.0));
            }
        }
        std::cout << "      totalDraws = " << totalDraws << std::endl;
        
//         for (int param = 0; param < numParams; ++param) {
//             std::cout << "      " << paramNames[param] << " = " << finalParams[param];
//             std::cout << " +/- " << sqrt(finalSigmas[param]) << std::endl;
//         }
        
        std::cout << "      Calculating parameter covariance for " << numParams << " parameters..." << std::endl;
        std::vector<double> covariance(numParams*numParams);
        for (int i = 0; i < numParams; ++i) {
            for (int j = i; j < numParams; ++j) {
                for (int tid = 0; tid < numThreads; ++tid) {
                    int threadDraw = draws[tid]-1.0;
                    for (long int draw = 0; draw < threadDraw; ++draw) {
                        covariance[j+i*numParams] += ((reals[tid][draw][i]-finalParams[i])*(reals[tid][draw][j]-finalParams[j]))/(totalDraws - 1.0);
                    }
                }
            }
        }
        
        std::cout << "      Calculating best fitting chi^2..." << std::endl;
        std::vector<double> model(numVals);
        ModelVals(model, finalParams, numParams, numVals, xvals);
        double chisq = chisqCalc(data, model, PsiVec, numVals);
        
        std::cout << "      Outputting summary..." << std::endl;
        sout << chisq << " ";
        for (int param = 0; param < numParams; ++param) {
            sout << finalParams[param] << " " << sqrt(finalSigmas[param]);
            if (param < numParams-1) sout << " ";
        }
        sout << std::endl;
        
        std::cout << "      Outputting basic information for the " << numParams << " parameters..." << std::endl;
        fout.open(outfile.c_str(), std::ios::out);
        //fout.precision(15);
        fout << "Best fitting chi^2 = " << chisq << "\n";
        for (int i = 0; i < numParams; ++i) {
            std::cout << "Parameter " << i << std::endl;
            fout << p.gets("paramNames", i) << " ";
            fout.flush();
            fout << finalParams[i] << " ";
            fout.flush();
            fout << sqrt(finalSigmas[i]);
            fout.flush();
            fout << " " << sqrt(covariance[i+numParams*i]) << "\n";
        }
        fout.close();
        
        if (p.getb("covarianceOut")) {
            std::cout << "      Outputting parameter covariance..." << std::endl;
            std::string covfile = filename(p.gets("covBase"), p.geti("digits"), file, p.gets("ext"));
            fout.open(covfile.c_str(), std::ios::out);
            fout.precision(15);
            fout.width(15);
            fout << "";
            for (int i = 0; i < numParams; ++i) {
                if (i > 0) fout.width(19);
                else fout.width(20);
                fout << p.gets("paramNames", i);
            }
            fout << "\n";
            for (int i = 0; i < numParams; ++i) {
                fout.width(15);
                fout << p.gets("paramNames", i);
                fout.width(i*20);
                fout << "";
                for (int j = i; j < numParams; ++j) {
                    fout.width(20);
                    fout << covariance[j+numParams*i];
                }
                fout << "\n";
            }
            fout.close();
        }
        
        if (p.getb("realsOut")) {
            std::cout << "      Outputting all realizations..." << std::endl;
            std::string realsfile = filename(p.gets("realsBase"), p.geti("digits"), file, p.gets("ext"));
            fout.open(realsfile.c_str(), std::ios::out);
            fout.precision(15);
            for (int tid = 0; tid < numThreads; ++tid) {
                for (int draw = 0; draw < draws[tid]-1; ++draw) {
                    for (int param = 0; param < numParams; ++param) {
                        fout << reals[tid][draw][param];
                        if (param < numParams-1) fout << " ";
                    }
                    fout << "\n";
                }
            }
            fout.close();
        }
        
    }
    
    return 0;
}
