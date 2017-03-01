#ifndef _MCMC_H_
#define _MCMC_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <omp.h>

template <typename T> class mcmc{
    int N, N_p, chains;
    double var = 0.1;
    std::vector<std::vector<std::vector<double>>> reals; // Stores all the accepted realizations
    std::vector<std::vector<double>> averages; // Stores the average of the accepted realizations
    std::vector<std::vector<double>> stdevs; // Stores the standard deviations of the accepted realizations
    std::vector<std::vector<double>> current_params; // Stores current values of parameters for each thread
    std::vector<std::vector<double>> param_covar; // Stores the covariance matrix of the parameters
    std::vector<std::vector<double>> limits;
    std::vector<int> total_draws;
    std::vector<double> variances;
    std::vector<bool> limit_params;
    std::vector<T> x_vals; // Stores the independent variable values
    std::vector<double> data_vals, Psi, mle_params;
    
    // Get a trial parameter vector
    void get_param_real(std::vector<double> &mod_params, std::vector<double> cur_pars, std::vector<double> rand);
    
    // Calculate the chi^2 of the model
    double chisq(std::vector<double> &mod_vals);
    
    // Calculate the likelihood assuming a multivariate Gaussian
    double likelihood(double chisq);
    
    // Calculate the model values for each (set of) independent variable(s).
    std::vector<double> calc_model(std::vector<double> mod_params, void *params = NULL);
    
    // Move the chain to a more optimal location
    void burn_in(std::vector<double> burn_params, int numBurn, int chains, void *params = NULL);
    
    // Calculate the optimal variance. This will be a fixed percentage of the initial guesses for parameter values
    void variance_calc(std::vector<double> start_params, void *params = NULL);
    
    // Since this is a virtual function, the library can be compiled, but code will not compile until
    // the function implementation is written in the code using the library.
    double model(T x, std::vector<double> mod_params, void *params = NULL);
    
//     virtual double model(std::vector<double> &x, std::vector<double> &mod_params, void *params = NULL);
    
    public:
        // Initialize the mcmc object with independent variables, data, inverse covariance, and number of parameters
        mcmc(std::vector<T> xvals, std::vector<double> data, std::vector<double> Psi, int numParams, int chains);
        
        // Run the MCMC chains. Calls variance_calc and burn_in, then does "draws" realizations in "chains"
        // threads, then checks is the chains have converged to within "tolerance"
        void run_chains(std::vector<double> &start_params, int draws, int numBurn, int chains, double tolerance, 
                        void *params = NULL);
        
        // Allows the user to specify which parameters have limits
        void set_limits(std::vector<bool> &lim_pars, std::vector<std::vector<double>> &lims);
        
        void write_reals_text(std::string file);
        
        void write_reals_binary(std::string file);
        
        void get_mle_params(std::vector<double> &pars);
        
        double get_chisq(std::vector<double> &pars, void *params = NULL);
        
        // HOW?!?!?!?
        void marginalize();
        
        void print();
        
};

template <typename T> void mcmc<T>::get_param_real(std::vector<double> &mod_params, 
                                                   std::vector<double> cur_pars, 
                                                   std::vector<double> rand) {
    for (int i = 0; i < mcmc<T>::N_p; ++i) {
        if (mcmc<T>::limit_params[i]) {
            if (cur_pars[i] - mcmc<T>::variances[i] >= mcmc<T>::limits[i][0] &&
                cur_pars[i] + mcmc<T>::var*cur_pars[i] <= mcmc<T>::limits[i][1]) {
                mod_params[i] = cur_pars[i] + rand[i]*mcmc<T>::variances[i];
            } else if (cur_pars[i] - mcmc<T>::variances[i] < mcmc<T>::limits[i][0]) {
                double center = mcmc<T>::limits[i][0] + variances[i];
                mod_params[i] = center + rand[i]*mcmc<T>::variances[i];
            } else if (cur_pars[i] + mcmc<T>::variances[i] > mcmc<T>::limits[i][1]) {
                double center = mcmc<T>::limits[i][1] - variances[i];
                mod_params[i] = center + rand[i]*mcmc<T>::variances[i];
            }
        } else {
            mod_params[i] = cur_pars[i] + rand[i]*mcmc<T>::variances[i];
        }
    }
}

template <typename T> double mcmc<T>::chisq(std::vector<double> &mod_vals) {
    double result = 0.0;
    for (int i = 0; i < mcmc<T>::N; ++i) {
        for (int j = i; j < mcmc<T>::N; ++j) {
            result += (mcmc<T>::data_vals[i] - mod_vals[i])*mcmc<T>::Psi[j + mcmc<T>::N*i]*
                      (mcmc<T>::data_vals[j] - mod_vals[j]);
        }
    }
    return result;
}

template <typename T> double mcmc<T>::likelihood(double chisq) {
    double likelihood = exp(-0.5*chisq);
    return likelihood;
}

template <typename T> std::vector<double> mcmc<T>::calc_model(std::vector<double> mod_params,
                                               void *params) {
    std::vector<double> mod_vals(mcmc<T>::N);
    for (int i = 0; i < mcmc<T>::N; ++i) {
        mod_vals[i] = mcmc<T>::model(mcmc<T>::x_vals[i], mod_params, params);
    }
    return mod_vals;
}

template <typename T> void mcmc<T>::burn_in(std::vector<double> burn_params, int numBurn, int chains, 
                                            void *params) {
    std::cout << "    Burning the first " << numBurn << " realizations for each thread..." << std::endl;
    
    omp_set_num_threads(chains);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        std::random_device seeder;
        std::mt19937_64 gen(seeder());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        
        std::vector<double> rands(mcmc<T>::N_p);
        
        for (int i = 0; i < mcmc<T>::N_p; ++i)
            rands[i] = dist(gen);
        mcmc<T>::get_param_real(current_params[tid], burn_params, rands);
        
        std::vector<double> mod_vals = mcmc<T>::calc_model(burn_params, params);
        double chisq_0 = mcmc<T>::chisq(mod_vals);
        double L_0 = likelihood(chisq_0);
        
        for (int i = 0; i < numBurn; ++i) {
            std::vector<double> pars(mcmc<T>::N_p);
            for (int m = 0; m < mcmc<T>::N_p; ++m) {
                rands[m] = dist(gen);
            }
            mcmc<T>::get_param_real(pars, current_params[tid], rands);
            std::vector<double> vals = mcmc<T>::calc_model(pars, params);
            double chisq_i = mcmc<T>::chisq(vals);
            double L_i = mcmc<T>::likelihood(chisq_i);
            double ratio = L_i/L_0;
            double test = (dist(gen) + 1.0)/2.0;
            
            if (ratio > test) {
                L_0 = L_i;
                for (int m = 0; m < mcmc<T>::N_p; ++m) 
                    current_params[tid][m] = pars[m];
            }
        }
    }
}

template <typename T> void mcmc<T>::variance_calc(std::vector<double> start_params, void *params) {
    std::cout << "    Tuning the variance..." << std::endl;
    
    std::cout << "        Setting up for the model..." << std::endl;
    std::vector<double> mod_vals = mcmc<T>::calc_model(start_params, params);
    std::cout << "        Calculating chi^2..." << std::endl;
    double chisq_0 = mcmc<T>::chisq(mod_vals);
    std::cout << "        chi^2 = " << chisq_0 << std::endl;
    std::cout << "        Calculating likelihood..." << std::endl;
    double L_0 = mcmc<T>::likelihood(chisq_0);
    std::cout << "        Likelihood = " << L_0 << std::endl;
    
    double acceptance = 1.0;
    
    std::cout << "        Setting up random number generator..." << std::endl;
    std::random_device seeder;
    std::mt19937_64 gen(seeder());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    std::vector<double> cur_par(mcmc<T>::N_p);
    std::vector<double> rands(mcmc<T>::N_p);
    
    std::cout << "        Setting the start parameters..." << std::endl;
    for (int i = 0; i < mcmc<T>::N_p; ++i) {
        cur_par[i] = start_params[i];
        mcmc<T>::variances[i] = start_params[i]*mcmc<T>::var;
    }
    
    std::cout << "        Burning some of the chain..." << std::endl;
    for (int i = 0; i < 1000; ++i) {
        std::vector<double> pars(mcmc<T>::N_p);
        for (int m = 0; m < mcmc<T>::N_p; ++m) {
            rands[m] = dist(gen);
        }
        mcmc<T>::get_param_real(pars, cur_par, rands);
        std::vector<double> vals =  mcmc<T>::calc_model(pars, params);
        double chisq_i = mcmc<T>::chisq(vals);
        double L_i = mcmc<T>::likelihood(chisq_i);
        double ratio = L_i/L_0;
        double test = (dist(gen) + 1.0)/2.0;
        
        if (ratio > test) {
            L_0 = L_i;
            for (int m = 0; m < mcmc<T>::N_p; ++m) 
                cur_par[m] = pars[m];
        }
    }
    std::cout << "        Likelihood = " << L_0 << std::endl;
    std::cout << "        Current parameter values: ";
    for (int i = 0; i < mcmc<T>::N_p; ++i)
        std::cout << cur_par[i] << " ";
    std::cout << std::endl;
    
    std::cout << "        Tuning..." << std::endl;
    while (acceptance >= 0.235 || acceptance <= 0.233) {
        int accept = 0;
        for (int i = 0; i < 1000; ++i) {
            std::vector<double> pars(mcmc<T>::N_p);
            for (int m = 0; m < mcmc<T>::N_p; ++m) {
                rands[m] = dist(gen);
            }
            mcmc<T>::get_param_real(pars, cur_par, rands);
            std::vector<double> vals = mcmc<T>::calc_model(pars, params);
            double chisq_i = mcmc<T>::chisq(vals);
            double L_i = mcmc<T>::likelihood(chisq_i);
            double ratio = L_i/L_0;
            double test = (dist(gen) + 1.0)/2.0;
            
            if (ratio > test) {
                ++accept;
                L_0 = L_i;
                for (int m = 0; m < mcmc<T>::N_p; ++m) 
                    cur_par[m] = pars[m];
            }
        }
        
        acceptance = double(accept)/1000.0;
        std::cout << "\r        acceptance ratio = " << acceptance;
        
        if (acceptance >= 0.235) {
            mcmc<T>::var *= 1.01;
            for (int i = 0; i < mcmc<T>::N_p; ++i)
                mcmc<T>::variances[i] = mcmc<T>::var*start_params[i];
        } if (acceptance <= 0.233) {
            mcmc<T>::var *= 0.99;
            for (int i = 0; i < mcmc<T>::N_p; ++i)
                mcmc<T>::variances[i] = mcmc<T>::var*start_params[i];
        }
    }
    std::cout << std::endl;
    
    for (int i = 0; i < mcmc<T>::N_p; ++i) {
        std::cout << mcmc<T>::variances[i] << std::endl;
    }
}

template <typename T> mcmc<T>::mcmc(std::vector<T> xvals, std::vector<double> data, 
                                    std::vector<double> Psi, int numParams, int chains) {
    mcmc<T>::N = data.size();
    mcmc<T>::N_p = numParams;
    for (int i = 0; i < mcmc<T>::N; ++i) {
        mcmc<T>::x_vals.push_back(xvals[i]);
        mcmc<T>::data_vals.push_back(data[i]);
        for (int j = 0; j < mcmc<T>::N; ++j) {
            mcmc<T>::Psi.push_back(Psi[j + mcmc<T>::N*i]);
        }
    }
    
    for (int tid = 0; tid < chains; ++tid) {
        std::vector<double> ptemp(numParams);
        mcmc<T>::current_params.push_back(ptemp);
        mcmc<T>::averages.push_back(ptemp);
        mcmc<T>::stdevs.push_back(ptemp);
    }
    
    for (int i = 0; i < numParams; ++i) {
        std::vector<double> row(numParams);
        for (int j = 0; j < numParams; ++j) {
            row[j] = 0.0;
        }
        mcmc<T>::param_covar.push_back(row);
        mcmc<T>::variances.push_back(0.0);
        mcmc<T>::limit_params.push_back(false);
    }
}

template <typename T> void mcmc<T>::run_chains(std::vector<double> &start_params, int draws, int numBurn, 
                                               int chains, double tolerance, void *params) {
    std::cout << "Running MCMC chains..." << std::endl;
    mcmc<T>::variance_calc(start_params, params);
    mcmc<T>::burn_in(start_params, numBurn, chains, params);
    bool converged = false;
    omp_set_num_threads(chains);
    mcmc<T>::chains = chains;
    
    mcmc<T>::reals.resize(chains);
    for (int i = 0; i < chains; ++i)
        mcmc<T>::total_draws.push_back(0);
    
    std::cout << "    Starting the runs..." << std::endl;
    while (!converged) {
        #pragma omp parallel
        {
            std::random_device seeder;
            std::mt19937_64 gen(seeder());
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            int tid = omp_get_thread_num();
            
            std::vector<double> rands(mcmc<T>::N_p);
            std::vector<double> real(mcmc<T>::N_p + 2);
            std::vector<double> mod_vals = mcmc<T>::calc_model(mcmc<T>::current_params[tid], params);
            double chisq_0 = mcmc<T>::chisq(mod_vals);
            double L_0 = mcmc<T>::likelihood(chisq_0);
            
            for (int i = 0; i < draws; ++i) {
                ++total_draws[tid];
                std::vector<double> pars(mcmc<T>::N_p);
                for (int par = 0; par < mcmc<T>::N_p; ++par)
                    rands[par] = dist(gen);
                mcmc<T>::get_param_real(pars, current_params[tid], rands);
                std::vector<double> vals = mcmc<T>::calc_model(pars, params);
                double chisq_i = mcmc<T>::chisq(vals);
                double L_i = mcmc<T>::likelihood(chisq_i);
                double ratio = L_i/L_0;
                double test = (dist(gen) + 1.0)/2.0;
                
                if (ratio > test) {
                    L_0 = L_i;
                    chisq_0 = chisq_i;
                    for (int par = 0; par < mcmc<T>::N_p; ++par) {
                        mcmc<T>::current_params[tid][par] = pars[par];
                        real[par] = pars[par];
                    }
                    real[mcmc<T>::N_p] = chisq_i;
                    real[mcmc<T>::N_p + 1] = L_i;
                } else {
                    for (int par = 0; par < mcmc<T>::N_p; ++par)
                        real[par] = current_params[tid][par];
                    real[mcmc<T>::N_p] = chisq_0;
                    real[mcmc<T>::N_p + 1] = L_0;
                }
                mcmc<T>::reals[tid].push_back(real);
                
                for (int par = 0; par < mcmc<T>::N_p; ++par) {
                    double avg = mcmc<T>::current_params[tid][par]/double(total_draws[tid]) + 
                                 ((double(total_draws[tid]) - 1.0)/double(total_draws[tid]))*
                                 mcmc<T>::averages[tid][par];
                    mcmc<T>::stdevs[tid][par] += (mcmc<T>::current_params[tid][par] - mcmc<T>::averages[tid][par])*
                                                 (mcmc<T>::current_params[tid][par] - avg);
                    mcmc<T>::averages[tid][par] = avg;
                }
            }
        }
        
        std::vector<double> avgavg(mcmc<T>::N_p);
        std::vector<double> varavg(mcmc<T>::N_p);
        for (int i = 0; i < chains; ++i) {
            for (int par = 0; par < mcmc<T>::N_p; ++par) {
                avgavg[par] += averages[i][par]/double(chains);
            }
        }
        
        for (int i = 0; i < chains; ++i) {
            for (int par = 0; par < mcmc<T>::N_p; ++par) {
                varavg[par] += ((averages[i][par] - avgavg[par])*(averages[i][par] - avgavg[par]))/(double(chains - 1.0));
            }
        }
        
        int paramconv = 0;
        for (int par = 0; par < mcmc<T>::N_p; ++par) {
            if (varavg[par] < tolerance) ++paramconv;
        }
        if (paramconv == mcmc<T>::N_p) converged = true;
    }
}

template <typename T> void mcmc<T>::set_limits(std::vector<bool> &lim_pars, std::vector<std::vector<double>> &lims) {
    for (int i = 0; i < mcmc<T>::N_p; ++i) {
        mcmc<T>::limit_params[i] = lim_pars[i];
        if (lim_pars[i]) {
            mcmc<T>::limits[i][0] = lims[i][0];
            mcmc<T>::limits[i][1] = lims[i][1];
        }
    }
}

template <typename T> void mcmc<T>::write_reals_text(std::string file) {
    std::ofstream fout;
    fout.open(file.c_str(), std::ios::out);
    fout.precision(15);
    for (int tid = 0; tid < mcmc<T>::chains; ++tid) {
        for (int i = 0; i < mcmc<T>::total_draws[tid]; ++i) {
            for (int j = 0; j < mcmc<T>::N_p + 2; ++j) {
                fout << mcmc<T>::reals[tid][i][j] << " ";
            }
            fout << "\n";
        }
    }
    fout.close();
}

template <typename T> void mcmc<T>::write_reals_binary(std::string file) {
    std::ofstream fout;
    fout.open(file.c_str(), std::ios::out|std::ios::binary);
    for (int tid = 0; tid < mcmc<T>::chains; ++tid) {
        for (int i = 0; i < mcmc<T>::total_draws[tid]; ++i) {
            fout.write((char *) &reals[tid][i], (mcmc<T>::N_p + 2)*sizeof(double));
        }
    }
    fout.close();
}

template <typename T> void mcmc<T>::get_mle_params(std::vector<double> &pars) {
    for (int i = 0; i < mcmc<T>::chains; ++i) {
        for (int j = 0; j < mcmc<T>::N_p; ++j) {
            pars[j] += averages[i][j]/double(mcmc<T>::chains);
        }
    }
}

template <typename T> double mcmc<T>::get_chisq(std::vector<double> &pars, void *params) {
    std::vector<double> mod_vals(mcmc<T>::N);
    mcmc<T>::calc_model(mod_vals, pars, params);
    return mcmc<T>::chisq(mod_vals);
}

template <typename T> void mcmc<T>::marginalize() {
    
}

template <typename T> void mcmc<T>::print() {
    for (int i = 0; i < mcmc<T>::N; ++i)
        std::cout << mcmc<T>::x_vals[i] << " " << mcmc<T>::data_vals[i] << std::endl;
}

#endif
