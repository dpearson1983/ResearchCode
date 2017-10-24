/* mcmc.h
 * David W. Pearson
 * October 24, 2017
 * 
 * This header file contains all the functionality to run the Markov Chain Monte Carlo model fitting
 * using the combined galaxy power spectrum monopole and galaxy bispectrum monopole. The total model
 * contains the power spectrum values first and then the bispectrum values, and a full covariance
 * matrix between the power spectrum and bispectrum is needed. An object of this class will initialize
 * an object of the powerspec class and one of the bispec class. Functionality in those classes will 
 * handle the model calculation.
 * 
 * The purpose of this code is to perform the MCMC model fitting by:
 *  1. Initializing the model objects
 *  2. Calculating the initial model
 *  3. Burning a specified number of realizations to move the chain to a high likelihood region
 *  4. Tuning the acceptance ratio for the MCMC trials
 *  5. Running the chain and writing the accepted realizations to a file for further processing.
 * 
 * NOTE: After each trial, the data stored in the powerspec and bispec objects is overwritten. As part of
 * the trial, the model values are copied back to the mcmc object. These are also overwritten on the next
 * trial.
 * 
 */

#ifndef _MCMC_H_
#define _MCMC_H_

#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <cuda.h>
#include <vector_types.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include "hide_harppi.h"
#include "powerspec.h"
#include "bispec.h"

class mcmc{
    size_t num_data, num_params, num_draws, num_burn, num_write;
    std::vector<double> data, model; // Vectors of size num_data to store the data and model
    std::vector<double> theta_0, theta_i, min_pars, max_pars, param_vars; // Vectors of size num_params
    std::vector<std::vector<double>> Psi; // 2D vector of size num_data*num_data
    std::vector<bool> limit_params; // Vector of size num_params
    std::string variances_file, reals_file;
    bool new_chain;
    double chisq_0, chisq_i;
    powerspec Pk_mod;
    bispec Bk_mod;
    std::random_device seeder;
    std::mt19937_64 gen;
    
    void set_Psi(std::string cov_file);
    
    void model_calc(std::vector<double> &pars, float3 *ks, double *Bk);
    
    void get_param_real();
    
    double calc_chi_squared();
    
    bool trial(float3 *ks, double *Bk);
    
    void write_theta_screen();
    
    void burn_in(int num_burn, float3 *ks, double *Bk);
    
    void tune_vars(float3 *ks, double *Bk);
    
    public:
        mcmc();
        
        mcmc(mcmc_parameters p, float *ks, double *Bk);
        
        void initialize(mcmc_paramters p, float *ks, double *Bk);
        
        void run_chain(float *ks, double *Bk);
        
};

void mcmc::set_Psi(std::string cov_file) {
    

// Call the calculate functions of the powerspec and bispec objects and then copy the data
void mcmc::model_calc(std::vector<double> &pars, float3 *ks, double *Bk) {
    mcmc::Pk_mod.calculate(pars);
    mcmc::Bk_mod.calculate(pars, ks, Bk);
    
    for (size_t i = 0; i < mcmc::Pk_mod.num_vals; ++i) {
        mcmc::model[i] = Pk_mod.get(i);
    }
    
    for (size_t i = 0; i < mcmc::Bk_mod.num_vals; ++i) {
        mcmc::mdoel[i + mcmc::Pk_mod.num_vals] = Bk_mod.get(i);
    }
}

// Get a random parameter realization from with a small volume of parameter space. 
void mcmc::get_param_real() {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < mcmc::num_params; ++i) {
        if (mcmc::limit_params[i]) {
            if (mcmc::theta_0[i] + mcmc::param_vars[i] > mcmc::max_pars[i]) {
                double center = mcmc::max_pars[i] - mcmc::param_vars[i];
                mcmc::theta_i[i] = center + dist(mcmc::gen)*mcmc::param_vars[i];
            } else if (mcmc::theta_0[i] - mcmc::param_vars[i] < mcmc::min_pars[i]) {
                double center = mcmc::min_pars[i] + mcmc::param_vars[i];
                mcmc::theta_i[i] = center + dist(mcmc::gen)*mcmc::param_vars[i];
            } else {
                mcmc::theta_i[i] = mcmc::theta_0[i] + dist(mcmc::gen)*mcmc::param_vars[i];
            }
        } else {
            mcmc::theta_i[i] = mcmc::theta_0[i] + dist(mcmc::gen)*mcmc::param_vars[i];
        }
    }
}

double mcmc::calc_chi_squared() {
    double chisq = 0.0;
    for (size_t i = 0; i < mcmc::num_data; ++i) {
        for (size_t j = i; j < mcmc::num_data; ++j) {
            chisq += (mcmc::data[i] - mcmc::model[i])*Psi[i][j]*(mcmc::data[j] - mcmc::model[j]);
        }
    }
    return chisq;
}

bool mcmc::trial(float3 *ks, double *Bk) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    mcmc::get_param_real();
    mcmc::model_calc(mcmc::theta_i, ks, Bk);
    mcmc::chisq_i = mcmc::calc_chi_squared();
    
    double L = exp(0.5*(mcmc::chisq_0 - mcmc::chisq_i));
    double R = dist(mcmc::gen);
    
    if (L > R) {
        for (size_t i = 0; i < mcmc::num_params; ++i)
            mcmc::theta_0[i] = mcmc::theta_i[i];
        mcmc::chisq_0 = mcmc::chisq_i;
        return true;
    } else {
        return false;
    }
}

void mcmc::write_theta_screen() {
    std::cout.precision(6);
    for (size_t i = 0; i < mcmc::num_write; ++i) {
        std::cout.width(10);
        std::cout << mcmc::theta_0[i];
    }
    std::cout.width(10);
    std::cout << pow(mcmc::theta_0[3]*mcmc::theta_0[4]*mcmc::theta_0[4], 1.0/3.0);
    std::cout.width(10);
    std::cout << mcmc::chisq_0;
    std::cout.flush();
}

void mcmc::burn_in(float3 *ks, double *Bk) {
    std::cout << "Burning the first " << mcmc::num_burn << " trials..." << std::endl;
    for (size_t i = 0; i < mcmc::num_burn; ++i) {
        bool move = mcmc::trial(ks, Bk);
        if (move) {
            std::cout << "\r";
            std::cout.width(10);
            std::cout << i;
            mcmc::write_theta_screen();
        }
    }
    std::cout << std::endl;
}

void mcmc::tune_vars(float3 *ks, double *Bk) {
    std::cout << "Tuning acceptance ratio..." << std::endl;
    double acceptance = 0.0;
    while (acceptance <= 0.233 || acceptance >= 0.235) {
        int accept = 0;
        for (size_t i = 0; i < 10000; ++i) {
            bool move = mcmc::trial(ks, Bk);
            if (move) {
                std::cout << "\r";
                mcmc::write_theta_screen();
                accept++;
            }
        }
        std::cout << std::endl;
        acceptance = double(accept)/10000.0;
        
        if (acceptance <= 0.233) {
            for (size_t i = 0; i < mcmc::num_params; ++i)
                mcmc::param_vars[i] *= 0.99;
        }
        if (acceptance >= 0.235) {
            for (size_t i = 0; i < mcmc::num_params; ++i)
                mcmc::param_vars[i] *= 1.01;
        }
        std::cout << "acceptance = " << acceptance << std::endl;
    }
    std::ofstream fout(mcmc::variances_file);
    fout.precision(15);
    for (size_t i = 0; i < mcmc::num_params; ++i)
        fout << mcmc::param_vars[i] << " ";
    fout << "\n";
    fout.close();
}

mcmc::mcmc(): gen(seeder()) {
}

mcmc::mcmc(mcmc_parameters p): gen(seeder()) {
    mcmc::initialize(p);
}

void mcmc::initialize(mcmc_parameters p) {
    mcmc::Pk_mod.initialize(p.pk_data_file, p.in_bao_file, p.in_nw_file);
    mcmc::Bk_mod.initialize(p.bk_data_file, p.in_nonlin_file);
    mcmc::num_data = mcmc::Pk_mod.num_vals + mcmc::Bk_mod.num_vals;
    
    for (size_t i = 0; i < mcmc::Pk_mod.num_vals; ++i) {
        mcmc::data.push_back(mcmc::Pk_mod.get(i));
        mcmc::model.push_back(0.0);
    }
    for (size_t i = 0l i < mcmc::Bk_mod.num_vals; ++i) {
        mcmc::data.push_back(mcmc::Bk_mod.get(i));
        mcmc::model.push_back(0.0);
    }
    
    
