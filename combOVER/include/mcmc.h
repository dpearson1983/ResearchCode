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
#include <limits>
#include <cuda.h>
#include <vector_types.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include "hide_harppi.h"
#include "powerspec.h"
#include "bispec.h"

class mcmc{
    size_t num_data, num_params, num_draws, num_burn, num_write, num_mocks;
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

// Takes an input covariance matrix stored in a square format in cov_file and calculates the unbiased
// estimate of the inverse covariance matrix needed for the likelihood calculation. This is then stored
// in the Psi data member of the mcmc object.
void mcmc::set_Psi(std::string cov_file) {
    if (check_file_exists(cov_file)) {
        gsl_matrix *cov = gsl_matrix_alloc(mcmc::num_data, mcmc::num_data);
        gsl_matrix *psi = gsl_matrix_alloc(mcmc::num_data, mcmc::num_data);
        gsl_permtuation *perm = gsl_permutation_alloc(mcmc::num_data);
        std::ifstream fin(cov_file);
        for (size_t i = 0; i < mcmc::num_data; ++i) {
            for (size_t j = 0; j < mcmc::num_data; ++j) {
                if (!fin.eof()) {
                    double val;
                    fin >> val;
                    gsl_matrix_set(cov, i, j, val);
                } else {
                    std::stringstream message;
                    message << "Unexpected end of file: " << cov_file << std::endl;
                    throw std::runtime_error(message.str());
                }
            }
        }
        fin.close();
        
        int s;
        gsl_linalg_LU_decomp(cov, perm, &s);
        gsl_linalg_LU_invert(cov, perm, psi);
        
        double D = double(mcmc::num_data + 1.0)/(double(mcmc::num_mocks - 1.0));
        
        for (size_t i = 0; i < mcmc::num_data; ++i) {
            std::vector<double> row;
            row.reserve(mcmc::num_data);
            for (size_t j = 0; j < mcmc::num_data; ++j) {
                row.push_back((1.0 - D)*gsl_matrix_get(psi, i, j));
            }
            mcmc::Psi.push_back(row);
        }
        
        gsl_matrix_free(cov);
        gsl_matrix_free(psi);
        gsl_permutation_free(perm);
    }
}

// Call the calculate functions of the powerspec and bispec objects and then copy the data
void mcmc::model_calc(std::vector<double> &pars, float3 *ks, double *Bk) {
    mcmc::Pk_mod.calculate(pars);
    mcmc::Bk_mod.calculate(pars, ks, Bk);
    
    for (size_t i = 0; i < mcmc::Pk_mod.num_vals; ++i) {
        mcmc::model[i] = Pk_mod.get(i);
    }
    
    for (size_t i = 0; i < mcmc::Bk_mod.num_vals; ++i) {
        mcmc::model[i + mcmc::Pk_mod.num_vals] = Bk_mod.get(i);
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

// Calculates the chi^2 value of the current model using the inverse covariance matrix. If you have only
// Gaussian variances, then setup a covariance matrix with all off diagonal elements equal to zero, and 
// diagonal elements equal to the variance of that data point.
double mcmc::calc_chi_squared() {
    double chisq = 0.0;
    for (size_t i = 0; i < mcmc::num_data; ++i) {
        for (size_t j = i; j < mcmc::num_data; ++j) {
            chisq += (mcmc::data[i] - mcmc::model[i])*Psi[i][j]*(mcmc::data[j] - mcmc::model[j]);
        }
    }
    return chisq;
}


// Performs one MCMC trial returning true is the proposed parameters are accepted, false if not. This is
// separated out to be used by the burn_in, tune_vars and run_chain functions greatly simplifying their
// implementations.
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

// Writes the current position of the MCMC chain to the screen. You can control the number of parameters
// to write in order to prevent line breaks. It is recommended that you structure your parameters to have
// the ones of interest at the beginning of the vector and any nuisance parameters at the end. All 
// parameters are written to the realization file.
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

// Performs a number of MCMC trials with the aim of moving the chain to a higher likelihood region of
// parameter space.
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

// Adjusts the user input search range around the parameters to hopefully optimize the acceptance ratio.
// This should ensure that the possible step sizes are large enough to avoid getting stuck in local minima,
// but not so large that the chain barely ever moves.
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

// Default constructor ensures that the random number generator is seeded and initialized.
mcmc::mcmc(): gen(seeder()) {
}

// Constructor that automatically calls the initialize function and also ensures that the random number
// generator is seeded and initialized.
mcmc::mcmc(mcmc_parameters p): gen(seeder()) {
    mcmc::initialize(p);
}

// Handles the initialization of the mcmc object so that the chain can be run.
void mcmc::initialize(mcmc_parameters p, float3 *ks, double *Bk) {
    // Initialize the powerspec and bispec objects
    mcmc::Pk_mod.initialize(p.pk_data_file, p.in_bao_file, p.in_nw_file);
    mcmc::Bk_mod.initialize(p.bk_data_file, p.in_nonlin_file);
    
    // Initialize some individual variables
    mcmc::num_data = mcmc::Pk_mod.num_vals + mcmc::Bk_mod.num_vals;
    mcmc::num_draws = p.num_draws;
    mcmc::num_mocks = p.num_mocks;
    mcmc::num_params = p.num_params;
    mcmc::num_burn = p.num_burn;
    mcmc::num_write = p.num_write;
    mcmc::new_chain = p.new_chain;
    mcmc::reals_file = p.reals_file;
    mcmc::variances_file = p.variances_file;
    
    // Copy the data to the mcmc object
    for (size_t i = 0; i < mcmc::Pk_mod.num_vals; ++i) {
        mcmc::data.push_back(mcmc::Pk_mod.get(i));
    }
    for (size_t i = 0; i < mcmc::Bk_mod.num_vals; ++i) {
        mcmc::data.push_back(mcmc::Bk_mod.get(i));
    }
    
    // Initialize all the stuff associated with the parameters
    for (size_t i = 0; i < mcmc::num_params; ++i) {
        mcmc::theta_0.push_back(p.start_params[i]);
        mcmc::min_pars.push_back(p.mins[i]);
        mcmc::max_pars.push_back(p.maxs[i]);
        mcmc::limit_params.push_back(p.limit_pars[i]);
        mcmc::theta_i.push_back(0.0);
        mcmc::param_vars.push_back(p.par_vars[i]);
    }
    
    // Read in the input covariance matrix and invert it
    mcmc::set_Psi(p.cov_file);
    
    // Initialize the device pointers needed for the bispectrum calculation
    gpuErrchk(cudaMemcpy(ks, mcmc::Bk_mod.ks.data(), mcmc::Bk_mod.num_vals*sizeof(float3), 
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(Bk, mcmc::Bk_mod.B.data(), mcmc::Bk_mod.num_vals*sizeof(double),
                         cudaMemcpyHostToDevice));
    
    // Calculate the initial model for the starting parameter values
    mcmc::Pk_mod.calculate(mcmc::theta_0);
    mcmc::Bk_mod.calculate(mcmc::theta_0, ks, Bk);
    
    // Copy that model back to the mcmc object
    for (size_t i = 0; i < mcmc::Pk_mod.num_vals; ++i) {
        mcmc::model.push_back(Pk_mod.get(i));
    }
    for (size_t i = 0; i < mcmc::Bk_mod.num_vals; ++i) {
        mcmc::model.push_back(Bk_mod.get(i));
    }
    
    // Get the chi^2 of the initial model
    mcmc::chisq_0 = mcmc::calc_chi_squared();
}

void mcmc::run_chain(float3 *ks, double *Bk) {
    size_t real_num = 0;
    
    // Check if this run it to resume a previous chain, if so read in the step sizes and the last accepted
    // realization.
    if (!mcmc::new_chain) {
        if (check_file_exists(mcmc::variances_file)) {
            std::ifstream fin(mcmc::variances_file);
            for (size_t i = 0; i < mcmc::num_params; ++i)
                fin >> mcmc::param_vars[i];
            fin.close();
        }
        
        if (check_file_exists(mcmc::reals_file)) {
            std::ifstream fin(mcmc::reals_file);
            while (!fin.eof()) {
                for (size_t i = 0; i < mcmc::num_params; ++i) {
                    fin >> mcmc::theta_0[i];
                }
                fin >> mcmc::chisq_0;
                ++real_num;
            }
        }
    }
    
    std::string error_msg;
    
    if (mcmc::new_chain && check_file_exists(mcmc::reals_file, error_msg)) {
        std::stringstream error;
        error << "The new_chain option was selected, but the realizations file already exists.\n";
        error << "Please change the file name or the new_chain option in the parameter file and re-run."
        error << std::endl;
        throw std::runtime_error(error.str());
    }
    
    std::ofstream fout(mcmc::reals_file, std::ios::app);
    fout.precision(std::numeric_limits<double>::digits10);
    for (size_t i = 0; i < mcmc::num_draws; ++i) {
        bool move = mcmc::trial(ks, Bk);
        ++real_num;
        for (size_t j = 0; j < mcmc::num_params; ++j)
            fout << mcmc::theta_0[i] << " ";
        fout << pow(mcmc::theta_0[3]*mcmc::theta_0[4]*mcmc::theta_0[4], 1.0/3.0) << " ";
        fout << mcmc::chisq_0 << "\n";
        if (move) {
            std::cout << "\r";
            std::cout.width(10);
            std::cout << real_num;
            mcmc::write_theta_screen();
        }
    }
    std::cout << std::endl;
    fout.close();
}
