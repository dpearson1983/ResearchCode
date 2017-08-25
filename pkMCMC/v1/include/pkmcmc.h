#ifndef _PKMCMC_H_
#define _PKMCMC_H_

#include <vector>
#include <string>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_matrix.h>

class pkmcmc{
    int num_data, num_pars, workspace_size;
    std::vector<double> data, model, k;
    std::vector<double> theta_0, theta_i, param_vars, min, max;
    std::vector<std::vector<double>> cov, Psi;
    std::vector<bool> limit_pars;
    gsl_spline *Pk;
    gsl_interp_accel *acc;
    gsl_integration_workspace *w;
    double chisq_0, chisq_i, abs_err, rel_err;
    
    // The function to be integrated by GSL
    double model_func(std::vector<double> &pars, int j); //done
    
    // Performs the integral needed for the model value at each data point
    void model_calc(std::vector<double> &pars); //done
    
    // Randomly selects a new parameter realization for a trial
    void get_param_real(); //done
    
    // Compute the chi^2 value of the current model with the data
    double calc_chi_squared(); //done
    
    // Perform one Metropolis-Hastings trial
    bool trial(); //done
    
    // Write the current accepted parameter realization to the screen
    void write_theta_screen(); //done
    
    // Perform a number of trials to move from the initial parameter guess to a higher likelihood region
    void burn_in(int num_burn); //done
    
    // Tune the acceptance ratio to be about 0.234
    void tune_vars(); //done
    
    public:
        // Initialization function
        pkmcmc(std::string data_file, std::string cov_file, std::string pk_file, std::vector<double> &pars, 
               std::vector<double> &vars, int int_workspace, double err_abs, double err_rel); //done
        
        // Check that the vectors have the expected sizes
        void check_init(); //done
        
        // Set which parameters have priors and what those priors are.
        void set_param_limits(std::vector<bool> &lim_pars, std::vector<double> &min_in, 
                              std::vector<double> &max_in); //done
        
        // Run a number of Metropolis-Hastings trials to form the MCMC chain
        void run_chain(int num_draws, std::string reals_file, bool new_chain);
        
        // Free the memory with various GSL class members
        void clean_up_gsl();
        
};

#endif
