#ifndef _BKMCMC_H_
#define _BKMCMC_H_

#include <vector>
#include <string>
#include <gsl/gsl_spline.h>
#include "tpods.h"
#include "harppi.h"

class bkmcmc{
    int num_data, num_pars, num_draws, num_write, num_burn, num_old_reals;
    std::vector<double> data, Bk; // These should have size of num_data
    std::vector<std::vector<double>> Psi; // This should have size num_data^2
    std::vector<double> theta_0, theta_i, param_vars, min, max; // These should all have size of num_pars
    std::vector<vec3<double>> k; // This should have size of num_data
    std::vector<bool> limit_pars; // This should have size of num_pars
    std::string reals_file, vars_file;
    bool new_chain, full_covar;
    gsl_spline *Pk_bao, *Pk_nw;
    gsl_interp_accel *acc_bao, *acc_nw;
    double chisq_0, chisq_i;
    
    // Calculates the power spectrum model needed for the bispectrum model
    double get_model_power(std::vector<double> &pars, double k_i);
    
    // Calculates the model bispectra (bao and nw) for the input parameters, pars.
    void model_calc(std::vector<double> &pars); // done
    
    // Sets the values of theta_i.
    void get_param_real(); // done
    
    // Calculates the chi^2 for the current proposal, theta_i
    double calc_chi_squared();
    
    // Performs one MCMC trial. Returns true if proposal accepted, false otherwise
    bool trial();
    
    // Writes the current accepted parameters to the screen
    void write_theta_screen(); // done
    
    // Burns the requested number of parameter realizations to move to a higher likelihood region
    void burn_in(int num_burn); // done
    
    // Changes the initial guesses for the search range around parameters until acceptance = 0.234
    void tune_vars(); // done
    
    // Displays information to the screen to check that the vectors are all the correct size
    void check_init(); // done
    
    void initialize_gsl_spline(std::string file, bool bao);
    
    public:
        // Initializes most of the data members and gets an initial chisq_0
        bkmcmc(parameters &p);
        
        ~bkmcmc();
        
        // Runs the MCMC chain for num_draws realizations, writing to reals_file
        void run_chain();
        
};

#endif
