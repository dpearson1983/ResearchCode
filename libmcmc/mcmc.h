#ifndef _MCMC_H_
#define _MCMC_H_

#include <vector>
#include <string>

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
    void get_param_real(std::vector<double> &mod_params, std::vector<double> &cur_pars, std::vector<double> &rand);
    
    // Calculate the chi^2 of the model
    double chisq(std::vector<double> &mod_vals);
    
    // Calculate the likelihood assuming a multivariate Gaussian
    double likelihood(double chisq);
    
    // Calculate the model values for each (set of) independent variable(s).
    void calc_model(std::vector<double> &mod_vals, std::vector<double> &mod_params, void *params = NULL);
    
    // Move the chain to a more optimal location
    void burn_in(std::vector<double> &burn_params, int numBurn, int chains, void *params = NULL);
    
    // Calculate the optimal variance. This will be a fixed percentage of the initial guesses for parameter values
    void variance_calc(std::vector<double> &start_params, void *params = NULL);
    
    // Since this is a virtual function, the library can be compiled, but code will not compile until
    // the function implementation is written in the code using the library.
    virtual double model(T &x, std::vector<double> &mod_params, void *params = NULL);
    
    public:
        // Initialize the mcmc object with independent variables, data, inverse covariance, and number of parameters
        mcmc(std::vector<T> &xvals, std::vector<double> &data, std::vector<double> &Psi, int numParams, int chains);
        
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
        
};

#endif
