#ifndef _MCMC_H_
#define _MCMC_H_

#include <vector>

template <typename T> class mcmc{
    int N, N_p;
    double var;
    std::vector<std::vector<std::vector<double>>> reals;
    std::vector<std::vector<double>> averages;
    std::vector<std::vector<double>> stdevs;
    std::vector<T> x_vals;
    std::vector<double> data_vals, Psi, mle_params;
    
    double chisq(std::vector<double> &mod_vals);
    
    double likelihood(double chisq, double detPsi);
    
    void calc_model(std::vector<double> &mod_vals, void *params = NULL);
    
    void burn_in(std::vector<double> &burn_params, void *params = NULL);
    
    double variance_calc(std::vector<double> &mod_params, void *params = NULL);
    
    // Since this is a virtual function, the library can be compiled, but code will not compile until
    // the function implementation is written in the code using the library.
    virtual double model(T &x, std::vector<double> &mod_params, void *params = NULL);
    
    public:
        mcmc();
        
        mcmc(std::vector<T> &xvals, std::vector<double> &data, std::vector<double> &Psi, int numParams);
        
        void run_chains(std::vector<double> &start_params, int draws, int chains, double tolerance, 
                        void *params = NULL);
        
        void write_reals();
        
        void get_mle_params(std::vector<double> &params);
        
        // HOW?!?!?!?
        void marginalize();
        
        void set_num_params(int N_p);
        
        void add_data(T x, double val);
        
        void add_inverse_covar_element(int i, int j, double val);
};
