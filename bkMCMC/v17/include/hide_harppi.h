#ifndef _HIDE_CLASS_H_
#define _HIDE_CLASS_H_

#include <vector>
#include <string>

struct mcmc_parameters{
    int num_params, N, num_threads, num_chains, num_burn, num_draws, num_data;
    double nbar, mcmc_tolerance, sigma8, k_nl;
    bool full_covar, new_chain;
    std::vector<double> start_params;
    std::string input_bao_power, input_nw_power, data_file, reals_file, param_covar_file, summary_file;
    std::string cov_file, input_linear_nw_power;
    std::vector<bool> limit_params;
    std::vector<double> min;
    std::vector<double> max;
    std::vector<double> var_i;
    
    mcmc_parameters(char *file);
};

#endif
