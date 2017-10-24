#ifndef _HIDE_HARPPI_H_
#define _HIDE_HARPPI_H_

#include <vector>
#include <string>

struct mcmc_parameters{
    size_t num_data, num_params, num_draws, num_burn, num_write, num_mocks;
    std::string in_bao_file, in_nw_file, in_nonlin_file;
    std::string reals_file, variances_file, cov_file;
    std::string pk_data_file, bk_data_file;
    bool new_chain;
    std::vector<double> par_vars, mins, maxs;
    std::vector<bool> limit_pars;
    
    mcmc_parameters(char *file);
};

#endif
