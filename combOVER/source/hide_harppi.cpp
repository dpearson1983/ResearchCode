#include <iostream>
#include <string>
#include <vector>
#include <harppi.h>
#include "../include/hide_harppi.h"

mcmc_parameters::mcmc_parameters(char *file) {
    parameters p(file);
    p.print();
    
    // Setup integer values
    mcmc_parameters::num_data = p.geti("num_data");
    mcmc_parameters::num_params = p.geti("num_params");
    mcmc_parameters::num_draws = p.geti("num_draws");
    mcmc_parameters::num_burn = p.geti("num_burn");
    mcmc_parameters::num_write = p.geti("num_write");
    mcmc_parameters::num_mocks = p.geti("num_mocks");
    
    // Setup file names
    mcmc_parameters::reals_file = p.gets("reals_file");
    mcmc_parameters::variances_file = p.gets("variances_file");
    mcmc_parameters::cov_file = p.gets("cov_file");
    mcmc_parameters::pk_data_file = p.gets("pk_data_file");
    mcmc_parameters::bk_data_file = p.gets("bk_data_file");
    mcmc_parameters::in_bao_file = p.gets("in_bao_file");
    mcmc_parameters::in_nw_file = p.gets("in_nw_file");
    mcmc_parameters::in_nonlin_file = p.gets("in_nonlin_file");
    
    // Set up booleans
    mcmc_parameters::new_chain = p.getb("new_chain");
    
    // Set up vectors
    for (int i = 0; i < mcmc_parameters::num_params; ++i) {
        mcmc_parameters::start_params.push_back(p.getd("start_params", i));
        mcmc_parameters::par_vars.push_back(p.getd("vars",i));
        mcmc_parameters::mins.push_back(p.getd("min_params", i));
        mcmc_parameters::maxs.push_back(p.getd("max_params", i));
        mcmc_parameters::limit_pars.push_back(p.getb("limit_params", i));
    }
}
