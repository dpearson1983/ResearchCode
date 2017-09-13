#include <iostream>
#include <string>
#include <vector>
#include <harppi.h>
#include "../include/hide_harppi.h"

mcmc_parameters::mcmc_parameters(char *file) {
    parameters p(file);
    p.print();
    
    mcmc_parameters::num_params = p.geti("num_params");
    mcmc_parameters::N = p.geti("N");
    mcmc_parameters::num_threads = p.geti("num_threads");
    mcmc_parameters::num_chains = p.geti("num_chains");
    mcmc_parameters::num_burn = p.geti("num_burn");
    mcmc_parameters::num_draws = p.geti("num_draws");
    mcmc_parameters::nbar = p.getd("nbar");
    mcmc_parameters::mcmc_tolerance = p.getd("mcmc_tolerance");
    mcmc_parameters::input_bao_power = p.gets("input_bao_power");
    mcmc_parameters::input_nw_power = p.gets("input_nw_power");
    mcmc_parameters::data_file = p.gets("data_file");
    mcmc_parameters::reals_file = p.gets("reals_file");
    mcmc_parameters::summary_file = p.gets("summary_file");
    mcmc_parameters::full_covar = p.getb("full_covar");
    mcmc_parameters::new_chain = p.getb("new_chain");
    mcmc_parameters::num_data = p.geti("num_data");
    mcmc_parameters::num_samps = p.geti("num_samps");
    mcmc_parameters::cov_file = p.gets("cov_file");
    
    for (int i = 0; i < mcmc_parameters::num_params; ++i) {
        mcmc_parameters::start_params.push_back(p.getd("start_params", i));
        mcmc_parameters::limit_params.push_back(p.getb("limit_params", i));
        mcmc_parameters::var_i.push_back(p.getd("vars",i));
        mcmc_parameters::min.push_back(p.getd("min_params", i));
        mcmc_parameters::max.push_back(p.getd("max_params", i));
    }
}
