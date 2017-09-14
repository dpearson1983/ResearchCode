#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <harppi.h>
#include "include/pkmcmc.h"

int main(int argc, char *argv[]) {
    // Use HARPPI to parse parameter file passed via command line
    parameters p(argv[1]);
    p.print();
    
    // Set up some vectors to easily pass some of the parameters when initializing pkmcmc class object
    std::vector<double> pars(p.geti("num_pars"));
    std::vector<double> vars(p.geti("num_pars"));
    std::vector<double> min_in(p.geti("num_pars"));
    std::vector<double> max_in(p.geti("num_pars"));
    std::vector<bool> lim_pars(p.geti("num_pars"));
    
    // Move values to those vectors
    for (int i = 0; i < p.geti("num_pars"); ++i) {
        pars[i] = p.getd("pars", i);
        vars[i] = p.getd("vars", i);
        min_in[i] = p.getd("mins", i);
        max_in[i] = p.getd("maxs", i);
        lim_pars[i] = p.getb("limit_params", i);
    }
    
    // Initialize the pkmcmc object
    pkmcmc pkFit(p.gets("data_file"), p.gets("cov_file"), p.gets("pk_bao_file"), p.gets("pk_nw_file"), pars,
                 vars, p.getb("mock_avg"));
    
    pkFit.check_init();
    
    pkFit.set_param_limits(lim_pars, min_in, max_in);
    
    pkFit.run_chain(p.geti("num_draws"), p.gets("reals_file"), p.getb("new_chain"));
    
    pkFit.clean_up_gsl();
    
    return 0;
}
