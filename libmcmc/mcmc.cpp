#include <vector>
#include <random>
#include <cmath>
#include <omp.h>

template <typename T> double mcmc<T>::chisq(std::vector<double> &mod_vals) {
    double result = 0.0;
    for (int i = 0; i < mcmc<T>::N; ++i) {
        for (int j = 0; j < mcmc<T>::N; ++j) {
            result += (mcmc<T>::data_vals[i] - mod_vals[i])*mcmc<T>::Psi[j + mcmc<T>::N*i]*
                      (mcmc<T>::data_vals[j] - mod_vals[j]);
        }
    }
    return result;
}

template <typename T> double mcmc<T>::likelihood(double chisq) {
    double likelihood = exp(-0.5*chisq);
    return likelihood;
}

template <typename T> void mcmc<T>::calc_model(std::vector<double> &mod_vals, void *params) {
    for (int i = 0; i < mcmc<T>::N; ++i) {
        mod_vals[i] = mcmc<T>::model(mcmc<T>::x_vals[i], params);
    }
}

template <typename T> void mcmc<T>::burn_in(std::vector<double> &burn_params) {
    
    
    std::vector<double> mod_vals(N);
    mcmc<T>::calc_model(mod_vals, params);

template <typename T> void mcmc<T>::variance_calc(std::vector<double> &start_params, void *params) {
    std::vector<double> mod_vals(N);
    mcmc<T>::calc_model(mod_vals, params);
    double chisq = mcmc<T>::chisq(mod_vals);
    double L = mcmc<T>::likelihood(chisq);
}

template <typename T> mcmc<T>::mcmc(std::vector<double> &params, std::vector<T> &xvals, 
                                      std::vector<double> &data) {
    mcmc<T>::N = data.size();
    for (int i = 0; i < mcmc<T>::N; ++i) {
        mcmc<T>::x_vals.push_back(xvals[i]);
        mcmc<T>::data_vals.push_back(data[i]);
    }
    
    int numParams = params.size();
    for (int i = 0; i < numParmas; ++i)
        mcmc<T>::mod_params.push_back(params[i]);
    
    mcmc<T>::calc_initial_likelihood();
}

template <typename T> mcmc<T>::run_chains(std::vector<double> &start_params, int draws, int chains, 
                                          double tolerance, void *params = NULL) {
    mcmc<T>::variance_calc(start_params, params);
    mcmc<T>::burn_in(start_params, params);
    bool converged = false;
    omp_set_num_threads(chains);
    
    while (!converged) {
        #pragma omp parallel
        {
            std::random_device seeder;
            std::mt19937_64 gen(seeder());
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            int tid = omp_get_thread_num();
        }
}

template class mcmc<double>;
template class mcmc<std::vector<double>>;
