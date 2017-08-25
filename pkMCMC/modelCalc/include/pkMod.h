#ifndef _PKMOD_H_
#define _PKMOD_H_

#include <vector>
#include <gsl/gsl_spline.h>

class pk_model{
    int num_data, num_pars;
    std::vector<double> k, data, model, params;
    double k_min, k_max;
    gsl_spline *Pk_m;
    gsl_interp_accel *acc;
    
    // The function to be integrated by GSL
    double model_func(std::vector<double> &pars, int j); //done
    
    // Performs the integral needed for the model value at each data point
    void model_calc(std::vector<double> &pars); //done
    
    void initialize_spline(std::string pk_file);
    
    void initialize_power_vectors(std::string data_file);
    
    void initialize_parameter_vector(std::vector<double> &pars);
    
    public:
        pk_model(std::string pk_file, std::string data_file, std::vector<double> &pars);
        
        ~pk_model();
        
        void calculate_model();
        
        void write_model_to_file(std::string out_file);
        
        void normalize_covariance(std::vector<std::vector<double>> &covariance);
        
        void write_normalized_data_to_file(std::string out_file);
        
};

#endif
