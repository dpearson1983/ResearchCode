/* powerspec.h
 * David W. Pearson
 * October 24, 2017
 * 
 * This header contains all the functionality needed to calculate the power spectrum model via the
 * Anderson et al. 2012 method. This method calculates a smooth power spectrum whose shape is modified
 * by a polynomial to account for the non-linear features in the data. The model has been slightly
 * modified from that of Anderson et al. to include a k^2 term. This is due to testing which showed that
 * the polynomial without that term was insufficient to completely correct the shape, leading to a bias
 * in the value of alpha.
 * 
 * An object of this class will be a member of the mcmc class along with an object of the bispec class.
 * Together, the two will enable the complete calculation of the model.
 * 
 */

#ifndef _POWERSPEC_H_
#define _POWERSPEC_H_

#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <gsl/gsl_spline.h>
#include "file_check.h" // contains check_file_exists(std::string file_name) function

class powerspec{
    size_t num_vals; // Keep track of the number of elements
    std::vector<double> k, P; // Storage for the model values, P, calculated at the specific k values
    gsl_spline *Pk_bao, *Pk_nw; // GSL splines for getting values at arbitrary k values
    gsl_interp_accel *acc_bao, *acc_nw; // GSL acceleration workspaces
    
    // Reads the data stored in in_file and initializes specified spline.
    void init_spline(std::string in_file, gsl_spline *Pk);
    
    // Reads the data file to determine the particular k values where the model needs to be evaluated. Also
    // the data itself is stored temporarily in P which can then be copied over to the mcmc data member.
    void read_data_file(std::string in_file);
    
    public:
        // Default constructor
        powerspec();
        
        // Constructor which automatically calls the initialize function to do all the setup needed
        powerspec(std::string input_data_file, std::string input_bao_file, std::string input_nw_file);
        
        // Initializes both splines and their acceleration spaces, reads in the data file and sets
        // the number of values.
        void initialize(std::string input_data_file, std::string input_bao_file, std::string input_nw_file);
        
        // Calculates the model power spectrum and stores it in P. NOTE: This means that once called,
        // the original data is no longer stored here, but should be copied to the mcmc object.
        void calculate(std::vector<double> &pars);
        
        // Returns the model value stored at P[i]
        double get(int i);
        
        // Returns the number of values
        size_t size();
};

void powerspec::init_spline(std::string in_file, gsl_spline *Pk) {
    // Check to make sure the given file name actually exists to avoid consuming all system memory
    if (check_file_exists(in_file)) {
        // Set things up to read in the file and put contents in temporary storage, then do that
        std::ifstream fin(in_file);
        std::vector<double> kin;
        std::vector<double> pin;
        while (!fin.eof()) {
            double kt, pt;
            fin >> kt >> pt;
            if (!fin.eof()) {
                kin.push_back(kt);
                pin.push_back(pt);
            }
        }
        fin.close();
        
        // Use the appropriate GSL functions to initialize the spline and workspace
        Pk = gsl_spline_alloc(gsl_interp_cspline, pin.size());
        gsl_spline_init(Pk, kin.data(), pin.data(), pin.size());
    }
}

void powerspec::read_data_file(std::string in_file) {
    // Check to make sure the given file name actually exists to avoid consuming all system memory
    if (check_file_exists(in_file)) {
        // Read file and store in data members
        std::ifstream fin(in_file);
        while (!fin.eof()) {
            double kt, pt, n;
            fin >> kt >> pt >> n;
            if (!fin.eof()) {
                powerspec::k.push_back(kt);
                powerspec::P.push_back(pt);
            }
        }
        fin.close();
        powerspec::num_vals = powerspec::P.size();
    }
}

powerspec::powerspec() {
    // Do something inconsequential
    powerspec::num_vals = 1;
}

powerspec::powerspec(std::string input_data_file, std::string input_bao_file, std::string input_nw_file) {
    powerspec::initialize(input_data_file, input_bao_file, input_nw_file);
}

void powerspec::initialize(std::string input_data_file, std::string input_bao_file, 
                           std::string input_nw_file) {
    if (check_file_exists(input_bao_file)) {
        // Set things up to read in the file and put contents in temporary storage, then do that
        std::ifstream fin(input_bao_file);
        std::vector<double> kin;
        std::vector<double> pin;
        while (!fin.eof()) {
            double kt, pt;
            fin >> kt >> pt;
            if (!fin.eof()) {
                kin.push_back(kt);
                pin.push_back(pt);
            }
        }
        fin.close();
        
        // Use the appropriate GSL functions to initialize the spline and workspace
        powerspec::Pk_bao = gsl_spline_alloc(gsl_interp_cspline, pin.size());
        gsl_spline_init(powerspec::Pk_bao, kin.data(), pin.data(), pin.size());
    }
    powerspec::acc_bao = gsl_interp_accel_alloc();
    if (check_file_exists(input_nw_file)) {
        // Set things up to read in the file and put contents in temporary storage, then do that
        std::ifstream fin(input_nw_file);
        std::vector<double> kin;
        std::vector<double> pin;
        while (!fin.eof()) {
            double kt, pt;
            fin >> kt >> pt;
            if (!fin.eof()) {
                kin.push_back(kt);
                pin.push_back(pt);
            }
        }
        fin.close();
        
        // Use the appropriate GSL functions to initialize the spline and workspace
        powerspec::Pk_nw = gsl_spline_alloc(gsl_interp_cspline, pin.size());
        gsl_spline_init(powerspec::Pk_nw, kin.data(), pin.data(), pin.size());
    }
    powerspec::acc_nw = gsl_interp_accel_alloc();
    powerspec::read_data_file(input_data_file);
}

void powerspec::calculate(std::vector<double> &pars) {
    for (size_t i = 0; i < powerspec::num_vals; ++i) {
//         std::cout << "Get k_i..." << std::endl;
        double k_i = powerspec::k[i]; // Convenient shorthand
        // Bispectrum needs a_para and a_perp, so they have to be combined here to calculate the power
        // spectrum. See Anderson et al. for details.
//         std::cout << "Calculate alpha..." << std::endl;
        double alpha = pow(pars[3]*pars[4]*pars[4], 1.0/3.0);
//         std::cout << "No-wiggle splines..." << std::endl;
        double P_nw = gsl_spline_eval(powerspec::Pk_nw, k_i, powerspec::acc_nw);
        double P_nwa = gsl_spline_eval(powerspec::Pk_nw, k_i/alpha, powerspec::acc_nw);
//         std::cout << "BAO spline..." << std::endl;
        double P_bao = gsl_spline_eval(powerspec::Pk_bao, k_i/alpha, powerspec::acc_bao);
//         std::cout << "Damp..." << std::endl;
        double damp = exp(-0.5*pars[7]*pars[7]*k_i*k_i);
//         std::cout << "Broadband..." << std::endl;
        double broadband = pars[8]*k_i*k_i + pars[9]*k_i + pars[10] + pars[11]/k_i + pars[12]/(k_i*k_i)
                           + pars[13]/(k_i*k_i*k_i);
//         std::cout << "Final calculation..." << std::endl;
        powerspec::P[i] = (pars[6]*pars[6]*P_nw + broadband)*(1.0 + (P_bao/P_nwa - 1.0)*damp);
    }
}

double powerspec::get(int i) {
    return powerspec::P[i];
}

size_t powerspec::size() {
    return powerspec::num_vals;
}

#endif
