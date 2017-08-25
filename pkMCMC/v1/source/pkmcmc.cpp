#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include "../include/pkmcmc.h"

std::random_device seeder;
std::mt19937_64 gen(seeder());
std::uniform_real_distribution<double> dist(-1.0, 1.0);

const 
std::vector<double> w_i = {0.096540088514728, 0.096540088514728, 0.095638720079275, 0.095638720079275,
                           0.093844399080805, 0.093844399080805, 0.091173878695764, 0.091173878695764,
                           0.087652093004404, 0.087652093004404, 0.083311924226947, 0.083311924226947,
                           0.078193895787070, 0.078193895787070, 0.072345794108849, 0.072345794108849,
                           0.065822222776362, 0.065822222776362, 0.058684093478536, 0.058684093478536,
                           0.050998059262376, 0.050998059262376, 0.042835898022227, 0.042835898022227,
                           0.034273862913021, 0.034273862913021, 0.025392065309262, 0.025392065309262,
                           0.016274394730906, 0.016274394730906, 0.007018610009470, 0.007018610009470};

const
std::vector<double> x_i = {-0.048307665687738, 0.048307665687738, -0.144471961582796, 0.144471961582796,
                           -0.239287362252137, 0.239287362252137, -0.331868602282128, 0.331868602282128,
                           -0.421351276130635, 0.421351276130635, -0.506899908932229, 0.506899908932229,
                           -0.587715757240762, 0.587715757240762, -0.663044266930215, 0.663044266930215,
                           -0.732182118740290, 0.732182118740290, -0.794483795967942, 0.794483795967942,
                           -0.849367613732570, 0.849367613732570, -0.896321155766052, 0.896321155766052,
                           -0.934906075937739, 0.934906075937739, -0.964762255587506, 0.964762255587506,
                           -0.985611511545268, 0.985611511545268, -0.997263861849481, 0.997263861849481};

double pkmcmc::model_func(std::vector<double> &pars, int j) {
    double result = 0.0;
    for (int i = 0; i < 32; ++i) {
        double mubar = sqrt(1.0 + x_i[i]*x_i[i]*((pars[3]*pars[3])/(pars[2]*pars[2]) - 1.0));
        double k_i = (pkmcmc::k[j]/pars[3])*mubar;
        if (k_i < pkmcmc::k_min || k_i > pkmcmc::k_max) {
            std::stringstream message;
            message << "Invalid value for interpolation." << std::endl;
            message << "     k = " << k_i << std::endl;
            message << "    mu = " << x_i[i] << std::endl;
            message << "a_para = " << pars[2] << std::endl;
            message << "a_perp = " << pars[3] << std::endl;
            message << " mubar = " << mubar << std::endl;
            throw std::runtime_error(message.str());
        }
        double mu = ((x_i[i]*pars[3])/pars[2])/mubar;
        double coeff = (pars[0] + mu*mu*pars[1])*(pars[0] + mu*mu*pars[1]);
        result += w_i[i]*coeff*gsl_spline_eval(pkmcmc::Pk, k_i, pkmcmc::acc);
    }
    return result;
}

void pkmcmc::model_calc(std::vector<double> &pars) {
    for (int i = 0; i < pkmcmc::num_data; ++i) {
        pkmcmc::model[i] = pkmcmc::model_func(pars, i);
    }
}

void pkmcmc::get_param_real() {
    for (int i = 0; i < pkmcmc::num_pars; ++i) {
        if (pkmcmc::limit_pars[i]) {
            if (pkmcmc::theta_0[i] + pkmcmc::param_vars[i] > pkmcmc::max[i]) {
                double center = pkmcmc::max[i] - pkmcmc::param_vars[i];
                pkmcmc::theta_i[i] = center + dist(gen)*pkmcmc::param_vars[i];
            } else if (pkmcmc::theta_0[i] - pkmcmc::param_vars[i] < pkmcmc::min[i]) {
                double center = pkmcmc::min[i] + pkmcmc::param_vars[i];
                pkmcmc::theta_i[i] = center + dist(gen)*pkmcmc::param_vars[i];
            } else {
                pkmcmc::theta_i[i] = pkmcmc::theta_0[i] + dist(gen)*pkmcmc::param_vars[i];
            }
        } else {
            pkmcmc::theta_i[i] = pkmcmc::theta_0[i] + dist(gen)*pkmcmc::param_vars[i];
        }
    }
}

double pkmcmc::calc_chi_squared() {
    double chisq = 0.0;
    for (int i = 0; i < pkmcmc::num_data; ++i) {
        for (int j = i; j < pkmcmc::num_data; ++j) {
            chisq += (pkmcmc::data[i] - pkmcmc::model[i])*Psi[i][j]*(pkmcmc::data[j] - pkmcmc::model[j]);
        }
    }
    return chisq;
}

bool pkmcmc::trial() {
    pkmcmc::get_param_real();
    pkmcmc::model_calc(pkmcmc::theta_i);
    pkmcmc::chisq_i = pkmcmc::calc_chi_squared();
    
    double L = exp(0.5*(pkmcmc::chisq_0 - pkmcmc::chisq_i));
    double R = (dist(gen) + 1.0)/2.0;
    
    if (L > R) {
        for (int i = 0; i < pkmcmc::num_pars; ++i)
            pkmcmc::theta_0[i] = pkmcmc::theta_i[i];
        pkmcmc::chisq_0 = pkmcmc::chisq_i;
        return true;
    } else {
        return false;
    }
}

void pkmcmc::write_theta_screen() {
    std::cout.precision(6);
    for (int i = 0; i < pkmcmc::num_pars; ++i) {
        std::cout.width(15);
        std::cout << pkmcmc::theta_0[i];
    }
    std::cout.width(15);
    std::cout << pkmcmc::chisq_0;
    std::cout.flush();
}

void pkmcmc::burn_in(int num_burn) {
    std::cout << "Burning the first " << num_burn << " trials to move to higher likelihood..." << std::endl;
    for (int i = 0; i < num_burn; ++i) {
        bool move = pkmcmc::trial();
        if (move) {
            std::cout << "\r";
            std::cout.width(10);
            std::cout << i;
            pkmcmc::write_theta_screen();
        }
    }
    std::cout << std::endl;
}

void pkmcmc::tune_vars() {
    std::cout << "Tuning the acceptance ratio to be close to 0.234..." << std::endl;
    double acceptance = 0.0;
    while (acceptance <= 0.233 || acceptance >= 0.235) {
        int accept = 0;
        for (int i = 0; i < 10000; ++i) {
            bool move = pkmcmc::trial();
            if (move) {
                std::cout << "\r";
                pkmcmc::write_theta_screen();
                accept++;
            }
        }
        std::cout << std::endl;
        acceptance = double(accept)/10000.0;
        
        if (acceptance <= 0.233) {
            for (int i = 0; i < pkmcmc::num_pars; ++i)
                pkmcmc::param_vars[i] *= 0.99;
        }
        if (acceptance >= 0.235) {
            for (int i = 0; i < pkmcmc::num_pars; ++i)
                pkmcmc::param_vars[i] *= 1.01;
        }
        std::cout << "acceptance = " << acceptance << std::endl;
    }
    std::ofstream fout;
    fout.open("pk_variances.dat", std::ios::out);
    for (int i = 0; i < pkmcmc::num_pars; ++i)
        fout << pkmcmc::param_vars[i] << " ";
    fout << "\n";
    fout.close();
}

pkmcmc::pkmcmc(std::string data_file, std::string cov_file, std::string pk_file, std::vector<double> &pars,
               std::vector<double> &vars, int int_workspace, double err_abs, double err_rel) {
    std::ifstream fin;
    
    std::cout << "Reading in power spectrum and creating interpolation spline..." << std::endl;
    if (std::ifstream(pk_file)) {
        fin.open(pk_file.c_str(), std::ios::in);
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
        pkmcmc::k_min = kin[0];
        pkmcmc::k_max = kin[kin.size() - 1];
        pkmcmc::Pk = gsl_spline_alloc(gsl_interp_cspline, pin.size());
        pkmcmc::acc = gsl_interp_accel_alloc();
        gsl_spline_init(pkmcmc::Pk, kin.data(), pin.data(), pin.size());
    } else {
        std::stringstream message;
        message << "Cannot open " << pk_file << std::endl;
        throw std::runtime_error(message.str());
    }
    
    std::cout << "Setting up the GSL integration workspace..." << std::endl;
    pkmcmc::workspace_size = int_workspace;
    pkmcmc::w = gsl_integration_workspace_alloc(int_workspace);
    pkmcmc::abs_err = err_abs;
    pkmcmc::rel_err = err_rel;
    
    std::cout << "Reading in and storing data file..." << std::endl;
    if (std::ifstream(data_file)) {
        fin.open(data_file.c_str(), std::ios::in);
        while (!fin.eof()) {
            double kt, P, sigma;
            fin >> kt >> P >> sigma;
            if (!fin.eof()) {
                pkmcmc::k.push_back(kt);
                pkmcmc::data.push_back(P);
                pkmcmc::model.push_back(0.0);
            }
        }
        fin.close();
    } else {
        std::stringstream message;
        message << "Cannot open " << data_file << std::endl;
        throw std::runtime_error(message.str());
    }
    
    pkmcmc::num_data = pkmcmc::data.size();
    std::cout << "num_data = " << pkmcmc::num_data << std::endl;
    std::cout << "number of k = " << pkmcmc::k.size() << std::endl;
    
    std::cout << "Reading in the covariance matrix and calculating its inverse..." << std::endl;
    
    gsl_matrix *cov = gsl_matrix_alloc(pkmcmc::num_data, pkmcmc::num_data);
    gsl_matrix *psi = gsl_matrix_alloc(pkmcmc::num_data, pkmcmc::num_data);
    gsl_permutation *perm = gsl_permutation_alloc(pkmcmc::num_data);
    
    if (std::ifstream(cov_file)) {
        fin.open(cov_file.c_str(), std::ios::in);
        for (int i = 0; i < pkmcmc::num_data; ++i) {
            for (int j = 0; j < pkmcmc::num_data; ++j) {
                double element;
                fin >> element;
                gsl_matrix_set(cov, i, j, element);
            }
        }
        fin.close();
    } else {
        std::stringstream message;
        message << "Cannot open " << cov_file << std::endl;
        throw std::runtime_error(message.str());
    }
    
    int s;
    gsl_linalg_LU_decomp(cov, perm, &s);
    gsl_linalg_LU_invert(cov, perm, psi);
    
    for (int i = 0; i < pkmcmc::num_data; ++i) {
        std::vector<double> row;
        row.reserve(pkmcmc::num_data);
        for (int j = 0; j < pkmcmc::num_data; ++j) {
            row.push_back(gsl_matrix_get(psi, i, j));
        }
        pkmcmc:Psi.push_back(row);
    }
    
    gsl_matrix_free(cov);
    gsl_matrix_free(psi);
    gsl_permutation_free(perm);
    
    std::cout << "Setting initial parameters and variances..." << std::endl;
    pkmcmc::num_pars = pars.size();
    std::cout << "num_pars = " << pkmcmc::num_pars << std::endl;
    
    for (int i = 0; i < pkmcmc::num_pars; ++i) {
        pkmcmc::theta_0.push_back(pars[i]);
        pkmcmc::theta_i.push_back(0.0);
        pkmcmc::limit_pars.push_back(false);
        pkmcmc::max.push_back(0.0);
        pkmcmc::min.push_back(0.0);
        pkmcmc::param_vars.push_back(vars[i]*pars[i]);
    }
    
    std::cout << "Calculating initial model and chi^2..." << std::endl;
    pkmcmc::model_calc(pkmcmc::theta_0);
    pkmcmc::chisq_0 = pkmcmc::calc_chi_squared();
}

void pkmcmc::check_init() {
    std::cout << "Number of data points: " << pkmcmc::num_data << std::endl;
    std::cout << "    data.size()      = " << pkmcmc::data.size() << std::endl;
    std::cout << "    model.size()     = " << pkmcmc::model.size() << std::endl;
    std::cout << "    k.size()         = " << pkmcmc::k.size() << std::endl;
    std::cout << "Number of parameters:  " << pkmcmc::num_pars << std::endl;
    std::cout << "    theta_0.size()   = " << pkmcmc::theta_0.size() << std::endl;
    std::cout << "    theta_i.size()   = " << pkmcmc::theta_i.size() << std::endl;
    std::cout << "    limit_pars.size()= " << pkmcmc::limit_pars.size() << std::endl;
    std::cout << "    min.size()       = " << pkmcmc::min.size() << std::endl;
    std::cout << "    max.size()       = " << pkmcmc::max.size() << std::endl;
    std::cout << "    param_vars.size()= " << pkmcmc::param_vars.size() << std::endl;
}

void pkmcmc::set_param_limits(std::vector<bool> &lim_pars, std::vector<double> &min_in,
                              std::vector<double> &max_in) {
    for (int i = 0; i < pkmcmc::num_pars; ++i) {
        pkmcmc::limit_pars[i] = lim_pars[i];
        pkmcmc::max[i] = max_in[i];
        pkmcmc::min[i] = min_in[i];
    }
}

void pkmcmc::run_chain(int num_draws, std::string reals_file, bool new_chain) {
    int num_old_rels = 0;
    if (new_chain) {
        std::cout << "Starting new chain..." << std::endl;
        pkmcmc::burn_in(10000);
        pkmcmc::tune_vars();
    } else {
        std::cout << "Resuming previous chain..." << std::endl;
        std::ifstream fin;
        if (std::ifstream("pk_variances.dat")) {
            fin.open("pk_variances.dat", std::ios::in);
            for (int i = 0; i < pkmcmc::num_pars; ++i) {
                double var;
                fin >> var;
                pkmcmc::param_vars[i] = var;
            }
            fin.close();
        } else {
            std::stringstream message;
            message << "Request to resume chain failed. pk_varainces.dat was not found." << std::endl;
            throw std::runtime_error(message.str());
        }
        
        if (std::ifstream(reals_file)) {
            fin.open(reals_file.c_str(), std::ios::in);
            while (!fin.eof()) {
                num_old_rels++;
                for (int i = 0; i < pkmcmc::num_pars; ++i)
                    fin >> pkmcmc::theta_0[i];
                fin >> pkmcmc::chisq_0;
            }
            fin.close();
            num_old_rels--;
        } else {
            std::stringstream message;
            message << "Request to resume chain failed. Cannot open " << reals_file << std::endl;
            throw std::runtime_error(message.str());
        }
    }
    
    std::ofstream fout;
    fout.open(reals_file.c_str(), std::ios::app);
    fout.precision(15);
    for (int i = 0; i < num_draws; ++i) {
        bool move = pkmcmc::trial();
        for (int par = 0; par < pkmcmc::num_pars; ++par) {
            fout << pkmcmc::theta_0[par] << " ";
        }
        fout << pkmcmc::chisq_0 << "\n";
        if (move) {
            std::cout << "\r";
            std::cout.width(15);
            std::cout << i + num_old_rels;
            pkmcmc::write_theta_screen();
        }
    }
    std::cout << std::endl;
    fout.close();
}

void pkmcmc::clean_up_gsl() {
    gsl_spline_free(pkmcmc::Pk);
    gsl_interp_accel_free(pkmcmc::acc);
    gsl_integration_workspace_free(pkmcmc::w);
}
