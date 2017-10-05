#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <vector>
#include <string>
#include <cmath>
#include <omp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_permutation.h>
#include "../include/bkmcmc.h"
#include "../include/file_check.h"
#include "../include/tpods.h"
#include "../include/harppi.h"

std::random_device seeder;
std::mt19937_64 gen(seeder());
std::uniform_real_distribution<double> dist(-1.0, 1.0);

double bkmcmc::get_model_power(std::vector<double> &pars, double k_i) {
    double P_nw = gsl_spline_eval(bkmcmc::Pk_nw, k_i, bkmcmc::acc_nw);
    double P_nwa = gsl_spline_eval(bkmcmc::Pk_nw, k_i/pars[1], bkmcmc::acc_nw);
    double P_bao = gsl_spline_eval(bkmcmc::Pk_bao, k_i/pars[1], bkmcmc::acc_bao);
    double damp = exp(-0.5*pars[2]*pars[2]*k_i*k_i);
    double broadband = pars[3]*k_i + pars[4] + pars[5]/k_i + pars[6]/(k_i*k_i) + pars[7]/(k_i*k_i*k_i);
    return (pars[0]*pars[0]*P_nw + broadband)*(1.0 + (P_bao/P_nwa - 1.0)*damp);
}
    

void bkmcmc::model_calc(std::vector<double> &pars) {
#pragma omp parallel for private(bkmcmc::Pk_bao, bkmcmc::acc_bao, bkmcmc::Pk_nw, bkmcmc::acc_nw)
    for (int i = 0; i < bkmcmc::num_data; ++i) {
        double P_1 = get_model_power(pars, bkmcmc::k[i].x);
        double P_2 = get_model_power(pars, bkmcmc::k[i].y);
        double P_3 = get_model_power(pars, bkmcmc::k[i].z);
        double broadband = pars[13]*(k[i].x/k[i].y + k[i].y/k[i].z + k[i].z/k[i].x + k[i].y/k[i].x + k[i].z/k[i].y + k[i].x/k[i].z) + pars[14]*((k[i].x*k[i].x)/(k[i].y*k[i].y) + (k[i].y*k[i].y)/(k[i].z*k[i].z) + (k[i].z*k[i].z)/(k[i].x*k[i].x) + (k[i].y*k[i].y)/(k[i].x*k[i].x) + (k[i].z*k[i].z)/(k[i].y*k[i].y) + (k[i].x*k[i].x)/(k[i].z*k[i].z)) + pars[15]*((k[i].x*k[i].x)/(k[i].y*k[i].z) + (k[i].y*k[i].y)/(k[i].z*k[i].x) + (k[i].z*k[i].z)/(k[i].x*k[i].y)) + pars[12]*(k[i].x + k[i].y + k[i].z) + pars[13] + pars[8]*(1.0/(k[i].x*k[i].y) + 1.0/(k[i].y*k[i].z) + 1.0/(k[i].z*k[i].x)) + pars[9]/(k[i].x*k[i].y*k[i].z) + pars[10] + pars[11]*(k[i].x + k[i].y + k[i].z);
        bkmcmc::Bk[i] = 2.0*(P_1*P_2 + P_2*P_3 + P_3*P_1) + broadband;
    }
}

void bkmcmc::get_param_real() {
    for (int i = 0; i < bkmcmc::num_pars; ++i) {
        if (bkmcmc::limit_pars[i]) {
            if (bkmcmc::theta_0[i] + bkmcmc::param_vars[i] > bkmcmc::max[i]) {
                double center = bkmcmc::max[i] - bkmcmc::param_vars[i];
                bkmcmc::theta_i[i] = center + dist(gen)*bkmcmc::param_vars[i];
            } else if (bkmcmc::theta_0[i] - bkmcmc::param_vars[i] < bkmcmc::min[i]) {
                double center = bkmcmc::min[i] + bkmcmc::param_vars[i];
                bkmcmc::theta_i[i] = center + dist(gen)*bkmcmc::param_vars[i];
            } else {
                bkmcmc::theta_i[i] = bkmcmc::theta_0[i] + dist(gen)*bkmcmc::param_vars[i];
            }
        } else {
            bkmcmc::theta_i[i] = bkmcmc::theta_0[i] + dist(gen)*bkmcmc::param_vars[i];
        }
    }
}

double bkmcmc::calc_chi_squared() {
    double chisq = 0.0;
    if (bkmcmc::full_covar) {
        for (int i = 0; i < bkmcmc::num_data; ++i) {
            for (int j = i; j < bkmcmc::num_data; ++j) {
                chisq += (bkmcmc::data[i] - bkmcmc::Bk[i])*Psi[i][j]*(bkmcmc::data[j] - bkmcmc::Bk[j]);
            }
        }
    } else {
        for (int i = 0; i < bkmcmc::num_data; ++i)
            chisq += (bkmcmc::data[i] - bkmcmc::Bk[i])*Psi[i][i]*(bkmcmc::data[i] - bkmcmc::Bk[i]);
    }
    return chisq;
}

bool bkmcmc::trial() {
    bkmcmc::get_param_real();
    bkmcmc::model_calc(bkmcmc::theta_i);
    bkmcmc::chisq_i = bkmcmc::calc_chi_squared();
    
    double L = exp(0.5*(bkmcmc::chisq_0 - bkmcmc::chisq_i));
    double R = (dist(gen) + 1.0)/2.0;
    
    if (L > R) {
        for (int i = 0; i < bkmcmc::num_pars; ++i)
            bkmcmc::theta_0[i] = bkmcmc::theta_i[i];
        bkmcmc::chisq_0 = bkmcmc::chisq_i;
        return true;
    } else {
        return false;
    }
}

void bkmcmc::write_theta_screen() {
    std::cout.precision(6);
    for (int i = 0; i < bkmcmc::num_write; ++i) {
        std::cout.width(15);
        std::cout << bkmcmc::theta_0[i];
    }
    std::cout.width(15);
    std::cout << bkmcmc::chisq_0;
    std::cout.flush();
}

void bkmcmc::burn_in(int num_burn) {
    std::cout << "Buring the first " << bkmcmc::num_burn << " trials to move to higher likelihood..." << std::endl;
    int accept = 0;
    int count = 0;
    for (int i = 0; i < bkmcmc::num_burn; ++i) {
        count++;
        bool move = bkmcmc::trial();
        if (move) {
            accept++;
            std::cout << "\r";
            std::cout.width(5);
            std::cout << i;
            bkmcmc::write_theta_screen();
            std::cout.width(10);
            std::cout << double(accept)/double(count);
            std::cout.flush();
        }
    }
    std::cout << std::endl;
}

void bkmcmc::tune_vars() {
    std::cout << "Tuning acceptance ratio..." << std::endl;
    double acceptance = 0.0;
    while (acceptance <= 0.233 || acceptance >= 0.235) {
        int accept = 0;
        for (int i = 0; i < 10000; ++i) {
            bool move = bkmcmc::trial();
            if (move) {
                std::cout << "\r";
                bkmcmc::write_theta_screen();
                accept++;
            }
        }
        std::cout << std::endl;
        acceptance = double(accept)/10000.0;
        
        if (acceptance <= 0.233) {
            for (int i = 0; i < bkmcmc::num_pars; ++i)
                bkmcmc::param_vars[i] *= 0.99;
        }
        if (acceptance >= 0.235) {
            for (int i = 0; i < bkmcmc::num_pars; ++i)
                bkmcmc::param_vars[i] *= 1.01;
        }
        std::cout << "acceptance = " << acceptance << std::endl;
    }
    std::ofstream fout(bkmcmc::vars_file);
    fout.precision(15);
    for (int i = 0; i < bkmcmc::num_pars; ++i)
        fout << bkmcmc::param_vars[i] << " ";
    fout << "\n";
    fout.close();
}

void bkmcmc::check_init() {
    std::cout << "Number of data points: " << bkmcmc::num_data << std::endl;
    std::cout << "    data.size()      = " << bkmcmc::data.size() << std::endl;
    std::cout << "    Bk_bao.size()    = " << bkmcmc::Bk.size() << std::endl;
    std::cout << "    Psi.size()       = " << bkmcmc::Psi.size() << std::endl;
    std::cout << "Number of parameters:  " << bkmcmc::num_pars << std::endl;
    std::cout << "    theta_0.size()   = " << bkmcmc::theta_0.size() << std::endl;
    std::cout << "    theta_i.size()   = " << bkmcmc::theta_i.size() << std::endl;
    std::cout << "    limit_pars.size()= " << bkmcmc::limit_pars.size() << std::endl;
    std::cout << "    min.size()       = " << bkmcmc::min.size() << std::endl;
    std::cout << "    max.size()       = " << bkmcmc::max.size() << std::endl;
    std::cout << "    param_vars.size()= " << bkmcmc::param_vars.size() << std::endl;
}

void bkmcmc::initialize_gsl_spline(std::string file, bool bao) {
    std::cout << file << std::endl;
    if (check_file_exists(file)) {
        std::ifstream fin(file);
        std::vector<double> x;
        std::vector<double> y;
        while (!fin.eof()) {
            double xt, yt;
            fin >> xt >> yt;
            if (!fin.eof()) {
                x.push_back(xt);
                y.push_back(yt);
            }
        }
        fin.close();
        if (bao) {
            bkmcmc::Pk_bao = gsl_spline_alloc(gsl_interp_cspline, y.size());
            bkmcmc::acc_bao = gsl_interp_accel_alloc();
            gsl_spline_init(bkmcmc::Pk_bao, x.data(), y.data(), y.size());
            std::cout << gsl_spline_eval(bkmcmc::Pk_bao, 0.1, acc_bao) << std::endl;
        } else {
            bkmcmc::Pk_nw = gsl_spline_alloc(gsl_interp_cspline, y.size());
            bkmcmc::acc_nw = gsl_interp_accel_alloc();
            gsl_spline_init(bkmcmc::Pk_nw, x.data(), y.data(), y.size());
            std::cout << gsl_spline_eval(bkmcmc::Pk_nw, 0.1, acc_nw) << std::endl;
        }
    }
}

bkmcmc::bkmcmc(parameters &p) {
    std::cout << "Initializing bispectrum MCMC class object..." << std::endl;
    
    std::cout << "    Setting various integer and boolean values..." << std::endl;
    bkmcmc::num_old_reals = 0;
    bkmcmc::num_data = p.geti("num_data");
    bkmcmc::num_pars = p.geti("num_pars");
    bkmcmc::num_draws = p.geti("num_draws");
    bkmcmc::num_write = p.geti("num_write");
    bkmcmc::num_burn = p.geti("num_burn");
    bkmcmc::full_covar = p.getb("full_covar");
    bkmcmc::new_chain = p.getb("new_chain");
    bkmcmc::reals_file = p.gets("reals_file");
    bkmcmc::vars_file = p.gets("vars_file");
    
    std::cout << "    Reading in data file and initializing data and model vectors..." << std::endl;
    if (check_file_exists(p.gets("data_file"))) {
        std::ifstream fin(p.gets("data_file"));
        for (int i = 0; i < bkmcmc::num_data; ++i) {
            vec3<double> kin;
            double B, var;
            fin >> kin.x >> kin.y >> kin.z >> B >> var;
            if (!fin.eof()) {
                bkmcmc::k.push_back(kin);
                bkmcmc::data.push_back(B);
                bkmcmc::Bk.push_back(0.0);
                if (!bkmcmc::full_covar) {
                    std::vector<double> row(bkmcmc::num_data);
                    row[i] = 1.0/var;
                    bkmcmc::Psi.push_back(row);
                }
            }
        }
        fin.close();
    }
    
    std::cout << "k.size() = " << bkmcmc::k.size() << std::endl;
    std::cout << "Bk.size() = " << bkmcmc::Bk.size() << std::endl;
    std::cout << "Psi.size() = " << bkmcmc::Psi.size() << std::endl;
    std::cout << "num_data = " << bkmcmc::num_data << std::endl;
    
    if (bkmcmc::full_covar) {
        if (check_file_exists(p.gets("covar_file"))) {
            std::cout << "    Reading in covariance matrix, inverting it, and storing..." << std::endl;
            std::ifstream fin(p.gets("covar_file"));
            gsl_matrix *cov = gsl_matrix_alloc(bkmcmc::num_data, bkmcmc::num_data);
            gsl_matrix *psi = gsl_matrix_alloc(bkmcmc::num_data, bkmcmc::num_data);
            gsl_permutation *perm = gsl_permutation_alloc(bkmcmc::num_data);
            
            for (int i = 0; i < bkmcmc::num_data; ++i) {
                for (int j = 0; j < bkmcmc::num_data; ++j) {
                    double element;
                    fin >> element;
                    gsl_matrix_set(cov, i, j, element);
                }
            }
            fin.close();
            
            double scale = (1000.0 - 691.0 - 2.0)/(1000.0 - 1.0);
            
            int s;
            gsl_linalg_LU_decomp(cov, perm, &s);
            gsl_linalg_LU_invert(cov, perm ,psi);
            
            for (int i = 0; i < bkmcmc::num_data; ++i) {
                std::vector<double> row;
                row.reserve(bkmcmc::num_data);
                for (int j = 0; j < bkmcmc::num_data; ++j) {
                    row.push_back(scale*gsl_matrix_get(psi, i, j));
                }
                bkmcmc::Psi.push_back(row);
            }
            
            gsl_matrix_free(cov);
            gsl_matrix_free(psi);
            gsl_permutation_free(perm);
        }
    }
    
    std::cout << "    Setting up power spectra interpolation splines..." << std::endl;
    bkmcmc::initialize_gsl_spline(p.gets("in_bao_power_file"), true);
    bkmcmc::initialize_gsl_spline(p.gets("in_nw_power_file"), false);
    
    std::cout << "    Testing splines with k = 0.1..." << std::endl;
    std::cout << "      Pk_bao(k = 0.1) = " << gsl_spline_eval(bkmcmc::Pk_bao, 0.1, bkmcmc::acc_bao) << std::endl;
    std::cout << "      Pk_nw(k = 0.1) = " << gsl_spline_eval(bkmcmc::Pk_nw, 0.1, bkmcmc::acc_nw) << std::endl;
    
    if (bkmcmc::new_chain) {
        std::cout << "    Setting the initial parameter values and step sizes..." << std::endl;
        for (int i = 0; i < bkmcmc::num_pars; ++i) {
            bkmcmc::theta_0.push_back(p.getd("pars", i));
            bkmcmc::theta_i.push_back(0.0);
            bkmcmc::param_vars.push_back(p.getd("vars", i));
            bkmcmc::min.push_back(0.0);
            bkmcmc::max.push_back(0.0);
            bkmcmc::limit_pars.push_back(false);
        }
        
        std::cout << "    Calculating initial model..." << std::endl;
        bkmcmc::model_calc(bkmcmc::theta_0);
        std::cout << "    Calculating initial chi^2..." << std::endl;
        bkmcmc::chisq_0 = bkmcmc::calc_chi_squared();
    } else {
        std::cout << "    Getting last parameter realization and adjusted step sizes..." << std::endl;
        if (check_file_exists(bkmcmc::vars_file)) {
            std::ifstream fin(bkmcmc::vars_file);
            for (int i = 0; i < bkmcmc::num_pars; ++i) {
                double var;
                fin >> var;
                bkmcmc::param_vars.push_back(var);
                bkmcmc::theta_0.push_back(0.0);
                bkmcmc::theta_i.push_back(0.0);
                bkmcmc::min.push_back(0.0);
                bkmcmc::max.push_back(0.0);
                bkmcmc::limit_pars.push_back(false);
            }
            fin.close();
        }
        
        if (check_file_exists(bkmcmc::reals_file)) {
            std::ifstream fin(bkmcmc::reals_file);
            while (!fin.eof()) {
                for (int i = 0; i < bkmcmc::num_pars; ++i)
                    fin >> bkmcmc::theta_0[i];
                fin >> bkmcmc::chisq_0;
                bkmcmc::num_old_reals++;
            }
            fin.close();
            bkmcmc::num_old_reals--;
        }
    }
    
    if (p.getb("set_parameter_limits")) {
        std::cout << "    Setting limits on the model parameters..." << std::endl;
        for (int i = 0; i < bkmcmc::num_pars; ++i) {
            bkmcmc::limit_pars[i] = p.getb("limit_pars", i);
            bkmcmc::min[i] = p.getd("min", i);
            bkmcmc::max[i] = p.getd("max", i);
        }
    }
    
    bkmcmc::check_init();
}

bkmcmc::~bkmcmc() {
    gsl_spline_free(bkmcmc::Pk_bao);
    gsl_spline_free(bkmcmc::Pk_nw);
    gsl_interp_accel_free(bkmcmc::acc_bao);
    gsl_interp_accel_free(bkmcmc::acc_nw);
}

void bkmcmc::run_chain() {
    bkmcmc::burn_in(bkmcmc::num_burn);
    bkmcmc::tune_vars();
    
    std::ofstream fout;
    fout.open(reals_file.c_str(), std::ios::app);
    fout.precision(15);
    for (int i = 0; i < num_draws; ++i) {
        bool move = bkmcmc::trial();
        for (int par = 0; par < bkmcmc::num_pars; ++par) {
            fout << bkmcmc::theta_0[par] << " ";
        }
        fout << bkmcmc::chisq_0 << "\n";
        if (move) {
            std::cout << "\r";
            std::cout.width(15);
            std::cout << i + bkmcmc::num_old_reals;
            bkmcmc::write_theta_screen();
        }
    }
    std::cout << std::endl;
    fout.close();
}
