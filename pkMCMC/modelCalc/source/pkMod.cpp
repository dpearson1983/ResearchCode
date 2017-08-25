#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include "../include/file_check.h"
#include "../include/pkMod.h"

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

double pk_model::model_func(std::vector<double> &pars, int j) {
    double result = 0.0;
    for (int i = 0; i < 32; ++i) {
        double mubar = sqrt(1.0 + x_i[i]*x_i[i]*((pars[3]*pars[3])/(pars[2]*pars[2]) - 1.0));
        double k_i = (pk_model::k[j]/pars[3])*mubar;
        if (k_i < pk_model::k_min || k_i > pk_model::k_max) {
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
        result += w_i[i]*coeff*gsl_spline_eval(pk_model::Pk_m, k_i, pk_model::acc);
    }
    return result;
}

void pk_model::model_calc(std::vector<double> &pars) {
    for (int i = 0; i < pk_model::num_data; ++i) {
        pk_model::model[i] = pk_model::model_func(pars, i);
    }
}

void pk_model::initialize_spline(std::string pk_file) {
    if (check_file_exists(pk_file)) {
        std::vector<double> kin;
        std::vector<double> pin;
        std::ifstream fin(pk_file);
        while(!fin.eof()) {
            double kt, pt;
            fin >> kt >> pt;
            if (!fin.eof()) {
                kin.push_back(kt);
                pin.push_back(pt);
            }
        }
        fin.close();
        pk_model::k_min = kin[0];
        pk_model::k_max = kin[kin.size() - 1];
        pk_model::Pk_m = gsl_spline_alloc(gsl_interp_cspline, pin.size());
        pk_model::acc = gsl_interp_accel_alloc();
        gsl_spline_init(pk_model::Pk_m, kin.data(), pin.data(), pin.size());
    }
}

void pk_model::initialize_power_vectors(std::string data_file) {
    if (check_file_exists(data_file)) {
        std::ifstream fin(data_file);
        while(!fin.eof()) {
            double kt, pt, sigma;
            fin >> kt >> pt >> sigma;
            if (!fin.eof()) {
                pk_model::k.push_back(kt);
                pk_model::data.push_back(pt);
                pk_model::model.push_back(0.0);
            }
        }
        fin.close();
        pk_model::num_data = pk_model::k.size();
    }
}

void pk_model::initialize_parameter_vector(std::vector<double> &pars) {
    pk_model::num_pars = pars.size();
    for (int i = 0; i < pk_model::num_pars; ++i)
        pk_model::params.push_back(pars[i]);
}

pk_model::pk_model(std::string pk_file, std::string data_file, std::vector<double> &pars) {    
    pk_model::initialize_spline(pk_file);
    pk_model::initialize_power_vectors(data_file);
    pk_model::initialize_parameter_vector(pars);
}

pk_model::~pk_model() {
    gsl_spline_free(pk_model::Pk_m);
    gsl_interp_accel_free(pk_model::acc);
}

void pk_model::calculate_model() {
    pk_model::model_calc(pk_model::params);
}

void pk_model::write_model_to_file(std::string out_file) {
    std::ofstream fout(out_file);
    for (int i = 0; i < pk_model::num_data; ++i)
        fout << pk_model::k[i] << " " << pk_model::model[i] << "\n";
    fout.close();
}

void pk_model::normalize_covariance(std::vector<std::vector<double>> &covariance) {
    for (int i = 0; i < pk_model::num_data; ++i) {
        for (int j = 0; j < pk_model::num_data; ++j) {
            covariance[i][j] /= (pk_model::model[i]*pk_model::model[j]);
        }
    }
}

void pk_model::write_normalized_data_to_file(std::string out_file) {
    std::ofstream fout(out_file);
    for (int i = 0; i < pk_model::num_data; ++i)
        fout << pk_model::k[i] << " " << pk_model::data[i]/pk_model::model[i] << "\n";
}
