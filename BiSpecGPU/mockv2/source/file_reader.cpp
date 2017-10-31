#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <gsl/gsl_integration.h>
#include <galaxy.h>
#include <file_check.h>
#include <tpods.h>
#include "../include/file_reader.h"

size_t readPatchy(std::string file, std::vector<double> &nden, double red_min, double red_max, double P_FKP,
                double Omega_M, double Omega_L, vec3<double> r_min, vec3<double> L, vec3<int> N,
                vec3<double> &pk_nbw, vec3<double> &bk_nbw, bool randoms) {
    if (check_file_exists(file)) {
        int num = 0;
        std::ifstream fin(file);
        gsl_integration_workspace *w_gsl = gsl_integration_workspace_alloc(10000000);
        while (!fin.eof()) {
            double ra, dec, red, mass, n, b, rf, cp;
            if (randoms) {
                fin >> ra >> dec >> red >> n >> b >> rf >> cp;
            } else {
                fin >> ra >> dec >> red >> mass >> n >> b >> rf >> cp;
            }
            if (!fin.eof() && red >= red_min && red < red_max) {
                double w_fkp = 1.0/(1.0 + n*P_FKP);
                double w = w_fkp*(rf + cp - 1.0);
                galaxy<double> gal(ra, dec, red, 0.0, 0.0, 0.0, n, b, w);
                gal.cartesian(Omega_M, Omega_L, w_gsl);
                gal.bin(nden.data(), L, N, r_min, pk_nbw, bk_nbw, P_FKP, 
                        galFlags::INPUT_WEIGHT|galFlags::CIC);
                ++num;
            }
        }
        fin.close();
        gsl_integration_workspace_free(w);
    }
    return num;
}

size_t readQPM(std::string file, std::vector<double> &nden, double red_min, double red_max, double P_FKP,
               double Omega_M, double Omega_L, vec3<double> r_min, vec3<double> L, vec3<int> N,
               vec3<double> &pk_nbw, vec3<double> &bk_nbw, bool randoms) {
    if (check_file_exists(file)) {
        int num = 0;
        std::ifstream fin(file);
        gsl_integration_workspace *w_gsl = gsl_integration_workspace_alloc(10000000);
        while (!fin.eof()) {
            double ra, dec, red, w_fkp, w_rfcp = 1.0;
            if (randoms) {
                fin >> ra >> dec >> red >> w_fkp;
            } else {
                fin >> ra >> dec >> red >> w_fkp >> w_rfcp;
            }
            if (!fin.eof() && red >= red_min && red < red_max) {
                galaxy<double> gal(ra, dec, red, 0.0, 0.0, 0.0, n, b, w_fkp*w_rfcp);
                gal.cartesian(Omega_M, Omega_L, w_gsl);
                gal.bin(nden.data(), L, N, r_min, pk_nbw, bk_nbw, P_FKP, 
                        galFlags::INPUT_WEIGHT|galFlags::CIC);
                ++num;
            }
        }
        fin.close();
        gsl_integration_workspace_free(w);
    }
    return num;
}

void readFits(std::string file, std::vector<double> &nden, std::vector<std::string> cols, bool randoms) {
    std::stringstream message;
    message << "Requested function not yet implemented." << std::endl;
    throw std::runtime_error(message.str());
}
