#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>
#include <galaxy.h>
#include <file_check.h>
#include <tpods.h>
#include "../include/file_reader.h"

size_t readPatchy(std::string file, desityField &nden, cosmology &cos, vec2<double> red_lim, double P_FKP, 
                  bool randoms) {
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
            if (!fin.eof() && red >= red_lim.x && red < red_lim.y) {
                double w_fkp = 1.0/(1.0 + n*P_FKP);
                double w = w_fkp;
                galaxy<double> gal(ra, dec, red, 0.0, n, b, w, rf, cp, P_FKP);
                nden.bin(gal, cos, w_gsl, "CIC");
                ++num;
            }
        }
        fin.close();
        gsl_integration_workspace_free(w);
    }
    return num;
}

size_t readQPM(std::string file,desityField &nden, cosmology &cos, vec2<double> red_lim, double P_FKP, 
               bool randoms, gsl_spline *NofZ, gsl_interp_accel *acc) {
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
                double n = gsl_spline_eval(NofZ, red, acc);
                galaxy<double> gal(ra, dec, red, 0.0, n, 0.0, w_fkp, w_rfcp, 0.0, P_FKP);
                nden.bin(gal, cos, w_gsl, "CIC");
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
