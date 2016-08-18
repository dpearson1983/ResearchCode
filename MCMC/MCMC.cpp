#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <omp.h>
#include <harppi.h>
#include <libmcmc.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_permutation.h>
#include "model.h"

std::string filename(std::string filebase, int digits, int filenum, std::string fileext) {
    std::stringstream file;
    file << filebase << std::setw(digits) << std::setfill('0') << filenum << fileext;
    return file.str();
}

int main(int argc, char *argv[]) {
    if (argc == 1) {
        std::cout << "ERROR: Expected parameter file.\n";
        std::cout << "       Usage: " << argv[0] << " ParameterFile.params\n";
        std::cout << "       Note that the parameter file can have any extension\n";
        std::cout << "       so long as it is a plain text file formated as specified\n";
        std::cout << "       libharppi documentation." << std::endl;
        return 0;
    }
    parameters p;
    p.readParams(argv[1]);
    
    std::ifstream fin;
    std::ofstream fout;
    
    std::vector< double > modparams;
    for (int i = 0; i < p.geti["numParams"]; ++i) {
        modparams.push_back(p.getd("paramVals", i));
    }
