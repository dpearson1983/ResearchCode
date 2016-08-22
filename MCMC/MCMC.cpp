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
    
    gsl_matrix *cov = gsl_matrix_alloc(p.geti("numVals"), p.geti("numVals"));
    gsl_matrix *Psi = gsl_matrix_alloc(p.geti("numVals"), p.gets("numVals"));
    double detPsi;
    
    readCov(p.gets("covFile"), p.geti("numVals"), cov, p.gets("covFormat"));
    calcPsi(cov, Psi, &detPsi, p.geti("numVals"), p.geti("numMeas"));
    
    gsl_matrix_free(cov);
    
    for (int file = p.geti("startNum"); file < p.geti("numFiles")+p.geti("startNum"); ++file) {
        std::string infile = filename(p.gets("inBase"), p.geti("digits"), file, p.gets("ext"));
        std::string outfile = filename(p.gets("outBase"), p.geti("digits"), file, p.gets("ext"));
        
        std::vector< double > modParams;
        std::vector< bool > limitParams;
        std::vector< double > paramMins;
        std::vector< double > paramMaxs;
        for (int i = 0; i < p.geti("numParams"); ++i) {
            modParams.push_back(p.getd("paramVals", i));
            if (p.checkParam("limitParams")) {
                limitParams.push_back(p.getb("limitParams", i));
                paramMins.push_back(p.getd("paramMins", i));
                paramMaxs.push_back(p.getd("paramMins", i));
            }
        }
        
        std::vector< double > data(p.geti("numVals"));
        std::vector< double > xvals(p.geti("numVals"));
        readData(infile, p.geti("numVals"), &data[0], &xvals[0]);
        
        double variance = variaceCalc(data, modParams, paramMins, paramMaxs, limitParams,
                                      xvals, p.geti("numVals"), p.geti("numParams"), Psi,
                                      detPsi);
        
        
        
