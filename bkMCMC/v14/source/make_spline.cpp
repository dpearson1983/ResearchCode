#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cspline.h>
#include "../include/make_spline.h"
#include "/usr/local/cuda-8.0/include/vector_types.h"

std::vector<float4> make_spline(std::string in_pk_file) {
    std::ifstream fin;
    
    std::vector<double> kin;
    std::vector<double> pin;
    
    fin.open(in_pk_file.c_str(), std::ios::in);
    while (!fin.eof()) {
        double kt, pt;
        fin >> kt >> pt;
        if (!fin.eof()) {
            kin.push_back(kt);
            pin.push_back(pt);
        }
    }
    fin.close();
    
    cspline<double> Pk_spline(kin, pin);
    
    std::cout << "Setting up the spline..." << std::endl;
    std::vector<float4> Pk;
    Pk_spline.set_pointer_for_device(Pk);

    return Pk;
}
