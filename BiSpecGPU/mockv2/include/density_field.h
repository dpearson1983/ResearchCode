#ifndef _DENSITY_FIELD_H_
#define _DENSITY_FIELD_H_

#include <vector>
#include <string>
#include <gsl/gsl_integration.h>
#include "cosmology.h"
#include "galaxy.h"
#include "tpods.h"

class densityField{
    std::vector<double> den;
    vec3<double> L, r_min, pk_nbw, bk_nbw, dr;
    vec3<size_t> N;
    
    void nearest_grid_point(vec3<double> pos, std::vector<size_t> &index, std::vector<double> &weight);
    
    void cloud_in_cell(vec3<double> pos, std::vector<size_t> &index, std::vector<double> &weight);
    
    public:
        densityField();
        
        densityField(vec3<double> Len, vec3<int> Num, vec3<double> rmin);
        
        void initialize(vec3<double> Len, vec3<int> Num, vec3<double> rmin);
        
        void bin(galaxy gal, cosmology cos, gsl_integration_workspace *w_gsl, std::string method = "CIC");
        
        double at(size_t index);
        
        double at(size_t i, size_t j, size_t k);
        
        double nbw();
        
        double nbw2();
        
        double nb2w2();
        
        double nb2w3();
        
        double nb3w3();
        
        double nbw3();

};

#endif
