#include <vector>
#include <cmath>
#include <gsl/gsl_integration.h>
#include "../include/cosmology.h"
#include "../include/galaxy.h"
#include "../include/tpods.h"
#include "../include/density_field.h"

void densityField::nearest_grid_point(vec3<double> pos, std::vector<size_t> &index, 
                                      std::vector<double> &weight) {
    vec3<size_t> ngp = {size_t((pos.x - densityField::r_min.x)/densityField::dr.x),
                        size_t((pos.y - densityField::r_min.x)/densityField::dr.x),
                        size_t((pos.z - densityField::r_min.x)/densityField::dr.x)};
    index.push_back(ngp.z + densityField::N.z*(ngp.y + densityField::N.y*ngp.x));
    weight.push_back(1.0);
}

void densityField::cloud_in_cell(vec3<double> pos, std::vector<size_t> &index, 
                                 std::vector<double> &weight) {
    vec3<size_t> ngp = {size_t((pos.x - densityField::r_min.x)/densityField::dr.x),
                        size_t((pos.y - densityField::r_min.x)/densityField::dr.x),
                        size_t((pos.z - densityField::r_min.x)/densityField::dr.x)};
    vec3<double> r_ngp = {densityField::r_min.x + (ngp.x + 0.5)*densityField::dr.x,
                          densityField::r_min.y + (ngp.y + 0.5)*densityField::dr.y,
                          densityField::r_min.z + (ngp.z + 0.5)*densityField::dr.z};
    vec3<double> delr = {pos.x - r_ngp.x, pos.y - r_ngp.y, pos.z - r_ngp.z};
    vec3<int> shift = {int(delr.x/fabs(delr.x)), int(delr.y/fabs(delr.y)), int(delr.z/fabs(delr.z))};
    delr.x = fabs(delr.x);
    delr.y = fabs(delr.y);
    delr.z - fabs(delr.z);
    
    index.push_back(ngp.z + densityField::N.z*(ngp.y + densityField::N.y*ngp.x));
    index.push_back(ngp.z + densityField::N.z*(ngp.y + densityField::N.y*(ngp.x + shift.x)));
    index.push_back(ngp.z + densityField::N.z*((ngp.y + shift.y) + densityField::N.y*ngp.x));
    index.push_back((ngp.z + shift.z) + densityField::N.z*(ngp.y + densityField::N.y*ngp.x));
    index.push_back(ngp.z + densityField::N.z*((ngp.y + shift.y) + densityField::N.y*(ngp.x + shift.x)));
    index.push_back((ngp.z + shift.z) + densityField::N.z*(ngp.y + densityField::N.y*(ngp.x + shift.x)));
    index.push_back((ngp.z + shift.z) + densityField::N.z*((ngp.y + shift.y) + densityField::N.y*ngp.x));
    index.push_back((ngp.z + shift.z) + densityField::N.z*((ngp.y + shift.y) + 
                                                            densityField::N.y*(ngp.x + shift.x)));
    
    double V_inv = 1.0/(densityField::dr.x*densityField::dr.y*densityField::dr.z);
    weight.push_back(((densityField::dr.x - delr.x)*(densityField::dr.y - delr.y)*(densityField::dr.z - delr.z))*V_inv);
    weight.push_back((delr.x*(densityField::dr.y - delr.y)*(densityField::dr.z - delr.z))*V_inv);
    weight.push_back(((densityField::dr.x - delr.x)*delr.y*(densityField::dr.z - delr.z))*V_inv);
    weight.push_back(((densityField::dr.x - delr.x)*(densityField::dr.y - delr.y)*delr.z)*V_inv);
    weight.push_back((delr.x*delr.y*(densityField::dr.z - delr.z))*V_inv);
    weight.push_back((delr.x*(densityField::dr.y - delr.y)*delr.z)*V_inv);
    weight.push_back(((densityField::dr.x - delr.x)*delr.y*delr.z)*V_inv);
    weight.push_back(delr.x*delr.y*delr.z*V_inv);
}

densityField::densityField() {
    densityField::N.x = 1;
    densityField::N.y = 1;
    densityField::N.z = 1;
    
    densityField::r_min.x = 0.0;
    densityField::r_min.y = 0.0;
    densityField::r_min.z = 0.0;
}

densityField::densityField(vec3<double> Len, vec3<int> Num, vec3<double> rmin) {
    densityField::initialize(Len, Num, rmin);
}

void densityField::initialize(vec3<double> Len, vec3<int> Num, vec3<double> rmin) {
    densityField::L.x = Len.x;
    densityField::L.y = Len.y;
    densityField::L.z = Len.z;
    
    densityField::N.x = Num.x;
    densityField::N.y = Num.y;
    densityField::N.z = Num.z;
    
    densityField::r_min.x = rmin.x;
    densityField::r_min.y = rmin.y;
    densityField::r_min.z = rmin.z;
    
    densityField::pk_nbw.x = 0.0;
    densityField::pk_nbw.y = 0.0;
    densityField::pk_nbw.z = 0.0;
    
    densityField::bk_nbw.x = 0.0;
    densityField::bk_nbw.y = 0.0;
    densityField::bk_nbw.z = 0.0;
    
    densityField::dr.x = densityField::L.x/double(densityField::N.x);
    densityField::dr.y = densityField::L.y/double(densityField::N.y);
    densityField::dr.z = densityField::L.z/double(densityField::N.z);
    
    densityField::den.reserve(Num.x*Num.y*Num.z);
    for (size_t i = 0; i < Num.x*Num.y*Num.z; ++i)
        den.push_back(0.0);
}

void densityField::bin(galaxy gal, cosmology cos, gsl_integration_workspace *w_gsl, std::string method) {
    vec3<double> pos = gal.cartesian(cos, w_gsl);
    std::vector<size_t> index;
    std::vector<double> weight;
    if (method == "NGP") {
        densityField::nearest_grid_point(pos, index, weight);
    } else if (method == "CIC") {
        densityField::cloud_in_cell(pos, index, weight);
    }
    
    size_t N = index.size();
    for (size_t i = 0; i < N; ++i) {
        den[index[i]] += weight[i]*gal.W(galFlags::INPUT_WEIGHT);
    }
    
    densityField::pk_nbw.x += gal.W(galFlags::INPUT_WEIGHT);
    densityField::pk_nbw.y += gal.W(galFlags::INPUT_WEIGHT)*gal.W(galFlags::INPUT_WEIGHT);
    densityField::pk_nbw.z += gal.N()*gal.W(galFlags::INPUT_WEIGHT)*gal.W(galFlags::INPUT_WEIGHT);
    
    densityField::bk_nbw.x += gal.W(galFlags::INPUT_WEIGHT)*gal.W(galFlags::INPUT_WEIGHT)*gal.W(galFlags::INPUT_WEIGHT);
    densityField::bk_nbw.y += gal.N()*gal.W(galFlags::INPUT_WEIGHT)*gal.W(galFlags::INPUT_WEIGHT)*gal.W(galFlags::INPUT_WEIGHT);
    densityField::bk_nbw.z += gal.N()*gal.N()*gal.W(galFlags::INPUT_WEIGHT)*gal.W(galFlags::INPUT_WEIGHT)*gal.W(galFlags::INPUT_WEIGHT);
}

double densityField::at(size_t index) {
    return densityField::den[index];
}

double densityField::at(size_t i, size_t j, size_t k) {
    int index = k + densityField::N.z*(j + densityField::N.y*i);
    return densityField::den[index];
}

double densityField::nbw() {
    return pk_nbw.x;
}

double densityField::nbw2() {
    return pk_nbw.y;
}

double densityField::nb2w2() {
    return pk_nbw.z;
}

double densityField::nb2w3() {
    return bk_nbw.y;
}

double densityField::nb3w3() {
    return bk_nbw.z;
}

double densityField::nbw3() {
    return bk_nbw.x;
}
