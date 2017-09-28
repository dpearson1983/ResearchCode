#ifndef _PK_SLOPE_H_
#define _PK_SLOPE_H_

#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <vector_types.h>
#include <gsl/gsl_deriv.h>
#include "cspline.h"
#include "file_check.h"

template <typename T> class pkSlope{
    int N;
    std::vector<T> k, Pk, n;
    cspline<T> slope;
    
    void get_pk_spline();
    
    void calculate_n();
    
    void get_n_spline();
    
    public:
        cspline<T> power;
        
        pkSlope();
        
        pkSlope(std::string in_pk_file);
        
        void initialize(std::string in_pk_file);
        
        void calculate();
        
        void get_device_spline(std::vector<float4> &d_spline);
        
        void get_device_spline(std::vector<double4> &d_spline);
        
};

template <typename T> double f(double x, void *params) {
    cspline<T> Pow = *(cspline<T> *)params;
    return Pow.evaluate(x);
}

template <typename T> void pkSlope<T>::get_pk_spline() {
    pkSlope<T>::power.initialize(pkSlope<T>::k, pkSlope<T>::Pk);
}

template <typename T> void pkSlope<T>::calculate_n() {
    gsl_function F;
    double result, err;
    
    F.function = &f<T>;
    F.params = &power;
    
    for (int i = 0; i < pkSlope<T>::N; ++i) {
        if (i == 0) {
            gsl_deriv_forward(&F, pkSlope<T>::k[i], 1e-8, &result, &err);
        } else if (i == pkSlope<T>::N - 1) {
            gsl_deriv_backward(&F, pkSlope<T>::k[i], 1e-8, &result, &err);
        } else {
            gsl_deriv_central(&F, pkSlope<T>::k[i], 1e-8, &result, &err);
        }
        pkSlope<T>::n.push_back(result);
    }
}

template <typename T> void pkSlope<T>::get_n_spline() {
    pkSlope<T>::slope.initialize(pkSlope<T>::k, pkSlope<T>::n);
}

template <typename T> pkSlope<T>::pkSlope() {
    pkSlope<T>::N = 0;
}

template <typename T> pkSlope<T>::pkSlope(std::string in_pk_file) {
    pkSlope<T>::initialize(in_pk_file);
}
    
template <typename T> void pkSlope<T>::initialize(std::string in_pk_file) {
    if (check_file_exists(in_pk_file)) {
        std::ifstream fin(in_pk_file);
        while(!fin.eof()) {
            T kt, pt;
            fin >> kt >> pt;
            if (!fin.eof()) {
                pkSlope<T>::k.push_back(log10(kt));
                pkSlope<T>::Pk.push_back(log10(pt));
            }
        }
        fin.close();
        
        pkSlope<T>::N = pkSlope<T>::k.size();
    }
}

template <typename T> void pkSlope<T>::calculate() {
    pkSlope<T>::get_pk_spline();
    pkSlope<T>::calculate_n();
    pkSlope<T>::get_n_spline();
}

template <typename T> void pkSlope<T>::get_device_spline(std::vector<float4> &d_spline) {
    pkSlope<T>::slope.set_pointer_for_device(d_spline);
}

template <typename T> void pkSlope<T>::get_device_spline(std::vector<double4> &d_spline) {
    pkSlope<T>::slope.set_pointer_for_device(d_spline);
}

#endif
