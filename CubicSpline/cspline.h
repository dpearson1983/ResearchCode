#ifndef _CSPLINE_H_
#define _CSPLINE_H_

#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <vector_types.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>

template <typename T> class cspline{
    std::vector<T> y;
    std::vector<T> t;
    std::vector<T> h;
    std::vector<T> z;
    int N;
    
    void h_calc();
    void b_calc(std::vector<T> &b);
    void v_calc(std::vector<T> &v);
    void u_calc(std::vector<T> &u, std::vector<T> &b);
    void z_calc(std::vector<T> &v, std::vector<T> &u);
    
    public:
        cspline();
        
        cspline(std::vector<T> &x, std::vector<T> &fx);
        
        void initialize(std::vector<T> &x, std::vector<T> &fx);
        
        void set_pointer_for_device(std::vector<float4> &spline);
        
        T evaluate(T x);
        
};

template<typename T> void cspline<T>::h_calc() {
    for (int i = 0; i < cspline<T>::N - 1; ++i)
        cspline<T>::h.push_back(t[i + 1] - t[i]);
}

template<typename T> void cspline<T>::b_calc(std::vector<T> &b) {
    for (int i = 0; i < cspline<T>::N - 1; ++i)
        b[i] = (cspline<T>::y[i + 1] - cspline<T>::y[i])/h[i];
}

template<typename T> void cspline<T>::v_calc(std::vector<T> &v) {
    for (int i = 1; i < cspline<T>::N - 1; ++i)
        v[i - 1] = 2.0*(cspline<T>::h[i - 1] + cspline<T>::h[i]);
}

template<typename T> void cspline<T>::u_calc(std::vector<T> &u, std::vector<T> &b) {
    for (int i = 1; i < cspline<T>::N - 1; ++i)
        u[i - 1] = 6.0*(b[i] - b[i - 1]);
}

template<typename T> void cspline<T>::z_calc(std::vector<T> &v, std::vector<T> &u) {
    gsl_matrix *A = gsl_matrix_alloc(cspline<T>::N - 2, cspline<T>::N - 2);
    gsl_vector *Z = gsl_vector_alloc(cspline<T>::N - 2);
    gsl_vector *U = gsl_vector_alloc(cspline<T>::N - 2);
    
    for (int i = 1; i < cspline<T>::N - 1; ++i) {
        gsl_vector_set(U, i - 1, u[i - 1]);
        for (int j = 1; j < cspline<T>::N - 1; ++j) {
            if (i == j) gsl_matrix_set(A, i - 1, j - 1, v[i - 1]);
            else if (j == i + 1) gsl_matrix_set(A, i - 1, j - 1, cspline<T>::h[i]);
            else if (j == i - 1) gsl_matrix_set(A, i - 1, j - 1, cspline<T>::h[j]);
            else gsl_matrix_set(A, i - 1, j - 1, 0.0);
        }
    }
    
    int s;
    gsl_permutation *p = gsl_permutation_alloc(cspline<T>::N - 2);
    gsl_linalg_LU_decomp(A, p, &s);
    gsl_linalg_LU_solve(A, p, U, Z);
    
    for (int i = 0; i < cspline<T>::N; ++i) {
        if (i == 0 || i == cspline<T>::N - 1) cspline<T>::z.push_back(0.0);
        else cspline<T>::z.push_back(gsl_vector_get(Z, i - 1));
    }
    
    gsl_vector_free(Z);
    gsl_vector_free(U);
    gsl_matrix_free(A);
    gsl_permutation_free(p);
}

template<typename T> cspline<T>::cspline() {
    cspline<T>::N = 1;
}

template<typename T> cspline<T>::cspline(std::vector<T> &x, std::vector<T> &fx) {
    cspline<T>::N = fx.size();
    
    for (int i = 0; i < cspline<T>::N; ++i) {
        cspline<T>::t.push_back(x[i]);
        cspline<T>::y.push_back(fx[i]);
    }
    
    std::vector<T> b(cspline<T>::N - 1);
    std::vector<T> v(cspline<T>::N - 2);
    std::vector<T> u(cspline<T>::N - 2);
    
    cspline<T>::h_calc();
    cspline<T>::b_calc(b);
    cspline<T>::v_calc(v);
    cspline<T>::u_calc(u, b);
    cspline<T>::z_calc(v, u);
}

template<typename T> void cspline<T>::initialize(std::vector<T> &x, std::vector<T> &fx) {
    cspline<T>::N = fx.size();
    
    for (int i = 0; i < cspline<T>::N; ++i) {
        cspline<T>::t.push_back(x[i]);
        cspline<T>::y.push_back(fx[i]);
    }
    
    std::vector<T> b(cspline<T>::N - 1);
    std::vector<T> v(cspline<T>::N - 2);
    std::vector<T> u(cspline<T>::N - 2);
    
    cspline<T>::h_calc();
    cspline<T>::b_calc(b);
    cspline<T>::v_calc(v);
    cspline<T>::u_calc(u, b);
    cspline<T>::z_calc(v, u);
}

template<typename T> void cspline<T>::set_pointer_for_device(std::vector<float4> &spline) {
    for (int i = 0; i < cspline<T>::N; ++i) {
        float4 temp = {cspline<T>::t[i], cspline<T>::y[i], cspline<T>::z[i], cspline<T>::h[i]};
        spline.push_back(temp);
    }
}

template<typename T> T cspline<T>::evaluate(T x) {
    if (x < cspline<T>::t[0] || x > cspline<T>::t[cspline<T>::N - 1]) {
        std::stringstream message;
        message << "The requested x value is outside of the allowed range." << std::endl;
        throw std::runtime_error(message.str());
    }
    
    int i = 0;
    while (cspline<T>::t[i] < x) ++i;
    i--;
    
    T val = (cspline<T>::z[i + 1]*(x - cspline<T>::t[i])*(x - cspline<T>::t[i])*(x - cspline<T>::t[i]))/(6.0*cspline<T>::h[i])
             + (cspline<T>::z[i]*(cspline<T>::t[i + 1] - x)*(cspline<T>::t[i + 1] - x)*(cspline<T>::t[i + 1] - x))/(6.0*cspline<T>::h[i])
             + (cspline<T>::y[i + 1]/cspline<T>::h[i] - (cspline<T>::z[i + 1]*cspline<T>::h[i])/6.0)*(x - cspline<T>::t[i])
             + (cspline<T>::y[i]/cspline<T>::h[i] - (cspline<T>::h[i]*cspline<T>::z[i])/6.0)*(cspline<T>::t[i + 1] - x);
             
    return val;
}

#endif
