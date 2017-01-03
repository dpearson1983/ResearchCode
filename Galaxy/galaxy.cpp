/* galaxy.cpp v1.0
 * December 29, 2016
 * 
 * LICENSE: GPL v3
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>. 
 */

#include <iostream>
#include <cmath>
#include <constants.h>
#include "galaxy.h"

struct intparams{
    double O_m, O_L;
};

// This defines the function to be integrated by GSL when locating the distance to a
// galaxy.
double f(double x, void *params) {
    intparams p = *(intparams *)params;
    double ff = c/(100.0*sqrt(p.O_m*(1.0+x)*(1.0+x)*(1.0+x)+p.O_L));
    return ff;
}

// Below is the function which will take a redshift in convert it into a distance.
// This will be needed when calculating the three dimensional positions of the
// galaxy and random catalogs.
double rz(double red, double O_m, double O_L, gsl_integration_workspace *w) {
    double error;
    double D;
    intparams p;
    p.O_m = O_m;
    p.O_L = O_L;
    gsl_function F;
    F.function = &f;
    F.params = &p;
    gsl_integration_qags(&F, 0.0, red, 1e-7, 1e-7, 10000000, w, &D, &error);
    
    return D;
}

double r2z(double r, double O_m, double O_L, double tolerance, gsl_integration_workspace *w) {
    double z = 100.0*r/c;
    double D = rz(z, O_m, O_L, w);
    while(D < r-tolerance || D > r+tolerance) {
        double dz = 100.0*(r - D)/c;
        z += dz;
        D = rz(z, O_m, O_L, w);
    }
    
    return z;
}

template <typename T> galaxy<T>::galaxy() {
    galaxy<T>::ra = 0.0;
    galaxy<T>::dec = 0.0;
    galaxy<T>::red = 0.0;
    galaxy<T>::x = 0.0;
    galaxy<T>::y = 0.0;
    galaxy<T>::z = 0.0;
    galaxy<T>::nbar = 0.0;
    galaxy<T>::bias = 0.0;
    galaxy<T>::w = 0.0;
}

template <typename T> galaxy<T>::galaxy(T RA, T DEC, T RED, T X, T Y, T Z, T NBAR, T BIAS, T W) {
    galaxy<T>::ra = RA;
    galaxy<T>::dec = DEC;
    galaxy<T>::red = RED;
    galaxy<T>::x = X;
    galaxy<T>::y = Y;
    galaxy<T>::z = Z;
    galaxy<T>::nbar = NBAR;
    galaxy<T>::bias = BIAS;
    galaxy<T>::w = W;
}

template <typename T> T galaxy<T>::wFKP(T P_FKP) {
    return 1.0/(1.0 + galaxy<T>::nbar*P_FKP);
}

template <typename T> T galaxy<T>::wPVP(T P_PVP) {
    return galaxy<T>::bias*galaxy<T>::bias*P_PVP/(1.0 + galaxy<T>::nbar*P_PVP);
}

template <typename T> T galaxy<T>::wPSG(T P_PSG) {
    return 1.0;
}

template <typename T> void galaxy<T>::cartesian(double Omega_M, double Omega_L, gsl_integration_workspace *w) {
    double r = rz(galaxy::red, Omega_M, Omega_L, w);
    galaxy<T>::x = r*cos(galaxy<T>::dec*pi/180.0)*cos(galaxy<T>::ra*pi/180.0);
    galaxy<T>::y = r*cos(galaxy<T>::dec*pi/180.0)*cos(galaxy<T>::ra*pi/180.0);
    galaxy<T>::z = r*sin(galaxy<T>::dec*pi/180.0);
}

template <typename T> void galaxy<T>::equatorial(double Omega_M, double Omega_L, gsl_integration_workspace *w) {
    double r_mag = sqrt(galaxy<T>::x*galaxy<T>::x + galaxy<T>::y*galaxy<T>::y + galaxy<T>::z*galaxy<T>::z);
    galaxy<T>::ra = atan2(galaxy<T>::y, galaxy<T>::x)*(180.0*pi);
    if (galaxy<T>::ra < 0) galaxy<T>::ra += 360.0;
    galaxy<T>::dec = atan(galaxy<T>::z/sqrt(galaxy<T>::x*galaxy<T>::x + galaxy<T>::y*galaxy<T>::y))*(180.0*pi);
    galaxy<T>::red = r2z(r_mag, Omega_M, Omega_L, 1E-12, w);
}

template <typename T> void galaxy<T>::bin(double *nden, vec3<double> L, vec3<int> N, vec3<double> r_min, vec3<double> &gal_nbw, double P_w, int flags) {
    vec3<double> dr = {L.x/double(N.x), L.y/double(N.y), L.z/double(N.z)};
    
    double w;
    if (flags & galFlags::FKP_WEIGHT) {
        w = galaxy<T>::wFKP(P_w);
        galaxy<T>::w = w;
    } else if (flags & galFlags::PVP_WEIGHT) {
        w = galaxy<T>::PVP(P_w);
        galaxy<T>::w = w;
    } else if (flags & galFlags::PSG_WEIGHT) {
        w = galaxy<T>::PSG(P_w);
        galaxy<T>::w = w;
    } else if (flags & galFlags::UNWEIGHTED) {
        w = 1.0;
        galaxy<T>::w = w;
    } else if (flags & galFlags::INPUT_WEIGHT) {
        w = galaxy<T>::w;
    }
    
    gal_nbw.x += w;
    gal_nbw.y += w*w;
    gal_nbw.z += galaxy<T>::nbar*w*w;
    
    if (flags & galFlags::NGP) {
        vec3<int> ngp = {(galaxy<T>::x - r_min.x)/dr.x, (galaxy<T>::y - r_min.y)/dr.y, (galaxy<T>::z - r_min.z)/dr.z};
        int index = ngp.z + N.z*(ngp.y + N.y*ngp.x);
        nden[index] += w;
    } else if (flags & galFlags::CIC) {
        vec3<int> ngp = {(galaxy<T>::x - r_min.x)/dr.x, (galaxy<T>::y - r_min.y)/dr.y, (galaxy<T>::z - r_min.z)/dr.z};
        vec3<double> pos_ngp = {(ngp.x + 0.5)*dr.x + r_min.x, (ngp.y + 0.5)*dr.y + r_min.y, 
            (ngp.z + 0.5)*dr.z + r_min.z};
        vec3<double> delr = {galaxy<T>::x-pos_ngp.x, galaxy<T>::y-pos_ngp.y, galaxy<T>::z-pos_ngp.z};
        vec3<int> shift = {delr.x/fabs(delr.x), delr.y/fabs(delr.y), delr.z/fabs(delr.z)};
        
        int pn[2] = {-1, 1};
        double V_tot = dr.x*dr.y*dr.z;
        for (int i = 0; i < 2; ++i) {
            double Lx = (1-i)*dr.x + pn[i]*fabs(delr.x);
            for (int j = 0; j < 2; ++j) {
                double Ly = (1-j)*dr.y + pn[j]*fabs(delr.y);
                for (int k = 0; k < 2; ++k) {
                    double Lz = (1-k)*dr.z + pn[k]*fabs(delr.z);
                    
                    double V = Lx*Ly*Lz;
                    long int index = (ngp.z + k*shift.z) + N.z*((ngp.y + j*shift.y) + N.y*(ngp.x + i*shift.x));
                    nden[index] += V/V_tot;
                }
            }
        }
    }
    
}

template <typename T> void galaxy<T>::rMax(vec3<double> &r_max) {
    if (galaxy<T>::x > r_max.x) r_max.x = galaxy<T>::x;
    if (galaxy<T>::y > r_max.y) r_max.y = galaxy<T>::y;
    if (galaxy<T>::z > r_max.z) r_max.z = galaxy<T>::z;
}

template <typename T> void galaxy<T>::rMin(vec3<double> &r_min) {
    if (galaxy<T>::x < r_min.x) r_min.x = galaxy<T>::x;
    if (galaxy<T>::y < r_min.y) r_min.y = galaxy<T>::y;
    if (galaxy<T>::z < r_min.z) r_min.z = galaxy<T>::z;
}
