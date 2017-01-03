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

double galaxy::wFKP(double P_FKP) {
    return 1.0/(1.0 + galaxy::nbar*P_FKP);
}

double galaxy::wPVP(double P_PVP) {
    return galaxy::bias*galaxy::bias*P_PVP/(1.0 + galaxy::nbar*P_PVP);
}

double galaxy::wPSG(double P_PSG) {
    return 1.0;
}

void galaxy::cartesian(double Omega_M, double Omega_L, gsl_integration_workspace *w) {
    double r = rz(galaxy::red, Omega_M, Omega_L, w);
    galaxy::x = r*cos(galaxy::dec*pi/180.0)*cos(galaxy::ra*pi/180.0);
    galaxy::y = r*cos(galaxy::dec*pi/180.0)*cos(galaxy::ra*pi/180.0);
    galaxy::z = r*sin(galaxy::dec*pi/180.0);
}

void galaxy::equatorial(double Omega_M, double Omega_L, gsl_integration_workspace *w) {
    double r_mag = sqrt(galaxy::x*galaxy::x + galaxy::y*galaxy::y + galaxy::z*galaxy::z);
    galaxy::ra = atan2(galaxy::y, galaxy::x)*(180.0*pi);
    if (galaxy::ra < 0) galaxy::ra += 360.0;
    galaxy::dec = atan(galaxy::z/sqrt(galaxy::x*galaxy::x + galaxy::y*galaxy::y))*(180.0*pi);
    galaxy::red = r2z(r_mag, Omega_M, Omega_L, 1E-12, w);
}

vec3<double> galaxy::bin(double *nden, vec3<double> L, vec3<int> N, vec3<double> r_min, double P_w, int flags) {
    vec3<double> dr = {L.x/double(N.x), L.y/double(N.y), L.z/double(N.z)};
    vec3<double> gal_nbw = {0.0, 0.0, 0.0};
    
    double w;
    if (flags & galFlags::FKP_WEIGHT) {
        w = galaxy::wFKP(P_w);
        galaxy::w = w;
    } else if (flags & galFlags::PVP_WEIGHT) {
        w = galaxy::PVP(P_w);
        galaxy::w = w;
    } else if (flags & galFlags::PSG_WEIGHT) {
        w = galaxy::PSG(P_w);
        galaxy::w = w;
    } else if (flags & galFlags::UNWEIGHTED) {
        w = 1.0;
        galaxy::w = w;
    } else if (flags & galFlags::INPUT_WIEGHT) {
        w = galaxy::w;
    }
    
    gal_nbw.x += w;
    gal_nbw.y += w*w;
    gal_nbw.z += galaxy::nbar*w*w;
    
    if (flags & galFlags::NGP) {
        vec3<int> ngp = {(galaxy::x - r_min.x)/dr.x, (galaxy::y - r_min.y)/dr.y, (galaxy::z - r_min.z)/dr.z};
        int index = ngp.z + N.z*(ngp.y + N.y*ngp.x);
        nden[index] += w;
    } else if (flags & galFlags::CIC) {
        vec3<int> ngp = {(galaxy::x - r_min.x)/dr.x, (galaxy::y - r_min.y)/dr.y, (galaxy::z - r_min.z)/dr.z};
        vec3<double> pos_ngp = {(ngp.x + 0.5)*dr.x + r_min.x, (ngp.y + 0.5)*dr.y + r_min.y, 
            (ngp.z + 0.5)*dr.z + r_min.z};
        vec3<double> delr = {gals[i].x-pos_ngp.x, gals[i].y-pos_ngp.y, gals[i].z-pos_ngp.z};
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
    
    return gal_nbw;
}

vec3<double> galaxy::rMax(vec3<double> r_max) {
    if (galaxy::x > r_max.x) r_max.x = galaxy::x;
    if (galaxy::y > r_max.y) r_max.y = galaxy::y;
    if (galaxy::z > r_max.z) r_max.z = galaxy::z;
    return r_max;
}

vec3<double> galaxy::rMin(vec3<double> r_min) {
    if (galaxy::x < r_min.x) r_min.x = galaxy::x;
    if (galaxy::y < r_min.y) r_min.y = galaxy::y;
    if (galaxy::z < r_min.z) r_min.z = galaxy::z;
    return r_min;
}
