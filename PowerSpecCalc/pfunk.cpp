#include <pods.h>
#include <fftw3.h>
#include <fstream>
#include <cmath>
#include <string>
#include <omp.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>

const double c = 299792.458;

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

void initArray(double *array, long int N) {
    for (int i = 0; i < N; ++i) {
        array[i] = 0.0;
    }
}

void initArray(fftw_complex *array, long int N) {
    for (int i = 0; i < N; ++i) {
        array[i][0] = 0.0;
        array[i][1] = 0.0;
    }
}

void initArray(double *dr, fftw_complex *dk, long int N_r, long int N_k) {
    for (int i = 0; i < N_r; ++i) {
        dr[i] = 0.0;
        if (i < N_k) {
            dk[i][0] = 0.0;
            dk[i][1] = 0.0;
        }
    }
}

void fftfreq(double *kvec, int N, double L) {
    double dk = (2.0*pi)/L;
    for (int i = 0; i <= N/2; ++i)
        kvec[i] = i*dk;
    for (int i = N/2 + 1; i < N; ++i)
        kvec[i] = (i - N)*dk;
}

void equatorial2cartesian(galaxy *gals, int N, double O_m, double O_L, gsl_integration_workspace *w) {
    gsl_function F;
    intparams p;
    p.O_m = O_m;
    p.O_L = O_L;
    F.function = &f;
    F.params = &p;
    for (int i = 0; i < N; ++i) {
        double r, error;
        gsl_integration_qags(&F, 0.0, gals[i].red, 1e-7, 1e-7, 10000000, w, &r, &error);;
        gals[i].x = r*cos(gals[i].dec*pi/180.0)*cos(gals[i].ra*pi/180.0);
        gals[i].y = r*cos(gals[i].dec*pi/180.0)*sin(gals[i].ra*pi/180.0);
        gals[i].z = r*sin(gals[i].dec*pi/180.0);
    }
}

void equatorial2cartesian(galaxyf *gals, double3 *pos, int N, double O_m, double O_L, gsl_integration_workspace *w) {
    gsl_function F;
    intparams p;
    p.O_m = O_m;
    p.O_L = O_L;
    F.function = &f;
    F.params = &p;
    for (int i = 0; i < N; ++i) {
        double r, error;
        gsl_integration_qags(&F, 0.0, gals[i].red, 1e-7, 1e-7, 10000000, w, &r, &error);;
        pos[i].x = r*cos(gals[i].dec*pi/180.0)*cos(gals[i].ra*pi/180.0);
        pos[i].y = r*cos(gals[i].dec*pi/180.0)*sin(gals[i].ra*pi/180.0);
        pos[i].z = r*sin(gals[i].dec*pi/180.0);
    }
}

void cartesian2equatorial(galaxy *gals, int N, double O_m, double O_L, gsl_integration_workspace *w) {
    for (int i = 0; i < N; ++i) {
        double r = sqrt(gals[i].x*gals[i].x + gals[i].y*gals[i].y + gals[i].z*gals[i].z);
        gals[i].ra = atan2(gals[i].y, gals[i].x)*(180.0/pi);
        if (gals[i].ra < 0) gals[i].ra += 360.0;
        gals[i].dec = atan2(gals[i].z, sqrt(gals[i].x*gals[i].x + gals[i].y*gals[i].y))*(180.0/pi);
        gals[i].red = r2z(r, O_m, O_L, 1E-12, w);
    }
}

double binNGP(galaxy *gals, double *nden, double3 dr, int numGals, int3 N) {
    double x_min = 1000000.0, y_min = 1000000.0, z_min = 1000000.0;
    double x_max = -1000000.0, y_max = -1000000.0, z_max = -1000000.0;
    for (int i = 0; i < numGals; ++i) {
        if (gals[i].x < x_min) x_min = gals[i].x;
        if (gals[i].y < y_min) y_min = gals[i].y;
        if (gals[i].z < z_min) z_min = gals[i].z;
        if (gals[i].x > x_max) x_max = gals[i].x;
        if (gals[i].y > y_max) y_max = gals[i].y;
        if (gals[i].z > z_max) z_max = gals[i].z;
        
        int3 ngp = {gals[i].x/dr.x, gals[i].y/dr.y, gals[i].z/dr.z};
        long int index = ngp.z + N.z*(ngp.y + N.y*ngp.x);
        
        nden[index] += 1.0;
    }
    return (x_max - x_min)*(y_max - y_min)*(z_max - z_min);
}

double binNGP(galaxy *gals, double *nden, double3 dr, int numGals, int3 N, double3 r_obs) {
    double RA_min = 1000000.0, Dec_min = 1000000.0, r_min = 1000000.0;
    double RA_max = -1000000.0, Dec_max = -1000000.0, r_max = -1000000.0;
    for (int i = 0; i < numGals; ++i) {
        double3 r = {gals[i].x-r_obs.x, gals[i].y-r_obs.y, gals[i].z-r_obs.z};
        double r_mag = sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
        double RA = atan2(r.y, r.x);
        double DEC = atan(r.z/sqrt(r.x*r.x + r.y*r.y));
        
        if (RA_min > RA) RA_min = RA;
        if (Dec_min > DEC) Dec_min = DEC;
        if (r_min > r_mag) r_min = r_mag;
        if (RA_max < RA) RA_max = RA;
        if (Dec_max < DEC) Dec_max = DEC;
        if (r_max < r_mag) r_max = r_mag;
        
        int3 ngp = {gals[i].x/dr.x, gals[i].y/dr.y, gals[i].z/dr.z};
        long int index = ngp.z + N.z*(ngp.y + N.y*ngp.x);
        
        nden[index] += 1.0;
    }
    double V = ((RA_max-RA_min)*(r_max*r_max*r_max - r_min*r_min*r_min)*(sin(Dec_max)-sin(Dec_min)))/3.0;
    return V;
}

double binCIC(galaxy *gals, double *nden, double3 dr, int numGals, int3 N) {
    double x_min = 1000000.0, y_min = 1000000.0, z_min = 1000000.0;
    double x_max = -1000000.0, y_max = -1000000.0, z_max = -1000000.0;
    for (int i = 0; i < numGals; ++i) {
        if (gals[i].x < x_min) x_min = gals[i].x;
        if (gals[i].y < y_min) y_min = gals[i].y;
        if (gals[i].z < z_min) z_min = gals[i].z;
        if (gals[i].x > x_max) x_max = gals[i].x;
        if (gals[i].y > y_max) y_max = gals[i].y;
        if (gals[i].z > z_max) z_max = gals[i].z;
        
        int3 ngp = {(gals[i].x)/dr.x, (gals[i].y)/dr.y, (gals[i].z)/dr.z};
        double3 pos_ngp = {(ngp.x + 0.5)*dr.x, (ngp.y + 0.5)*dr.y, (ngp.z + 0.5)*dr.z};
        double3 delr = {gals[i].x-pos_ngp.x, gals[i].y-pos_ngp.y, gals[i].z-pos_ngp.z};
        int3 shift = {delr.x/fabs(delr.x), delr.y/fabs(delr.y), delr.z/fabs(delr.z)};
        
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
    return (x_max - x_min)*(y_max - y_min)*(z_max - z_min);
}

double binCIC(galaxy *gals, double *nden, double3 dr, int numGals, int3 N, double3 r_obs) {
    double RA_min = 1000000.0, Dec_min = 1000000.0, r_min = 1000000.0;
    double RA_max = -1000000.0, Dec_max = -1000000.0, r_max = -1000000.0;
    for (int i = 0; i < numGals; ++i) {
        double3 r = {gals[i].x-r_obs.x, gals[i].y-r_obs.y, gals[i].z-r_obs.z};
        double r_mag = sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
        double RA = atan2(r.y, r.x);
        double DEC = atan(r.z/sqrt(r.x*r.x + r.y*r.y));
        
        if (RA_min > RA) RA_min = RA;
        if (Dec_min > DEC) Dec_min = DEC;
        if (r_min > r_mag) r_min = r_mag;
        if (RA_max < RA) RA_max = RA;
        if (Dec_max < DEC) Dec_max = DEC;
        if (r_max < r_mag) r_max = r_mag;
        
        int3 ngp = {(gals[i].x)/dr.x, (gals[i].y)/dr.y, (gals[i].z)/dr.z};
        double3 pos_ngp = {(ngp.x + 0.5)*dr.x, (ngp.y + 0.5)*dr.y, (ngp.z + 0.5)*dr.z};
        double3 delr = {gals[i].x-pos_ngp.x, gals[i].y-pos_ngp.y, gals[i].z-pos_ngp.z};
        int3 shift = {delr.x/fabs(delr.x), delr.y/fabs(delr.y), delr.z/fabs(delr.z)};
        
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
    double V = ((RA_max-RA_min)*(r_max*r_max*r_max - r_min*r_min*r_min)*(sin(Dec_max)-sin(Dec_min)))/3.0;
    return V;
}

double3 wbinCIC(galaxy *gals, double *nden, double3 dr, int numGals, int3 N, double3 rmin, double *redshift, double *nbar, int nVals, double Pk_FKP) {
    gsl_spline *nb_of_z = gsl_spline_alloc(gsl_interp_cspline, nVals);
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline_init(nb_of_z, redshift, nbar, nVals);
    double3 gal_nbw = {0.0, 0.0, 0.0};
    for (int i = 0; i < numGals; ++i) {
        gals[i].nbar = gsl_spline_eval(nb_of_z, gals[i].red, acc);
        double w = 1.0/(1.0 + gals[i].nbar*Pk_FKP);
        int3 ngp = {(gals[i].x-rmin.x)/dr.x, (gals[i].y-rmin.y)/dr.y, (gals[i].z-rmin.z)/dr.z};
        double3 pos_ngp = {(ngp.x + 0.5)*dr.x, (ngp.y + 0.5)*dr.y, (ngp.z + 0.5)*dr.z};
        double3 delr = {gals[i].x-pos_ngp.x, gals[i].y-pos_ngp.y, gals[i].z-pos_ngp.z};
        int3 shift = {delr.x/fabs(delr.x), delr.y/fabs(delr.y), delr.z/fabs(delr.z)};
        
        gal_nbw.x += w;
        gal_nbw.y += w*w;
        gal_nbw.z += gals[i].nbar*w*w;
        
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
                    nden[index] += (V/V_tot)*w;
                }
            }
        }
    }
    gsl_spline_free(nb_of_z);
    gsl_interp_accel_free(acc);
    return gal_nbw;
}

double3 wbinCIC(galaxyf *gals, double *nden, double3 dr, int numGals, int3 N, double3 rmin, double *redshift, double *nbar, int nVals, double Pk_FKP, double Omega_M, double Omega_L) {
    gsl_spline *nb_of_z = gsl_spline_alloc(gsl_interp_cspline, nVals);
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline_init(nb_of_z, redshift, nbar, nVals);
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(10000000);
    double3 gal_nbw = {0.0, 0.0, 0.0};
    for (int i = 0; i < numGals; ++i) {
        double3 pos;
        equatorial2cartesian(&gals[i], &pos, 1, Omega_M, Omega_L, w);
        gals[i].nbar = gsl_spline_eval(nb_of_z, gals[i].red, acc);
        double w = 1.0/(1.0 + gals[i].nbar*Pk_FKP);
        int3 ngp = {(pos.x-rmin.x)/dr.x, (pos.y-rmin.y)/dr.y, (pos.z-rmin.z)/dr.z};
        double3 pos_ngp = {rmin.x + (ngp.x + 0.5)*dr.x, rmin.y + (ngp.y + 0.5)*dr.y, rmin.z + (ngp.z + 0.5)*dr.z};
        double3 delr = {pos.x-pos_ngp.x, pos.y-pos_ngp.y, pos.z-pos_ngp.z};
        int3 shift = {delr.x/fabs(delr.x), delr.y/fabs(delr.y), delr.z/fabs(delr.z)};
        
        gal_nbw.x += w;
        gal_nbw.y += w*w;
        gal_nbw.z += gals[i].nbar*w*w;
        
        delr.x = fabs(delr.x);
        delr.y = fabs(delr.y);
        delr.z = fabs(delr.z);
        
        double V = 1.0/(dr.x*dr.y*dr.z);
        double V1 = delr.x*delr.y*delr.z*V;
        double V2 = (dr.x - delr.x)*delr.y*delr.z*V;
        double V3 = delr.x*(dr.y - delr.y)*delr.z*V;
        double V4 = delr.y*delr.y*(dr.z - delr.z)*V;
        double V5 = (dr.x - delr.x)*(dr.y - delr.y)*delr.z*V;
        double V6 = (dr.x - delr.x)*delr.y*(dr.z - delr.z)*V;
        double V7 = delr.x*(dr.y - delr.y)*(dr.z - delr.z)*V;
        double V8 = (dr.x - delr.x)*(dr.y - delr.y)*(dr.z - delr.z)*V;
        
        int index = ngp.z + N.z*(ngp.y + N.y*ngp.x);
        nden[index] += V8*w;
        nden[index + shift.x*N.z*N.y] += V7*w;
        nden[index + shift.y*N.z] += V6*w;
        nden[index + shift.z] += V5*w;
        nden[index + shift.y*N.z + shift.x*N.z*N.y] += V4*w;
        nden[index + shift.z + shift.x*N.z*N.y] += V3*w;
        nden[index + shift.z + shift.y*N.z] += V2*w;
        nden[index + shift.z + shift.y*N.z + shift.x*N.z*N.y] += V1*w;
    }
    gsl_spline_free(nb_of_z);
    gsl_interp_accel_free(acc);
    gsl_integration_workspace_free(w);
    return gal_nbw;
}

double3 wbinNGP(galaxyf *gals, double *nden, double3 dr, int numGals, int3 N, double3 rmin, double *redshift, double *nbar, int nVals, double Pk_FKP, double Omega_M, double Omega_L) {
    gsl_spline *nb_of_z = gsl_spline_alloc(gsl_interp_cspline, nVals);
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline_init(nb_of_z, redshift, nbar, nVals);
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(10000000);
    double3 gal_nbw = {0.0, 0.0, 0.0};
    for (int i = 0; i < numGals; ++i) {
        double3 pos;
        equatorial2cartesian(&gals[i], &pos, 1, Omega_M, Omega_L, w);
        gals[i].nbar = gsl_spline_eval(nb_of_z, gals[i].red, acc);
        double w = 1.0/(1.0 + gals[i].nbar*Pk_FKP);
        int3 ngp = {(pos.x-rmin.x)/dr.x, (pos.y-rmin.y)/dr.y, (pos.z-rmin.z)/dr.z};
        
        gal_nbw.x += w;
        gal_nbw.y += w*w;
        gal_nbw.z += gals[i].nbar*w*w;
        
        int index = ngp.z + N.z*(ngp.y + N.y*ngp.x);
        nden[index] += w;
    }
    gsl_spline_free(nb_of_z);
    gsl_interp_accel_free(acc);
    gsl_integration_workspace_free(w);
    return gal_nbw;
}

double gridCorNGP(double kx, double ky, double kz, double3 dr) {
    double sincx = sin(0.5*kx*dr.x + 1E-17)/(0.5*kx*dr.x + 1E-17);
    double sincy = sin(0.5*ky*dr.y + 1E-17)/(0.5*ky*dr.y + 1E-17);
    double sincz = sin(0.5*kz*dr.z + 1E-17)/(0.5*kz*dr.z + 1E-17);
    double prodsinc = sincx*sincy*sincz;
    return 1.0/prodsinc;
}
    
double gridCorCIC(double kx, double ky, double kz, double3 dr) {
    double sincx = sin(0.5*kx*dr.x + 1E-17)/(0.5*kx*dr.x + 1E-17);
    double sincy = sin(0.5*ky*dr.y + 1E-17)/(0.5*ky*dr.y + 1E-17);
    double sincz = sin(0.5*kz*dr.z + 1E-17)/(0.5*kz*dr.z + 1E-17);
    double prodsinc = sincx*sincy*sincz;
    return 1.0/(prodsinc*prodsinc);
}    

void binFreqPP(fftw_complex dk, double *P_0, double *P_2, double *P_2shot, int bin, double mu, 
               double grid_cor, double shotnoise) {
    P_0[bin] += (dk[0]*dk[0] + dk[1]*dk[1] - shotnoise)*grid_cor*grid_cor;
    P_2[bin] += 2.5*(3.0*mu*mu - 1.0)*(dk[0]*dk[0] + dk[1]*dk[1])*grid_cor*grid_cor;
    P_2shot[bin] += 2.5*(3.0*mu*mu - 1.0)*shotnoise*grid_cor*grid_cor;
}
    

void freqBinPP(fftw_complex *dk3d, double *P_0, double *P_2, double *P_2shot, int *N_k, int3 N, 
               double3 L, double shotnoise, double k_min, double k_max, int kBins, bool corr, 
               int type) {
    double *kx = new double[N.x];
    double *ky = new double[N.y];
    double *kz = new double[N.z];
    fftfreq(kx, N.x, L.x);
    fftfreq(ky, N.y, L.y);
    fftfreq(kz, N.z, L.z);
    double binWidth = (k_max - k_min)/double(kBins);
    double3 dr = {L.x/double(N.x), L.y/double(N.y), L.z/double(N.z)};
    
    int3 Nstop = {ceil(k_max/kx[1]), ceil(k_max/ky[1]), ceil(k_max/kz[1])};
    for (int i = 0; i < Nstop.x; ++i) {
        int i2 = N.x - i;
        for (int j = 0; j < Nstop.y; ++j) {
            int j2 = N.y - j;
            for (int k = 0; k < Nstop.z; ++k) {
                double k_tot = sqrt(kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k]);
                
                if (k_tot >= k_min && k_tot <= k_max) {
                    int bin = (k_tot-k_min)/binWidth;
                    double mu =  0.0;
                    if (k_tot > 0) mu = kx[i]/k_tot;
                    
                    double grid_cor = 1.0;
                    if (corr) {
                        if (type == 0) grid_cor = gridCorNGP(kx[i], ky[j], kz[k], dr);
                        if (type == 1) grid_cor = gridCorCIC(kx[i], ky[j], kz[k], dr);
                    }
                    
                    if (i != 0 && j != 0) {
                        int index1 = k + (N.z/2 + 1)*(j  + N.y*i );
                        int index2 = k + (N.z/2 + 1)*(j2 + N.y*i );
                        int index3 = k + (N.z/2 + 1)*(j  + N.y*i2);
                        int index4 = k + (N.z/2 + 1)*(j2 + N.y*i2);
                        binFreqPP(dk3d[index1], P_0, P_2, P_2shot, bin, mu, grid_cor, shotnoise);
                        binFreqPP(dk3d[index2], P_0, P_2, P_2shot, bin, mu, grid_cor, shotnoise);
                        binFreqPP(dk3d[index3], P_0, P_2, P_2shot, bin, mu, grid_cor, shotnoise);
                        binFreqPP(dk3d[index4], P_0, P_2, P_2shot, bin, mu, grid_cor, shotnoise);
                        N_k[bin] += 4;
                    } else if (i != 0 && j == 0) {
                        int index1 = k + (N.z/2 + 1)*(j + N.y*i );
                        int index2 = k + (N.z/2 + 1)*(j + N.y*i2);
                        binFreqPP(dk3d[index1], P_0, P_2, P_2shot, bin, mu, grid_cor, shotnoise);
                        binFreqPP(dk3d[index2], P_0, P_2, P_2shot, bin, mu, grid_cor, shotnoise);
                        N_k[bin] += 2;
                    } else if (i == 0 && j != 0) {
                        int index1 = k + (N.z/2 + 1)*(j  + N.y*i);
                        int index2 = k + (N.z/2 + 1)*(j2 + N.y*i);
                        binFreqPP(dk3d[index1], P_0, P_2, P_2shot, bin, mu, grid_cor, shotnoise);
                        binFreqPP(dk3d[index2], P_0, P_2, P_2shot, bin, mu, grid_cor, shotnoise);
                        N_k[bin] += 2;
                    } else {
                        binFreqPP(dk3d[k], P_0, P_2, P_2shot, bin, mu, grid_cor, shotnoise);
                        N_k[bin] += 1;
                    }
                }
            }
        }
    }
    delete[] kx;
    delete[] ky;
    delete[] kz;
}

void normalizePk(double *P_0, double *P_2, double *P_2shot, int *N_k, double gal_nbsqwsq, int N) {
    for (int i = 0; i < N; ++i) {
        P_0[i] /= N_k[i];
        P_0[i] /= gal_nbsqwsq;
        P_2[i] -= P_2shot[i];
        P_2[i] /= N_k[i];
        P_2[i] /= gal_nbsqwsq;
    }
}

void correct_discreteness(std::string cor_file, double *P_0, double *P_2, int kbins) {
    std::ifstream fin;
    
    fin.open(cor_file.c_str(), std::ios::in);
    for (int i = 0; i < kbins; ++i) {
        double cor;
        fin >> cor;
        P_0[i] -= cor;
    }
    for (int i = 0; i < kbins; ++i) {
        double cor;
        fin >> cor;
        P_2[i] -= cor;
    }
    fin.close();
}

void calcB_ij(double *B_ij, double *nden_gal, double *nden_ran, double alpha, double3 dr,
               double3 rmin, double3 robs, int3 N, std::string ij) {
    double4 r;
    for (int i = 0; i < N.x; ++i) {
        r.x = (rmin.x + (i + 0.5)*dr.x) - robs.x;
        for (int j = 0; j < N.y; ++j) {
            r.y = (rmin.y + (j + 0.5)*dr.y) - robs.y;
            for (int k = 0; k < N.z; ++k) {
                r.z = (rmin.z + (k + 0.5)*dr.z) - robs.z;
                
                r.w = 1.0/(r.x*r.x + r.y*r.y + r.z*r.z);
                double coeff;
                if (r.w > 0) {
                    if (ij == "xx") coeff = r.x*r.x*r.w;
                    else if (ij == "yy") coeff = r.y*r.y*r.w;
                    else if (ij == "zz") coeff = r.z*r.z*r.w;
                    else if (ij == "xy" || ij == "yx") coeff = r.x*r.y*r.w;
                    else if (ij == "xz" || ij == "zx") coeff = r.x*r.z*r.w;
                    else if (ij == "yz" || ij == "zy") coeff = r.z*r.y*r.w;
                } else {
                    coeff = 0.0;
                }
                
                long int index = k + N.z*(j + N.y*i);
                B_ij[index] = coeff*(nden_gal[index] - alpha*nden_ran[index]);
            }
        }
    }
}

void accumulateA_2(fftw_complex *A_2, fftw_complex *B_ij, int3 N, double3 L, std::string ij) {
    double *kx = new double[N.x];
    double *ky = new double[N.y];
    double *kz = new double[N.z];
    fftfreq(kx, N.x, L.x);
    fftfreq(ky, N.y, L.y);
    fftfreq(kz, N.z, L.z);
    
    for (int i = 0; i < N.x; ++i) {
        for (int j = 0; j < N.y; ++j) {
            for (int k = 0; k <= N.z/2; ++k) {
                double k_tot = kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k];
                
                double coeff;
                if (k_tot > 0 ) {
                    if (ij == "xx") coeff = (kx[i]*kx[i])/k_tot;
                    else if (ij == "yy") coeff = (ky[j]*ky[j])/k_tot;
                    else if (ij == "zz") coeff = (kz[k]*kz[k])/k_tot;
                    else if (ij == "xy" || ij == "yx") coeff = (2.0*kx[i]*ky[j])/k_tot;
                    else if (ij == "xz" || ij == "zx") coeff = (2.0*kx[i]*kz[k])/k_tot;
                    else if (ij == "yz" || ij == "zy") coeff = (2.0*ky[j]*kz[k])/k_tot;
                } else {
                    coeff = 0;
                }
                
                long int index = k + (N.z/2 + 1)*(j + N.y*i);
                A_2[index][0] += coeff*B_ij[index][0];
                A_2[index][1] += coeff*B_ij[index][1];
            }
        }
    }
    
    delete[] kx;
    delete[] ky;
    delete[] kz;
}

void binFreqBS(fftw_complex A_0, fftw_complex A_2, double *P_0, double *P_2, int bin, 
               double grid_cor, double shotnoise) {
    P_0[bin] += (A_0[0]*A_0[0] + A_0[1]*A_0[1] - shotnoise)*grid_cor*grid_cor;
    P_2[bin] += 2.5*(3.0*(A_2[0]*A_0[0] + A_2[1]*A_0[1]) - (A_0[0]*A_0[0] + A_0[1]*A_0[1]))*grid_cor*grid_cor;
}

void freqBinBS(fftw_complex *A_0, fftw_complex *A_2, double *P_0, double *P_2, int *N_k, int3 N,
               double3 L, double shotnoise, double kmin, double kmax, int kBins, bool corr, 
               int type) {
    double *kx = new double[N.x];
    double *ky = new double[N.y];
    double *kz = new double[N.z];
    fftfreq(kx, N.x, L.x);
    fftfreq(ky, N.y, L.y);
    fftfreq(kz, N.z, L.z);
    double binWidth = (kmax - kmin)/double(kBins);
    double3 dr = {L.x/double(N.x), L.y/double(N.y), L.z/double(N.z)};
    
    int3 Nstop = {ceil(kmax/kx[1]), ceil(kmax/ky[1]), ceil(kmax/kz[1])};
    
    for (int i = 0; i < Nstop.x; ++i) {
        int i2 = N.x - i;
        for (int j = 0; j < Nstop.y; ++j) {
            int j2 = N.y - j;
            for (int k = 0; k < Nstop.z; ++k) {
                double k_tot = sqrt(kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k]);
                
                if (k_tot >= kmin && k_tot <= kmax) {
                    int bin = (k_tot-kmin)/binWidth;
                    
                    double grid_cor = 1.0;
                    if (corr) {
                        if (type == 0) grid_cor = gridCorNGP(kx[i], ky[j], kz[k], dr);
                        if (type == 1) grid_cor = gridCorCIC(kx[i], ky[j], kz[k], dr);
                    }
                    
                    if (i != 0 && j != 0) {
                        int index1 = k + (N.z/2 + 1)*(j  + N.y*i );
                        int index2 = k + (N.z/2 + 1)*(j2 + N.y*i );
                        int index3 = k + (N.z/2 + 1)*(j  + N.y*i2);
                        int index4 = k + (N.z/2 + 1)*(j2 + N.y*i2);
                        binFreqBS(A_0[index1], A_2[index1], P_0, P_2, bin, grid_cor, shotnoise);
                        binFreqBS(A_0[index2], A_2[index2], P_0, P_2, bin, grid_cor, shotnoise);
                        binFreqBS(A_0[index3], A_2[index3], P_0, P_2, bin, grid_cor, shotnoise);
                        binFreqBS(A_0[index4], A_2[index4], P_0, P_2, bin, grid_cor, shotnoise);
                        N_k[bin] += 4;
                    } else if (i != 0 && j == 0) {
                        int index1 = k + (N.z/2 + 1)*(j + N.y*i );
                        int index2 = k + (N.z/2 + 1)*(j + N.y*i2);
                        binFreqBS(A_0[index1], A_2[index1], P_0, P_2, bin, grid_cor, shotnoise);
                        binFreqBS(A_0[index2], A_2[index2], P_0, P_2, bin, grid_cor, shotnoise);
                        N_k[bin] += 2;
                    } else if (i == 0 && j != 0) {
                        int index1 = k + (N.z/2 + 1)*(j  + N.y*i);
                        int index2 = k + (N.z/2 + 1)*(j2 + N.y*i);
                        binFreqBS(A_0[index1], A_2[index1], P_0, P_2, bin, grid_cor, shotnoise);
                        binFreqBS(A_0[index2], A_2[index2], P_0, P_2, bin, grid_cor, shotnoise);
                        N_k[bin] += 2;
                    } else {
                        binFreqBS(A_0[k], A_2[k], P_0, P_2, bin, grid_cor, shotnoise);
                        N_k[bin] += 1;
                    }
                }
            }
        }
    }
    
    delete[] kx;
    delete[] ky;
    delete[] kz;
}

void normalizePk(double *P_0, double *P_2, int *N_k, double gal_nbsqwsq, int N) {
    for (int i = 0; i < N; ++i) {
        P_0[i] /= N_k[i];
        P_0[i] /= gal_nbsqwsq;
        P_2[i] /= N_k[i];
        P_2[i] /= gal_nbsqwsq;
    }
}
