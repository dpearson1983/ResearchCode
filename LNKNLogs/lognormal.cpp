#include <cmath>
#include <fstream>
#include <cstring>
#include <sstream>
#include <cstdlib>
#include <random>
#include <fftw.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>
#include "pods.h"
#include "lognormal.h"

const long double pi = acos(-1);

void fftfreq1d(double *k, int N, double L) {
    double dk = 2.0*pi/L;
    
    for (int i = 0; i <= N/2; ++i) {
        k[i] = i*dk;
    }
    for (int i = N/2 + 1; i < N; ++i) {
        k[i] = (i-N)*dk;
    }
}

void fftfreq2d(double2 *k, int2 N, double2 L) {
    double dkx = 2.0*pi/L.x;
    double dky = 2.0*pi/L.y;
    
    for (int i = 0; i < N.x; ++i) {
        if (i <= N.x/2) {k[i].x = i*dkx;}
        else {k[i].x = (i-N.x)*dkx;}
    }
    
    for (int i = 0; i < N.y; ++i) {
        if (i <= N.y/2) {k[i].y = i*dky;}
        else {k[i].y = (i-N.y)*dky;}
    }
    
}

void fftfreq3d(double3 *k, int3 N, double3 L) {
    double dkx = 2.0*pi/L.x;
    double dky = 2.0*pi/L.y;
    double dkz = 2.0*pi/L.z;
    
    for (int i = 0; i < N.x; ++i) {
        if (i <= N.x/2) {k[i].x = i*dkx;}
        else {k[i].x = (i-N.x)*dkx;}
    }
    
    for (int i = 0; i < N.y; ++i) {
        if (i <= N.y/2) {k[i].y = i*dky;}
        else {k[i].y = (i-N.y)*dky;}
    }
    
    for (int i = 0; i < N.z; ++i) {
        if (i <= N.z/2) {k[i].z = i*dkz;}
        else {k[i].z = (i-N.z)*dkz;}
    }
    
}

void Pk_CAMB(int3 N, double3 *vk, double *k_mag, double *P, int numKVals, double *Pk) {
    gsl_spline *Power = gsl_spline_alloc(gsl_interp_cspline, numKVals);
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    
    gsl_spline_init(Power, k_mag, P, numKVals);
    
    for (int i = 0; i < N.x; ++i) {
        
        for (int j = 0; j < N.y; ++j) {
            
            for (int k = 0; k <= N.z/2; ++k) {
                
                int index = k + (N.z/2 + 1)*(j + N.y*i);
                double k_tot = sqrt(vk[i].x*vk[i].x + vk[j].y*vk[j].y + vk[k].z*vk[k].z);
                
                if (k_tot > 0) {
                    Pk[index] = gsl_spline_eval(Power, k_tot, acc);
                } else {
                    Pk[index] = 0.0;
                }
            }
        }
    }
    
    gsl_spline_free(Power);
    gsl_interp_accel_free(acc);
}

void Genddk(int3 N, double3 L, double3 *vk, double b, double f, double *Pk,
            fftw_complex *dk3d) {
    double V = L.x*L.y*L.z;
    // Loop through the different combinations of frequency components
    for (int i = 0; i < N.x; ++i) {        
        
        for (int j = 0; j < N.y; ++j) {
            
            for (int k = 0; k <= N.z/2; ++k) {
                int index = k + (N.z/2 + 1)*(j+N.y*i);
                
                double k_tot = sqrt(vk[i].x*vk[i].x+vk[j].y*vk[j].y+vk[k].z*vk[k].z); // Magnitude of frequency
                if (k_tot != 0) {
                    double mu = vk[i].x/k_tot; // Anisotropy factor
                    double power = (b + mu*mu*f)*(b + mu*mu*f)*Pk[index]; // Calculate the power
                    
                    dk3d[index].re = power/V;
                    dk3d[index].im = 0.0;
                }
                else {
                    dk3d[index].re = 0.0;
                    dk3d[index].im = 0.0;
                }
            }
        }
    }
}

void Genddk_2Tracer(int3 N, double3 L, double3 *vk, double2 b, double f, double *Pk,
                    fftw_complex *dk3d1, fftw_complex *dk3d2) {
    double V = L.x*L.y*L.z;
    // Loop through the different combinations of frequency components
    for (int i = 0; i < N.x; ++i) {        
        
        for (int j = 0; j < N.y; ++j) {
            
            for (int k = 0; k <= N.z/2; ++k) {
                int index = k + (N.z/2 + 1)*(j+N.y*i);
                
                double k_tot = sqrt(vk[i].x*vk[i].x+vk[j].y*vk[j].y+vk[k].z*vk[k].z); // Magnitude of frequency
                if (k_tot != 0) {
                    double mu = vk[i].x/k_tot; // Anisotropy factor
                    double power1 = (b.x + mu*mu*f)*(b.x + mu*mu*f)*Pk[index]; // Calculate the power
                    double power2 = (b.y + mu*mu*f)*(b.y + mu*mu*f)*Pk[index];
                    
                    dk3d1[index].re = power1/V;
                    dk3d1[index].im = 0.0;
                    
                    dk3d2[index].re = power2/V;
                    dk3d2[index].im = 0.0;
                }
                else {
                    dk3d1[index].re = 0.0;
                    dk3d1[index].im = 0.0;
                    
                    dk3d2[index].re = 0.0;
                    dk3d2[index].im = 0.0;
                }
            }
        }
    }
}

void Sampdk(int3 N, double3 *vk, fftw_complex *dk3di, fftw_complex *dk3d) {
    std::mt19937_64 generator; // Mersenne twister random number generator for real
    generator.seed(time(0)); // Seed the random number generator
    // Loop through the different combinations of frequency components
    for (int i = 0; i < N.x; ++i) {
        int i2 = (2*N.x - i) % N.x;
        
        for (int j = 0; j < N.y; ++j) {
            int j2 = (2*N.y - j) % N.y;
            
            for (int k = 0; k <= N.z/2; ++k) {
                int index1 = k + (N.z/2 + 1)*(j+N.y*i);
                int index2 = k + (N.z/2 + 1)*(j2+N.y*i2);
                
                double k_tot = sqrt(vk[i].x*vk[i].x+vk[j].y*vk[j].y+vk[k].z*vk[k].z); // Magnitude of frequency
                //if (k_tot == 0) continue;
                
                //double Power = sqrt(deltak3di[index].re*deltak3di[index].re+deltak3di[index].im*deltak3di[index].im);
                double Power = dk3di[index1].re;

                if (Power > 0) {
                    if ((i == 0 || i == N.x/2) && (j == 0 || j == N.y/2) && (k == 0 || k == N.z/2)){
                        std::normal_distribution<double> distribution(0.0,sqrt(Power));
                        
                        dk3d[index1].re = distribution(generator);
                        dk3d[index1].im = 0.0;
                    } else if (k == 0 || k == N.z/2) {
                        std::normal_distribution<double> distribution(0.0,sqrt(Power/2.0));
                        
                        double dkre = distribution(generator);
                        double dkim = distribution(generator);
                        
                        dk3d[index1].re = dkre;
                        dk3d[index1].im = dkim;
                        
                        dk3d[index2].re = dkre;
                        dk3d[index2].im = -dkim;
                    } else {
                        std::normal_distribution<double> distribution(0.0,sqrt(Power/2.0));
                        
                        dk3d[index1].re = distribution(generator);
                        dk3d[index1].im = distribution(generator);
                    }
                }
                else {
                    dk3d[index1].re = 0.0;
                    dk3d[index1].im = 0.0;
                }
            }
        }
    }
}

void Sampdk_2Tracer(int3 N, double3 *vk, fftw_complex *dk3di, fftw_complex *dk3d1,
                    fftw_complex *dk3d2, double *ratio) {
    std::mt19937_64 generator; // Mersenne twister random number generator for real
    generator.seed(time(0)); // Seed the random number generator
    // Loop through the different combinations of frequency components
    for (int i = 0; i < N.x; ++i) {
        int i2 = (2*N.x - i) % N.x;
        
        for (int j = 0; j < N.y; ++j) {
            int j2 = (2*N.y - j) % N.y;
            
            for (int k = 0; k <= N.z/2; ++k) {
                int index1 = k + (N.z/2 + 1)*(j+N.y*i);
                int index2 = k + (N.z/2 + 1)*(j2+N.y*i2);
                
                double k_tot = sqrt(vk[i].x*vk[i].x+vk[j].y*vk[j].y+vk[k].z*vk[k].z); // Magnitude of frequency
                double Power = dk3di[index1].re;

                if (Power > 0) {
                    if ((i == 0 || i == N.x/2) && (j == 0 || j == N.y/2) && (k == 0 || k == N.z/2)) {
                        double variance = sqrt(Power); 
                        std::normal_distribution<double> distribution(0.0,variance);
                        
                        double deltare = distribution(generator);
                        
                        dk3d1[index1].re = deltare;
                        dk3d1[index1].im = 0.0;
                        
                        if (ratio[index1] > 0) {
                            dk3d2[index1].re = sqrt(ratio[index1])*deltare;
                            dk3d2[index1].im = 0.0;
                        } else {
                            dk3d2[index1].re = 0.0;
                            dk3d2[index1].im = 0.0;
                        }
                    } else if (k == 0 || k == N.z/2) {
                        double variance = sqrt(Power/2.0); 
                        std::normal_distribution<double> distribution(0.0,variance);
                        
                        double deltare = distribution(generator);
                        double deltaim = distribution(generator);
                        
                        dk3d1[index1].re = deltare;
                        dk3d1[index1].im = deltaim;
                        
                        dk3d1[index2].re = deltare;
                        dk3d1[index2].im = -deltaim;
                        
                        if (ratio[index1] > 0) {
                            dk3d2[index1].re = sqrt(ratio[index1])*deltare;
                            dk3d2[index1].im = sqrt(ratio[index1])*deltaim;
                            
                            dk3d2[index2].re = sqrt(ratio[index1])*deltare;
                            dk3d2[index2].im = -sqrt(ratio[index1])*deltaim;
                        } else {
                            dk3d2[index1].re = 0.0;
                            dk3d2[index1].im = 0.0;
                            
                            dk3d2[index2].re = 0.0;
                            dk3d2[index2].im = 0.0;
                        }
                    } else {
                        double variance = sqrt(Power/2.0);
                        std::normal_distribution<double> distribution(0.0,variance);
                        
                        double deltare = distribution(generator);
                        double deltaim = distribution(generator);
                        
                        dk3d1[index1].re = deltare;
                        dk3d1[index1].im = deltaim;
                        
                        if (ratio[index1] > 0) {
                            dk3d2[index1].re = sqrt(ratio[index1])*deltare;
                            dk3d2[index1].im = sqrt(ratio[index1])*deltaim;
                        } else {
                            dk3d2[index1].re = 0.0;
                            dk3d2[index1].im = 0.0;
                        }
                    }
                }
                else {
                    dk3d1[index1].re = 0.0;
                    dk3d1[index1].im = 0.0;
                    
                    dk3d2[index1].re = 0.0;
                    dk3d2[index1].im = 0.0;
                }
            }
        }
    }
}

void Genddr(int3 N, double3 L, double nbar, std::string file, double variance,
            fftw_real *dr3d) {
    std::mt19937_64 generator;
    generator.seed(time(0));
    
    std::ofstream fout; // Output filestream
    
    double3 dL = {L.x/N.x, L.y/N.y, L.z/N.z};
    double n = nbar*dL.x*dL.y*dL.z;
    
    fout.open(file.c_str(),std::ios::out); // Open output file
    fout.precision(15); // Set the number of digits to output
    
    // Loop through all the grid points, assign galaxies uniform random positions within
    // each cell.
    for (int i = 0; i < N.x; ++i) {
        double xmin = i*dL.x;
        double xmax = (i+1)*dL.x;
        
        // Uniform random distribution for the x coordinate
        std::uniform_real_distribution<double> xpos(xmin, xmax);
        for (int j = 0; j < N.y; ++j) {
            double ymin = j*dL.y;
            double ymax = (j+1)*dL.y;
            
            // Uniform random distribution for the y coordinate
            std::uniform_real_distribution<double> ypos(ymin, ymax);
            for (int k = 0; k < N.z; ++k) {
                double zmin = k*dL.z;
                double zmax = (k+1)*dL.z;
                
                // Uniform random distribution for the z coordinate
                std::uniform_real_distribution<double> zpos(zmin, zmax);
                
                int index = k + N.z*(j+N.y*i); // Calculate grid index
                
                // Initialize the Poisson distribution with the value of the matter field
                double density = n*exp(dr3d[index]-variance/2);
                std::poisson_distribution<int> distribution(density);
                int numGal = distribution(generator); // Randomly Poisson sample
                
                // Randomly generate positions for numGal galaxies within the cell
                for (int gal = 0; gal < numGal; ++gal) {
                    fout << xpos(generator) << " " << ypos(generator) << " " << zpos(generator) << "\n";
                }
            }
        }
    }
    fout.close(); // Close file
}

void Genddr_2Tracer(int3 N, double3 L, double2 nbar, string2 file, double2 variance,
                    fftw_real *dr3d1, fftw_real *dr3d2) {
    // Create Mersenne twister random number generators for the Poisson sampling, and
    // to distribute galaxies in a uniform random manner within the cells.
    std::mt19937_64 generator;
    // Seed the generators
    generator.seed(time(0));
    
    std::ofstream fout; // Output filestream
    std::ofstream fout2;
    
    double3 dL = {L.x/N.x, L.y/N.y, L.z/N.z};
    double n1 = nbar.x*dL.x*dL.y*dL.z;
    double n2 = nbar.y*dL.x*dL.y*dL.z;
    
    fout.open(file.x.c_str(),std::ios::out); // Open output file
    fout.precision(15); // Set the number of digits to output
    
    fout2.open(file.y.c_str(),std::ios::out);
    fout2.precision(15);
    
    // Loop through all the grid points, assign galaxies uniform random positions within
    // each cell.
    for (int i = 0; i < N.x; ++i) {
        double xmin = i*dL.x;
        double xmax = (i+1)*dL.x;
        
        // Uniform random distribution for the x coordinate
        std::uniform_real_distribution<double> xpos(xmin, xmax);
        for (int j = 0; j < N.y; ++j) {
            double ymin = j*dL.y;
            double ymax = (j+1)*dL.y;
            
            // Uniform random distribution for the y coordinate
            std::uniform_real_distribution<double> ypos(ymin, ymax);
            for (int k = 0; k < N.z; ++k) {
                double zmin = k*dL.z;
                double zmax = (k+1)*dL.z;
                
                // Uniform random distribution for the z coordinate
                std::uniform_real_distribution<double> zpos(zmin, zmax);
                
                int index = k + N.z*(j+N.y*i); // Calculate grid index
                
                // Check that the matter density field is positive. Set to zero if not.
                //if (deltar3d[index] < 0.0) continue;
                
                // Initialize the Poisson distribution with the value of the matter field
                double density1 = n1*exp(dr3d1[index]-variance.x/2);
                double density2 = n2*exp(dr3d2[index]-variance.y/2);
                std::poisson_distribution<int> distribution(density1);
                std::poisson_distribution<int> distribution2(density2);
                int numGal = distribution(generator); // Randomly Poisson sample
                int numGal2 = distribution2(generator);
                
                // Randomly generate positions for numGal galaxies within the cell
                for (int gal = 0; gal < numGal; ++gal) {
                    fout << xpos(generator) << " " << ypos(generator) << " " << zpos(generator) << "\n";
                }
                
                for (int gal = 0; gal < numGal2; ++gal) {
                    fout2 << xpos(generator) << " " << ypos(generator) << " " << zpos(generator) << "\n";
                }
            }
        }
    }
    fout.close(); // Close file
    fout2.close();
}