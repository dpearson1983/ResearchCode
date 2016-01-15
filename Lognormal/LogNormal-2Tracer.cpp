/* LogNormal-2Tracer.cpp
 * David W. Pearson & Lado Samushia
 * 
 * The purpose of this code is to generate mock galaxy catalogs with two different tracers,
 * each having their own relative bias and growth with respect to the input matter power
 * spectrum.
 * 
 * This code is based on LogNormal.cpp which created a single tracer mock. Large portions
 * of the code are based on a Python script written by Lado Samushia which aimed to generate
 * simple Gaussian random mocks.
 * 
 * This code is structured as follows:
 * 
 * Compile with:
 * g++ -std=c++11 -lgsl -lgslcblas -lfftw -lrfftw -lfftw_threads -lrfftw_threads -lm -fopenmp -march=native -mtune=native -O2 LogNormal-2Tracer.cpp -o LogNormal-2Tracer
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <rfftw_threads.h>
#include <omp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>

const int numMocks = 1024; // Number of mocks to generate
const int startNum = 1;

const double pi = 3.14159265359; // Mmmmmmm, Pi!

const std::string base = "LRG"; // Base of the filename for the mocks
const std::string base2 = "ELG";
const std::string matbase = "matterfield"; // Base of the filename for the matter field
const std::string powbase = "powerfield"; // Base of the filename for the power field
const std::string ext = ".dat"; // Extension to use for creating plain text files
const std::string extbin = ".bin"; // Extension to use for creating binary files
const std::string CAMBfile = "camb_95039962_matterpower_z0.57.dat"; // Input power spectrum

const bool matOut = false; // Control whether you want to output the matter field
const bool powOut = false; // Control whether you want to output the raw power array

const int N = 512; // Number of grid points in one dimension.
const long int N_tot = N*N*N; // Total number of elements in arrays for FFTs
const long int N_im = N*N*(N/2 + 1); // Number of imaginary arrray elements
const double A = 80000.0; // Amplitude of the input power spectrum
const double mean = 0.01; // Location of the peak of the input power spectrum
const double sig = 0.15; // Width of the input power spectrum
// Numbers of galaxies per mock
const int num_gal1 = 500000;
const int num_gal2 = 286913;
const double L = 1024.0; // Size of the cube
const double dL = L/N; // Size of a grid cell
const double V = L*L*L; // Volume of the cube
// Number densities
const double nbar1 = num_gal1/V;
const double nbar2 = num_gal2/V;

const double b1 = 2.0; // Bias of tracer 1
const double b2 = 1.0; // Bias of tracer 2
const double f = 0.537244398679; // Growth parameter

// The following variable controls the number of processor cores used by FFTW and parallel
// sections of the code. By using omp_get_max_threads(), both the FFTW and parallel sections
// will use all of the cores of your processor to achieve maximum performance. If you wish to 
// not monopolize your computing resources to run this program, this number can be reduced to
// any integer value.
const int numCores = omp_get_max_threads();

struct Pk {
    double k, P;
};

double Pk_CAMB(gsl_spline *Pow, gsl_interp_accel *a, double k) {
    double power = gsl_spline_eval(Pow, k, a);
    
    return power;
}

// This function distributes the input power to the grid. 
void Gendk(double *kvec, gsl_spline *Pow, gsl_interp_accel *a, fftw_complex *deltak3d1, fftw_complex *deltak3d2) {
    
    // Loop through the different combinations of frequency components
    for (int i = 0; i < N; ++i) {
        
        double kx = 2*pi*kvec[i]; // x frequency
        
        for (int j = 0; j < N; ++j) {
            
            double ky = 2*pi*kvec[j]; // y frequency
            
            for (int k = 0; k <= N/2; ++k) {
                
                int index = k + (N/2 + 1)*(j+N*i); // Calculate the associate array index
                
                double kz = 2*pi*kvec[k]; // z frequency
                
                double k_tot = sqrt(kx*kx+ky*ky+kz*kz); // Magnitude of frequency
                if (k_tot != 0) {
                    double mu = ky/k_tot; // Anisotropy factor
                    double Pk_in = Pk_CAMB(Pow, a, k_tot);
                    double power1 = (b1 + mu*mu*f)*(b1 + mu*mu*f)*Pk_in; // Calculate the power
                    double power2 = (b2 + mu*mu*f)*(b2 + mu*mu*f)*Pk_in;
                    deltak3d1[index].re = power1/V; // Assign the power to the grid
                    deltak3d1[index].im = 0.0; // Set imaginary part to zero
                    deltak3d2[index].re = power2/V;
                    deltak3d2[index].im = 0.0;
                }
                else {
                    deltak3d1[index].re = 0.0;
                    deltak3d1[index].im = 0.0;
                    deltak3d2[index].re = 0.0;
                    deltak3d2[index].im = 0.0;
                }
            }
        }
    }
}

// This function distributes the input power to the grid. 
void Sampdk(double *kvec, fftw_complex *deltak3di, fftw_complex *deltak3d1, fftw_complex *deltak3d2, double *ratio) {    
    std::mt19937_64 generator; // Mersenne twister random number generator for real
    generator.seed(time(0)); // Seed the random number generator
    // Loop through the different combinations of frequency components
    for (int i = 0; i < N; ++i) {
        int i2 = (2*N - i) % N;
        
        double kx = 2*pi*kvec[i]; // x frequency
        
        for (int j = 0; j < N; ++j) {
            int j2 = (2*N - j) % N;
            
            double ky = 2*pi*kvec[j]; // y frequency
            
            for (int k = 0; k <= N/2; ++k) {
                
                int index = k + (N/2 + 1)*(j+N*i); // Calculate the associate array index
                
                double kz = 2*pi*kvec[k]; // z frequency
                
                double k_tot = sqrt(kx*kx+ky*ky+kz*kz); // Magnitude of frequency
                //if (k_tot == 0) continue;
                
                //double Power = sqrt(deltak3di[index].re*deltak3di[index].re+deltak3di[index].im*deltak3di[index].im);
                double Power = deltak3di[index].re;

                if (Power > 0) {
                    if (i == N/2 || j == N/2 || k == N/2 || (i == 0 && j == 0 && k == 0)) {
                        double variance = sqrt(Power); 
                        std::normal_distribution<double> distribution(0.0,variance);
                        
                        double deltare = distribution(generator);
                        
                        deltak3d1[index].re = deltare;
                        deltak3d1[index].im = 0.0;
                        
                        if (ratio[index] > 0) {
                            deltak3d2[index].re = sqrt(ratio[index])*deltare;
                            deltak3d2[index].im = 0.0;
                        } else {
                            deltak3d2[index].re = 0.0;
                            deltak3d2[index].im = 0.0;
                        }
                    } else if (k == 0) {
                        int index2 = k + (N/2 + 1)*(j2+N*i2);
                        double variance = sqrt(Power/2.0);
                        std::normal_distribution<double> distribution(0.0,variance);
                        
                        double deltare = distribution(generator);
                        double deltaim = distribution(generator);
                        
                        deltak3d1[index].re = deltare;
                        deltak3d1[index].im = deltaim;
                        
                        deltak3d1[index2].re = deltare;
                        deltak3d1[index2].im = -deltaim;
                        
                        if (ratio[index] > 0) {
                        deltak3d2[index].re = sqrt(ratio[index])*deltare;
                        deltak3d2[index].im = sqrt(ratio[index])*deltaim;
                        
                        deltak3d2[index2].re = sqrt(ratio[index])*deltare;
                        deltak3d2[index2].im = -sqrt(ratio[index])*deltaim;
                        } else {
                            deltak3d2[index].re = 0.0;
                            deltak3d2[index].im = 0.0;
                        
                            deltak3d2[index2].re = 0.0;
                            deltak3d2[index2].im = 0.0;
                        }
                    } else {
                        double variance = sqrt(Power/2.0);
                        std::normal_distribution<double> distribution(0.0,variance);
                        
                        double deltare = distribution(generator);
                        double deltaim = distribution(generator);
                        
                        deltak3d1[index].re = deltare;
                        deltak3d1[index].im = deltaim;
                        
                        if (ratio[index] > 0) {
                        deltak3d2[index].re = sqrt(ratio[index])*deltare;
                        deltak3d2[index].im = sqrt(ratio[index])*deltaim;
                        } else {
                            deltak3d2[index].re = 0.0;
                            deltak3d2[index].im = 0.0;
                        }
                    }
                }
                else {
                    deltak3d1[index].re = 0.0;
                    deltak3d1[index].im = 0.0;
                    
                    deltak3d2[index].re = 0.0;
                    deltak3d2[index].im = 0.0;
                }
            }
        }
    }
}

// This function Poisson samples the inverse Fourier transformed power and populates the volume
// with galaxies.
void Gendr(std::string file1, std::string file2, double variance1, double variance2, fftw_real *deltar3d1, fftw_real *deltar3d2) {
    // Create Mersenne twister random number generators for the Poisson sampling, and
    // to distribute galaxies in a uniform random manner within the cells.
    std::mt19937_64 generator;
    //std::mt19937_64 xgen;
    //std::mt19937_64 ygen;
    //std::mt19937_64 zgen;
    
    // Seed the generators
    generator.seed(time(0));
//     xgen.seed(time(0));
//     ygen.seed(time(0));
//     zgen.seed(time(0));
    
    std::ofstream fout; // Output filestream
    std::ofstream fout2;
    
    double n1 = nbar1*dL*dL*dL;
    double n2 = nbar2*dL*dL*dL;
    int count = 0;
    int count2 = 0;
    int count3 = 0;
    
    fout.open(file1.c_str(),std::ios::out); // Open output file
    fout.precision(15); // Set the number of digits to output
    
    fout2.open(file2.c_str(),std::ios::out);
    fout2.precision(15);
    
    // Loop through all the grid points, assign galaxies uniform random positions within
    // each cell.
    for (int i = 0; i < N; ++i) {
        double xmin = i*dL;
        double xmax = (i+1)*dL;
        
        // Uniform random distribution for the x coordinate
        std::uniform_real_distribution<double> xpos(xmin, xmax);
        for (int j = 0; j < N; ++j) {
            double ymin = j*dL;
            double ymax = (j+1)*dL;
            
            // Uniform random distribution for the y coordinate
            std::uniform_real_distribution<double> ypos(ymin, ymax);
            for (int k = 0; k < N; ++k) {
                double zmin = k*dL;
                double zmax = (k+1)*dL;
                
                // Uniform random distribution for the z coordinate
                std::uniform_real_distribution<double> zpos(zmin, zmax);
                
                int index = k + N*(j+N*i); // Calculate grid index
                
                // Check that the matter density field is positive. Set to zero if not.
                //if (deltar3d[index] < 0.0) continue;
                
                // Initialize the Poisson distribution with the value of the matter field
                double density1 = n1*exp(deltar3d1[index]-variance1/2);
                double density2 = n2*exp(deltar3d2[index]-variance2/2);
                std::poisson_distribution<int> distribution(density1);
                std::poisson_distribution<int> distribution2(density2);
                int numGal = distribution(generator); // Randomly Poisson sample
                int numGal2 = distribution2(generator);
                count += numGal;
                count2 += numGal2;
                
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
    fout.open("LRGNum.dat",std::ios::app);
    fout << count << " " << file1 << "\n";
    fout.close();
    fout.open("ELGNum.dat",std::ios::app);
    fout << count2 << " " << file2 << "\n";
    fout.close();
}

// This is a helper function to determine the values of frequencies to use when distributing
// the power to the grid.
void fftfreq(double *kvec) {
    double dk = 1.0/L;
    
    for (int i = 0; i <= N/2; ++i) {
        kvec[i] = i*dk;
    }
    for (int i = N/2+1; i < N; ++i) {
        kvec[i] = (i-N)*dk;
    }
}

// This function dynamically names files in the format "filebase####.fileext". The . should 
// be part of the fileext string.
std::string filename(std::string filebase, int filenum, std::string fileext) {
    std::string file; // Declare a string that will be returned
    
    std::stringstream ss; // Create a string stream to put the pieces together
    ss << filebase << std::setw(4) << std::setfill('0') << filenum << fileext; // Put pieces together
    file = ss.str(); // Save the filename to the string
    
    return file; // Return the string
}

int main() {
    omp_set_num_threads(numCores); // Set the number of threads for OpenMP parallel sections
    fftw_threads_init(); // Initialize threaded FFTs
    rfftwnd_plan dp_c2r; // Inverse FFT plan
    rfftwnd_plan dp_r2c; // Forward FFT plan
    // Create the plans using FFTW_MEASURE to get fastest transforms, do this here so
    // that it is only done once and the plans reused.
    
    std::cout << "Creating FFTW plans...\n";
    dp_c2r = rfftw3d_create_plan(N, N, N, FFTW_COMPLEX_TO_REAL, FFTW_MEASURE);
    dp_r2c = rfftw3d_create_plan(N, N, N, FFTW_REAL_TO_COMPLEX, FFTW_MEASURE);
    
    double *kvec = new double[N];
    fftfreq(kvec);
    
    std::ofstream fout;
    std::ofstream tout;
    std::ifstream fin;
    
    fout.open("LRGNum.dat",std::ios::out);
    fout.close();
    fout.open("ELGNum.dat",std::ios::out);
    fout.close();
    
    std::vector< Pk > InputPower;
    int numKModes = 0;
    
    std::cout << "Reading input power file: " << CAMBfile << "\n";
    fin.open(CAMBfile.c_str(),std::ios::in);
    while (!fin.eof()) {
        Pk Input_temp;
        fin >> Input_temp.k >> Input_temp.P;
        
        if (!fin.eof()) {
            InputPower.push_back(Input_temp);
            ++numKModes;
        }
    }
    fin.close();
    
    double *kvals = new double[numKModes];
    double *InPow = new double[numKModes];
    
    for (int i = 0; i < numKModes; ++i) {
        kvals[i] = InputPower[i].k;
        InPow[i] = InputPower[i].P;
    }
    
    gsl_spline *Power = gsl_spline_alloc(gsl_interp_cspline, numKModes);
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    
    gsl_spline_init(Power, kvals, InPow, numKModes);
    
    fftw_complex *deltak3di1 = new fftw_complex[N_im];
    fftw_real *deltar3di1 = new fftw_real[N_tot];
    fftw_complex *deltak3di2 = new fftw_complex[N_im];
    fftw_real *deltar3di2 = new fftw_real[N_tot];
    
#pragma omp parallel for
    for (int i = 0; i < N_tot; ++i) {
        deltar3di1[i] = 0.0;
        deltar3di2[i] = 0.0;
        if (i < N_im) {
            deltak3di1[i].re = 0.0;
            deltak3di1[i].im = 0.0;
            deltak3di2[i].re = 0.0;
            deltak3di2[i].im = 0.0;
        }
    }
    
    std::cout << "Distributing power over volume...\n";
    Gendk(kvec, Power, acc, deltak3di1, deltak3di2); // Call function to populate the power grid
    
    std::cout << "Performing initial one-time inverse FFT...\n";
    rfftwnd_threads_one_complex_to_real(numCores,dp_c2r,deltak3di1,deltar3di1); // FFT
    rfftwnd_threads_one_complex_to_real(numCores,dp_c2r,deltak3di2,deltar3di2);
    
    std::cout << "Taking the natural log...\n";
#pragma omp parallel for
    for (int i = 0; i < N_tot; ++i) {
        deltar3di1[i] = log(1.0 + deltar3di1[i]);
        deltar3di2[i] = log(1.0 + deltar3di2[i]);
        if (i < N_im) {
            deltak3di1[i].re = 0.0;
            deltak3di1[i].im = 0.0;
            deltak3di2[i].re = 0.0;
            deltak3di2[i].im = 0.0;
        }
    }
    
    std::cout << "Performing initial one-time forward FFT...\n";
    rfftwnd_threads_one_real_to_complex(numCores,dp_r2c,deltar3di1,deltak3di1);
    rfftwnd_threads_one_real_to_complex(numCores,dp_r2c,deltar3di2,deltak3di2);
    
    double *ratio = new double[N_im];
        
    std::cout << "Normalizing...\n";
#pragma omp parallel for
    for (int i = 0; i < N_im; ++i) {
        deltak3di1[i].re /= N_tot;
        deltak3di1[i].im /= N_tot;
        deltak3di2[i].re /= N_tot;
        deltak3di2[i].im /= N_tot;
        
        ratio[i] = deltak3di2[i].re/deltak3di1[i].re;
    }
    
    delete[] deltar3di1;
    delete[] deltar3di2;
    
    tout.open("Timings.dat",std::ios::out);
    std::cout << "Starting to generate mocks...\n";
    for (int mock = startNum-1; mock < numMocks; ++mock) {
        double start_time = omp_get_wtime();
        std::string lrgfile = filename(base, mock+1, ext);
        std::string elgfile = filename(base2, mock+1, ext);
        std::cout << "Generating mock " << lrgfile << " and " << elgfile << "\n";
        
        fftw_complex *deltak3d1 = new fftw_complex[N_im];
        fftw_real *deltar3d1 = new fftw_real[N_tot];
        fftw_complex *deltak3d2 = new fftw_complex[N_im];
        fftw_real *deltar3d2 = new fftw_real[N_tot];
        
        // Initialize power array. Do it in parallel to speed things up.        
#pragma omp parallel for
        for (int i = 0; i < N_tot; ++i) {
            deltar3d1[i] = 0.0;
            deltar3d2[i] = 0.0;
            if (i < N_im) {
                deltak3d1[i].re = 0.0;
                deltak3d1[i].im = 0.0;
                deltak3d2[i].re = 0.0;
                deltak3d2[i].im = 0.0;
            }
        }
        
        std::cout << "    Setting up for the inverse FFT...\n";
        Sampdk(kvec, deltak3di1, deltak3d1, deltak3d2, ratio);
        
        std::cout << "    Performing second inverse FFT...\n";
        rfftwnd_threads_one_complex_to_real(numCores,dp_c2r,deltak3d1,deltar3d1);
        rfftwnd_threads_one_complex_to_real(numCores,dp_c2r,deltak3d2,deltar3d2);
        
        double mean1 = 0.0;
        double mean2 = 0.0;
        double variance1 = 0.0;
        double variance2 = 0.0;
        double dr_max1 = 0.0;
        double dr_min1 = 0.0;
        double dr_max2 = 0.0;
        double dr_min2 = 0.0;
        
        for (int i = 0; i < N_tot; ++i) {
            mean1 += deltar3d1[i]/N_tot;
            mean2 += deltar3d2[i]/N_tot;
            if (deltar3d1[i] > dr_max1) dr_max1 = deltar3d1[i];
            if (deltar3d1[i] < dr_min1) dr_min1 = deltar3d1[i];
            if (deltar3d2[i] > dr_max2) dr_max2 = deltar3d2[i];
            if (deltar3d2[i] < dr_min2) dr_min2 = deltar3d2[i];
        }
        std::cout << "    Max 1  = " << dr_max1 << "\n";
        std::cout << "    Min 1  = " << dr_min1 << "\n";
        std::cout << "    Mean 1 = " << mean1 << "\n";
        std::cout << "    Max 2  = " << dr_max2 << "\n";
        std::cout << "    Min 2  = " << dr_min2 << "\n";
        std::cout << "    Mean 2 = " << mean2 << "\n";
        
        std::cout << "    Calculating variance...\n";
        for (int i = 0; i < N_tot; ++i) {
            deltar3d1[i] -= mean1;
            deltar3d2[i] -= mean2;
            variance1 += (deltar3d1[i]*deltar3d1[i])/(N_tot-1);
            variance2 += (deltar3d2[i]*deltar3d2[i])/(N_tot-1);
        }
        
        std::cout << "    Poisson sampling...\n";
        Gendr(lrgfile, elgfile, variance1, variance2, deltar3d1, deltar3d2);
        
        delete[] deltak3d1;
        delete[] deltar3d1;
        delete[] deltak3d2;
        delete[] deltar3d2;
        
        double totaltime = omp_get_wtime()-start_time;
        std::cout << "    Time to generate mock: " << totaltime << " seconds\n";
        tout << lrgfile << " " << totaltime << "\n";
    }
    tout.close();
    
    delete[] kvec;
    delete[] deltak3di1;
    delete[] deltak3di2;
    delete[] kvals;
    delete[] InPow;
    delete[] ratio;
    
    rfftwnd_destroy_plan(dp_r2c);
    rfftwnd_destroy_plan(dp_c2r);
    
    gsl_spline_free(Power);
    gsl_interp_accel_free(acc);
    
    return 0;
}