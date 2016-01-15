// LogNormal.cpp
// 12/21/2015
// David W. Pearson & Lado Samushia
//
// This code is designed to generate log-normal mock galaxy catalogs. While these catalogs
// are perhaps not as good as those generated from N-body simulations, or 2LPT methods,
// they are computationally much cheaper. A lot of this code is lifted from MockGen.cpp, 
// which was intended to generate mocks more directly from an input power spectrum. MockGen
// was a C++ port of Lado Samushia's python code which performed the same function, but 
// used much less memory than numpy's FFT functions.
//
// This code follows the outline for log-normal mock generation provided in Beutler et al.
// 2011. It starts by taking an input power spectrum from a file generated using CAMB, then
// distributes that power divided by the volume to the real parts of the elements of an
// fftw_complex array of size N*N*(N/2 + 1), while setting the imaginary components to zero.
// A complex to real FFT is performed, and natural log of 1 plus each element is taken. 
// A real to complex FFT is performed, and the result is normalized by the number of real
// array elements. After that, the code enters the mock generation loop where the normalized
// output of the real to complex FFT is Gaussian random sampled. That data is then put
// through another complex to real FFT and the outputs mean and variance calculated. The
// final step is to take the output of the last FFT, calculate n*exp(dr[i] - var/2), and
// then Poisson sample that to determine the number of galaxies to place in each cell i.
// The galaxies are placed in a uniform random manner in the cell.
//
// Compile with:
// g++ -std=c++11 -lgsl -lgslcblas -lfftw -lrfftw -lfftw_threads -lrfftw_threads -lm -fopenmp -march=native -mtune=native -O2 LogNormal.cpp -o LogNormal

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

const int numMocks = 20; // Number of mocks to generate
const int startNum = 1;

const double pi = 3.14159265359; // Mmmmmmm, Pi!

const std::string base = "galaxies"; // Base of the filename for the mocks
const std::string matbase = "matterfield"; // Base of the filename for the matter field
const std::string powbase = "powerfield"; // Base of the filename for the power field
const std::string ext = ".dat"; // Extension to use for creating plain text files
const std::string extbin = ".bin"; // Extension to use for creating binary files
const std::string CAMBfile = "camb_95039962_matterpower_z0.57.dat"; // Input power spectrum

const bool matOut = false; // Control whether you want to output the matter field
const bool powOut = false; // Control whether you want to output the raw power array

const int N = 1024; // Number of grid points in one dimension.
const long int N_tot = N*N*N; // Total number of elements in arrays for FFTs
const long int N_im = N*N*(N/2 + 1); // Number of imaginary arrray elements
const double A = 80000.0; // Amplitude of the input power spectrum
const double mean = 0.01; // Location of the peak of the input power spectrum
const double sig = 0.15; // Width of the input power spectrum
const int num_gal = 4000000; // Number of galaxies per mock
const int num_gal2 = 286913;
const double L = 2048.0; // Size of the cube
const double dL = L/N; // Size of a grid cell
const double nbar = num_gal/(L*L*L);
const double nbar2 = num_gal2/(L*L*L);
const double V = L*L*L;

const double b = 2.0; // Bias of galaxies
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

// This function defines an Gaussian power spectrum using the parameters from above.
double Pk_Gauss(double mu, double k) {
    double diffsq = (k-mean)*(k-mean);
    double sigsq = sig*sig;
    double exponent = -diffsq/(2.0*sigsq);
    double isoPk = A*exp(exponent);
    double power = (b+mu*mu*f)*(b+mu*mu*f)*isoPk;
    
    return power;
}

double Pk_CAMB(gsl_spline *Pow, gsl_interp_accel *a, double mu, double k) {
    double power = gsl_spline_eval(Pow, k, a);
    
    //power *= (b + mu*mu*f)*(b + mu*mu*f);
    
    return power;
}

// This function distributes the input power to the grid. 
void Gendk(double *kvec, gsl_spline *Pow, gsl_interp_accel *a, fftw_complex *deltak3d) {
    
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
                    double Pk_in = Pk_CAMB(Pow, a, mu, k_tot);
                    double power = (b + mu*mu*f)*(b + mu*mu*f)*Pk_in; // Calculate the power
                    deltak3d[index].re = power/V; // Assign the power to the grid
                    deltak3d[index].im = 0.0; // Set imaginary part to zero
                }
                else {
                    deltak3d[index].re = 0.0;
                    deltak3d[index].im = 0.0;
                }
            }
        }
    }
}

// This function distributes the input power to the grid. 
void Sampdk(double *kvec, fftw_complex *deltak3di, fftw_complex *deltak3d) {    
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
                        
                        deltak3d[index].re = distribution(generator);
                        deltak3d[index].im = 0.0;
                    } else if (k == 0) {
                        int index2 = k + (N/2 + 1)*(j2+N*i2);
                        double variance = sqrt(Power/2.0);
                        std::normal_distribution<double> distribution(0.0,variance);
                        
                        double deltare = distribution(generator);
                        double deltaim = distribution(generator);
                        
                        deltak3d[index].re = deltare;
                        deltak3d[index].im = deltaim;
                        
                        deltak3d[index2].re = deltare;
                        deltak3d[index2].im = -deltaim;
                    } else {
                        double variance = sqrt(Power/2.0);
                        std::normal_distribution<double> distribution(0.0,variance);
                        
                        deltak3d[index].re = distribution(generator);
                        deltak3d[index].im = distribution(generator);
                    }
                }
                else {
                    deltak3d[index].re = 0.0;
                    deltak3d[index].im = 0.0;
                }
            }
        }
    }
}

// This function Poisson samples the inverse Fourier transformed power and populates the volume
// with galaxies.
void Gendr(std::string file, double variance, fftw_real *deltar3d) {
    // Create Mersenne twister random number generators for the Poisson sampling, and
    // to distribute galaxies in a uniform random manner within the cells.
    std::mt19937_64 generator;
    
    // Seed the generator
    generator.seed(time(0));
    
    std::ofstream fout; // Output filestream
    
    double n = nbar*dL*dL*dL;
    int count = 0;
    
    fout.open(file.c_str(),std::ios::out); // Open output file
    fout.precision(15); // Set the number of digits to output
    
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
                double density = n*exp(deltar3d[index]-variance/2);
                std::poisson_distribution<int> distribution(density);
                int numGal = distribution(generator); // Randomly Poisson sample
                count += numGal;
                
                // Randomly generate positions for numGal galaxies within the cell
                for (int gal = 0; gal < numGal; ++gal) {
                    fout << xpos(generator) << " " << ypos(generator) << " " << zpos(generator) << "\n";
                }
            }
        }
    }
    fout.close(); // Close file
    fout.open("GalaxyNum.dat",std::ios::app);
    fout << count << " " << file << "\n";
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
    
    fout.open("GalaxyNum.dat",std::ios::out);
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
    
    fftw_complex *deltak3di = new fftw_complex[N_im];
    fftw_real *deltar3di = new fftw_real[N_tot];
    
#pragma omp parallel for
    for (int i = 0; i < N_tot; ++i) {
        deltar3di[i] = 0.0;
        if (i < N_im) {
            deltak3di[i].re = 0.0;
            deltak3di[i].im = 0.0;
        }
    }
    
    std::cout << "Distributing power over volume...\n";
    Gendk(kvec, Power, acc, deltak3di); // Call function to populate the power grid
    
    std::cout << "Performing initial one-time inverse FFT...\n";
    rfftwnd_threads_one_complex_to_real(numCores,dp_c2r,deltak3di,deltar3di); // FFT
    
    std::cout << "Taking the natural log...\n";
#pragma omp parallel for
    for (int i = 0; i < N_tot; ++i) {
        deltar3di[i] = log(1.0 + deltar3di[i]);
        if (i < N_im) {
            deltak3di[i].re = 0.0;
            deltak3di[i].im = 0.0;
        }
    }
    
    std::cout << "Performing initial one-time forward FFT...\n";
    rfftwnd_threads_one_real_to_complex(numCores,dp_r2c,deltar3di,deltak3di);
        
    std::cout << "Normalizing...\n";
#pragma omp parallel for
    for (int i = 0; i < N_im; ++i) {
        deltak3di[i].re /= N_tot;
        deltak3di[i].im /= N_tot;
    }
    
    delete[] deltar3di;
    
    tout.open("Timings.dat",std::ios::out);
    std::cout << "Starting to generate mocks...\n";
    for (int mock = startNum-1; mock < numMocks; ++mock) {
        double start_time = omp_get_wtime();
        std::string lrgfile = filename(base, mock+1, ext);
        std::cout << "Generating mock " << lrgfile << "\n";
        
        fftw_complex *deltak3d = new fftw_complex[N_im];
        fftw_real *deltar3d = new fftw_real[N_tot];
        
        // Initialize power array. Do it in parallel to speed things up.        
#pragma omp parallel for
        for (int i = 0; i < N_tot; ++i) {
            deltar3d[i] = 0.0;
            if (i < N_im) {
                deltak3d[i].re = 0.0;
                deltak3d[i].im = 0.0;
            }
        }
        
        std::cout << "    Setting up for the inverse FFT...\n";
        Sampdk(kvec, deltak3di, deltak3d);
        
        if (powOut) {
            std::cout << "    Outputting raw power array...\n";
            std::string powerfile = filename(powbase, mock+1, extbin);
            fout.open(powerfile.c_str(),std::ios::out|std::ios::binary);
            fout.write((char *) deltak3d, N_im*sizeof(fftw_complex));
            fout.close();
        }
        
        std::cout << "    Performing second inverse FFT...\n";
        rfftwnd_threads_one_complex_to_real(numCores,dp_c2r,deltak3d,deltar3d);
        
        if (matOut) {
            std::cout << "    Outputting matter field array...\n";
            std::string matterfile = filename(matbase, mock+1, extbin);
            fout.open(matterfile.c_str(),std::ios::out|std::ios::binary);
            fout.write((char *) deltar3d, N_tot*sizeof(fftw_real));
            fout.close();
        }
        
        double mean = 0.0;
        double variance = 0.0;
        double dr_max = 0.0;
        double dr_min = 0.0;
        
        for (int i = 0; i < N_tot; ++i) {
            mean += deltar3d[i]/N_tot;
            if (deltar3d[i] > dr_max) dr_max = deltar3d[i];
            if (deltar3d[i] < dr_min) dr_min = deltar3d[i];
        }
        std::cout << "    Max  = " << dr_max << "\n";
        std::cout << "    Min  = " << dr_min << "\n";
        std::cout << "    Mean = " << mean << "\n";
        
        std::cout << "    Calculating variance...\n";
        for (int i = 0; i < N_tot; ++i) {
            deltar3d[i] -= mean;
            variance += (deltar3d[i])*(deltar3d[i])/(N_tot-1);
        }
        
        std::cout << "    Poisson sampling...\n";
        Gendr(lrgfile, variance, deltar3d);
        
        delete[] deltak3d;
        delete[] deltar3d;
        
        double totaltime = omp_get_wtime()-start_time;
        std::cout << "    Time to generate mock: " << totaltime << " seconds\n";
        tout << lrgfile << " " << totaltime << "\n";
    }
    tout.close();
    
    delete[] kvec;
    delete[] deltak3di;
    delete[] kvals;
    delete[] InPow;
    
    rfftwnd_destroy_plan(dp_r2c);
    rfftwnd_destroy_plan(dp_c2r);
    
    gsl_spline_free(Power);
    gsl_interp_accel_free(acc);
    
    return 0;
}