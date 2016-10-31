/* LNKNLogs.cpp v0.0
 * David W. Pearson & Lado Samushia
 * 5/5/2016
 * 
 * The goal of this code is to create lognormal mock catalogs with velocities assigned to the
 * galaxies in such a way as to correctly imprint the linear growth rate induced anisotropies
 * when transforming to redshift space. This will be achieved through a particle-mesh type
 * approach which takes the Gaussian density field generated during the lognormal procedure
 * and uses it to infer the velocities as v_x(k) = -H_0*f*k_x*d_G/k^2, and similarly for the
 * other two components.
 * 
 * The code will work as follows:
 *  1. Read in the parameter file containing the file name for the input power spectrum, the
 *     dimensions of the grid, the side lengths of the cuboid, the desired bias, the desired
 *     linear growth rate, desired number density, number of mocks to generate, starting 
 *     value for file numbering, the number of processor cores to use to perform FFTs and for
 *     any parallel sections of the code.
 *  2. Read in the input power spectrum and fit a cubic spline to ensure that all values
 *     needed for the filling the initial cube are present.
 *  3. Fill the initial k-space cube with b^2*P(k)/V going to only the real components.
 *  4. Perform complex-to-real transform with FFTW.
 *  5. Modify the resulting real space grid by taking ln(1+E(r)) at each point.
 *  6. Perform real-to-complex transform with FFTW.
 *  7. Fill k-space grid with values randomly drawn from a normal distribution with zero mean
 *     and RMS sqrt{max[0,Re(P_LN(k))/2]}
 *  8. Fill three other k-space grids with the random realization multiplied by 
 *     -f*H_0*k_i/k^2, where k_i is the component of k in a particular direction (x,y,z),
 *     to generate velocity field.
 *  9. Perform 4 total complex-to-real transforms to get the real space Gaussian field, and
 *     the three dimensional velocity.
 * 10. Calculate the mean and variance of the Gaussian field.
 * 11. Poisson sample the lognormal field N(r) = n*V_c*exp(d_G(r)-sigma_G^2/2) and assign
 *     the velocity associated with the cell to each tracer.
 * 
 * Initially this will be fairly crude, but should have the desired effect. With the 
 * positions and velocities, it will be possible to select a point of observation and use 
 * the velocities to shift the tracers in redshift space, imprinting the redshift space 
 * distortion induced anisotropies.
 * 
 * Compile with:
 * g++ -std=c++14 -lgsl -lgslcblas -lfftw3 -lfftw3_omp -lm -fopenmp -march=native -mtune=native -O3 -o LNKNLogs LNKNLogs.cpp lognormal.o
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>
#include <cmath>
#include <fftw3.h>
#include <omp.h>
#include "pods.h"
#include "lognormal.h"

struct parameters{
    std::string inPkFile, outbase, ext, dk3difile, ratiofile;
    int numMocks, startNum, digits, numCores, numTracers;
    double f, h, Omega_L, Omega_m, z;
    std::vector< double > b;
    std::vector< int > numGals;
    int3 N;
    double3 L;
} p;


void assignParams(std::vector< std::string > input) {
    std::string name = input[0];
    std::string val = input[1];
    if (name == "inPkFile") p.inPkFile = val;
    else if (name == "dk3difile") p.dk3difile = val;
    else if (name == "ratiofile") p.ratiofile = val;
    else if (name == "outbase") p.outbase = val;
    else if (name == "ext") p.ext = val;
    else if (name == "numMocks") p.numMocks = atof(val.c_str());
    else if (name == "startNum") p.startNum = atof(val.c_str());
    else if (name == "digits") p.digits = atof(val.c_str());
    else if (name == "b" || name == "bias") {
        std::istringstream iss(input[1]);
        std::string s;
        while (std::getline(iss, s, ',')) {
            p.b.push_back(atof(s.c_str()));
        }
    }
    else if (name == "f" || name == "growth") p.f = atof(val.c_str());
    else if (name == "h") p.h = atof(val.c_str());
    else if (name == "Nx") p.N.x = atof(val.c_str());
    else if (name == "Ny") p.N.y = atof(val.c_str());
    else if (name == "Nz") p.N.z = atof(val.c_str());
    else if (name == "Lx") p.L.x = atof(val.c_str());
    else if (name == "Ly") p.L.y = atof(val.c_str());
    else if (name == "Lz") p.L.z = atof(val.c_str());
    else if (name == "Omega_L") p.Omega_L = atof(val.c_str());
    else if (name == "Omega_m") p.Omega_m = atof(val.c_str());
    else if (name == "z") p.z = atof(val.c_str());
    else if (name == "numCores") p.numCores = atof(val.c_str());
    else if (name == "numGals") {
        std::istringstream iss(input[1]);
        std::string s;
        while (std::getline(iss, s, ',')) {
            p.numGals.push_back(atof(s.c_str()));
        }
    }
    else if (name == "numTracers") p.numTracers = atof(val.c_str());
    else {
        std::cout << "WARNING: Unrecognized key in parameter file\n";
        std::cout << "    " << name << " is not a valid parameter\n";
    }
}

void readParams(char *file) {
    std::ifstream fin;
    std::string line;
    
    fin.open(file, std::ios::in);
    while (std::getline(fin, line, '\n')) {
        std::istringstream iss(line);
        std::vector< std::string > input;
        std::string s;
        int entry = 0;
        while (std::getline(iss, s, '=')) {
            input.push_back(s);
            entry++;
        }
        assignParams(input);
    }
    fin.close();
}

std::string filename(std::string filebase, int digits, int filenum, std::string fileext) {
    std::string file;
    
    std::stringstream ss;
    ss << filebase << std::setw(digits) << std::setfill('0') << filenum << fileext;
    file = ss.str();
    
    return file;
}

int main(int argc, char *argv[]) {
    std::ifstream fin;
    std::ofstream fout;
    std::cout << "LNKNLogs v0.0\n\n";
    
    if (argc == 1) {
        std::cout << "ERROR: No parameter file specified.\n";
        std::cout << "    This code requires a parameter file to run correctly. When\n";
        std::cout << "    executing this code from the command line please specify the\n";
        std::cout << "    name of the parameter file.\n\n";
        std::cout << "    Ex:\n";
        std::cout << "        $ ./LNKNLogs MyParameters.params\n\n";
        std::cout << "    where '.params' can be any extension and the file is a plain\n";
        std::cout << "    text file with key, value pairs. See the included example file\n";
        std::cout << "    and documentation for details.\n";
        return 0;
    }
    
    readParams(argv[1]);
    
    double bias = 0.0;
    for (int i = 0; i < p.numTracers; ++i) {
        std::cout << p.b[i] << "\n";
        if (p.b[i] > bias) bias = p.b[i];
    }
    std::cout << "Highest Bias: " << bias << "\n";
    
    std::cout << "Preparing to generate " << p.numMocks << " mock catalogs...\n";
    
    const long int N_tot = p.N.x*p.N.y*p.N.z;
    const long int N_im = p.N.x*p.N.y*(p.N.z/2 + 1);
    double V = p.L.x*p.L.y*p.L.z;
    double *nbar = new double[p.numTracers];
    for (int i = 0; i < p.numTracers; ++i) {
        nbar[i] = double(p.numGals[i])/V;
    }
    
    std::cout << "    Creating Fourier transform plans...\n";
    fftw_init_threads();
    double *dr3di = new double[N_tot];
    fftw_complex *dk3di = new fftw_complex[N_im];
    double *dr3dm = new double[N_tot];
    fftw_complex *dk3dm = new fftw_complex[N_im];
    
    fftw_import_wisdom_from_filename("FFTWWisdom.dat");
    fftw_plan_with_nthreads(p.numCores);
    fftw_plan dk3di2dr3di = fftw_plan_dft_c2r_3d(p.N.x, p.N.y, p.N.z, dk3di, dr3di, 
                                                 FFTW_MEASURE);
    fftw_plan dr3di2dk3di = fftw_plan_dft_r2c_3d(p.N.x, p.N.y, p.N.z, dr3di, dk3di, 
                                                 FFTW_MEASURE);
    fftw_plan dk3dm2dr3dm = fftw_plan_dft_c2r_3d(p.N.x, p.N.y, p.N.z, dk3dm, dr3dm, 
                                                 FFTW_MEASURE);
    fftw_plan dr3dm2dk3dm = fftw_plan_dft_r2c_3d(p.N.x, p.N.y, p.N.z, dr3dm, dk3dm, 
                                                 FFTW_MEASURE);
    fftw_export_wisdom_to_filename("FFTWWisdom.dat");
    
    std::cout << "    Reading in input power spectrum...\n";
    
    std::vector< double > kin;
    std::vector< double > Pin;
    fin.open(p.inPkFile.c_str(), std::ios::in);
    while (!fin.eof()) {
        double ktemp;
        double Ptemp;
        
        fin >> ktemp >> Ptemp;
        Ptemp /= V;
        if (!fin.eof()) {
            kin.push_back(ktemp);
            Pin.push_back(Ptemp);
        }
    }
    fin.close();
    
    std::cout << "    Filling k-space cubes...\n";
    Gendk(p.N, p.L, bias, &kin[0], &Pin[0], Pin.size(), dk3di);
    Gendk(p.N, p.L, 1.0, &kin[0], &Pin[0], Pin.size(), dk3dm);
    
    std::cout << "    Taking the inverse Fourier transform...\n";
    fftw_execute(dk3di2dr3di);
    fftw_execute(dk3dm2dr3dm);
    
    std::cout << "    Taking the natural log...\n";
    //#pragma omp parallel for schedule(auto)
    for (int i = 0; i < N_tot; ++i) {
        dr3di[i] = log(1.0+dr3di[i]);
        dr3dm[i] = log(1.0+dr3dm[i]);
    }
    
    std::cout << "    Going back to k-space...\n";
    fftw_execute(dr3di2dk3di);
    fftw_execute(dr3dm2dk3dm);
    
    std::cout << "    Normalizing and transfering to permenant storage...\n";
    //#pragma omp parallel for schedule(auto)
    for(int i = 0; i < N_im; ++i) {
        dk3di[i][0] /= double(N_tot);
        dk3di[i][1] /= double(N_tot);
        dk3dm[i][0] /= double(N_tot);
        dk3dm[i][1] /= double(N_tot);
        
        if (dk3di[i][0] < 0) dk3di[i][0] = 0.0;
        if (dk3dm[i][0] < 0) dk3dm[i][0] = 0.0;
        
        if (dk3dm[i][0] > 0) dk3dm[i][1] = dk3di[i][0]/dk3dm[i][0];
        else dk3dm[i][1] = 0.0;
    }
    
    std::cout << "Zeroth element of k-space array: " << dk3di[0][0] << "\n";
    
    double fileioTime = omp_get_wtime();
    fout.open(p.dk3difile.c_str(), std::ios::out|std::ios::binary);
    fout.write((char *) dk3di, N_im*sizeof(fftw_complex));
    fout.close();
    std::cout << "    Time to write binary file: " << omp_get_wtime()-fileioTime << " s\n";
    
    delete[] dr3di;
    delete[] dk3di;
    delete[] dk3dm;
    delete[] dr3dm;
    
    std::cout << "Starting mock catalog creation...\n";
    std::vector< double > dr3d(N_tot);
    std::vector< double > vr3dx(N_tot);
    std::vector< double > vr3dy(N_tot);
    std::vector< double > vr3dz(N_tot);
    std::vector< fftw_complex > dk3d(N_im);
    std::vector< fftw_complex > vk3dx(N_im);
    std::vector< fftw_complex > vk3dy(N_im);
    std::vector< fftw_complex > vk3dz(N_im);
    
    fftw_plan dk3d2dr3d = fftw_plan_dft_c2r_3d(p.N.x, p.N.y, p.N.z, &dk3d[0], &dr3d[0], 
                                               FFTW_MEASURE);
    fftw_plan vk3dx2vr3dx = fftw_plan_dft_c2r_3d(p.N.x, p.N.y, p.N.z, &vk3dx[0], &vr3dx[0], 
                                                 FFTW_MEASURE);
    fftw_plan vk3dy2vr3dy = fftw_plan_dft_c2r_3d(p.N.x, p.N.y, p.N.z, &vk3dy[0], &vr3dy[0], 
                                                 FFTW_MEASURE);
    fftw_plan vk3dz2vr3dz = fftw_plan_dft_c2r_3d(p.N.x, p.N.y, p.N.z, &vk3dz[0], &vr3dz[0], 
                                                 FFTW_MEASURE);
    fftw_destroy_plan(dr3di2dk3di);
    fftw_destroy_plan(dk3di2dr3di);
    fftw_destroy_plan(dk3dm2dr3dm);
    fftw_destroy_plan(dr3dm2dk3dm);
    for (int mock = p.startNum; mock < p.numMocks+p.startNum; ++mock) {
        double startTime = omp_get_wtime();
        
        std::string outfile = filename(p.outbase, p.digits, mock, p.ext);
        std::cout << "  Creating " << outfile << "\n";
        
        std::cout << "    Random sampling...\n";
        double sampTime = omp_get_wtime();
        Smpdk(p.N, p.L, bias, p.h, p.f, p.dk3difile, &dk3d[0], &vk3dx[0], &vk3dy[0], &vk3dz[0]);
        std::cout << "    Sampling Time: " << omp_get_wtime()-sampTime << " s\n";
        
        double fftwTime = omp_get_wtime();
        fftw_execute(vk3dx2vr3dx);
        std::cout << "    Time for transform: " << omp_get_wtime()-fftwTime << " s\n";
        fftwTime = omp_get_wtime();
        fftw_execute(vk3dy2vr3dy);
        std::cout << "    Time for transform: " << omp_get_wtime()-fftwTime << " s\n";
        fftwTime = omp_get_wtime();
        fftw_execute(vk3dz2vr3dz);
        std::cout << "    Time for transform: " << omp_get_wtime()-fftwTime << " s\n";
        
        std::cout << p.h << ", " << p.f << "\n";
//#pragma omp parallel for schedule(auto)
        for (int i = 0; i < N_tot; ++i) {
            vr3dx[i] *= 100.0*p.h*p.f*sqrt(p.Omega_m*(1.0+p.z)*(1.0+p.z)*(1.0+p.z)+p.Omega_L);
            vr3dy[i] *= 100.0*p.h*p.f*sqrt(p.Omega_m*(1.0+p.z)*(1.0+p.z)*(1.0+p.z)+p.Omega_L);
            vr3dz[i] *= 100.0*p.h*p.f*sqrt(p.Omega_m*(1.0+p.z)*(1.0+p.z)*(1.0+p.z)+p.Omega_L);
        }
        
        std::cout << "    Transforming dk3d...\n";
        fftwTime = omp_get_wtime();
        fftw_execute(dk3d2dr3d);
        std::cout << "    Time for transform: " << omp_get_wtime()-fftwTime << " s\n";
        
        double mean = 0.0;
        double variance = 0.0;
        
        for (int i = 0; i < N_tot; ++i)
            mean += dr3d[i];
        
        mean /= double(N_tot);
        std::cout << "    mean = " << mean << "\n";
        
        for (int i = 0; i < N_tot; ++i) {
            dr3d[i] -= mean;
            variance += ((dr3d[i])*(dr3d[i]))/double(N_tot - 1.0);
        }
        
        std::cout << "    variance = " << variance << "\n";
        
        std::cout << "    Poisson sampling...\n";
        sampTime = omp_get_wtime();
        Gendr(p.N, p.L, nbar, p.numTracers, outfile, variance, &dr3d[0], &vr3dx[0], 
                     &vr3dy[0], &vr3dz[0], p.b, bias);
        std::cout << "    Sampling Time: " << omp_get_wtime()-sampTime << " s\n";
        
        std::cout << "  Time to create mock: " << omp_get_wtime()-startTime << " s\n";
        
//         fout.open("DensitySlice.dat",std::ios::out);
//         for (int i = 0; i < p.N.x; ++i) {
//             double x = (p.L.x/double(p.N.x))*(i+0.5);
//             for (int j = 0; j < p.N.y; ++j) {
//                 double y = (p.L.y/double(p.N.y))*(j+0.5);
//                 double density = 0.0;
//                 //for (int k = 0; k < 11; ++k) {
//                 int k = 364;
//                     int index = k + p.N.z*(j + p.N.y*i);
//                     density += dr3d[index];
//                 //}
//                 fout << x << " " << y << " " << density <<  " " << vr3dx[index] << " " << vr3dy[index] << "\n";
//             }
//         }
//         fout.close();
    }
    delete[] nbar;
    fftw_destroy_plan(dk3d2dr3d);
    fftw_destroy_plan(vk3dx2vr3dx);
    fftw_destroy_plan(vk3dy2vr3dy);
    fftw_destroy_plan(vk3dz2vr3dz);
    
    std::remove(p.dk3difile.c_str());
    std::remove(p.ratiofile.c_str());
    
    return 0;
}
