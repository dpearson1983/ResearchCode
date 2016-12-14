/* LNPk-rad.cpp
 * David W. Pearson
 * November 11, 2016
 * 
 * This code will calculate the power spectrum multipoles using the Biachi-Scoccimarro
 * FFT based implementation of the Yamamoto estimator. The code will start by only calculating
 * the monopole and quadrupole, though it should be straightforward to extend to calculate the 
 * hexadecapole. In order to keep memory usage to a reasonable level, the code will allocate
 * one pair of arrays for all necessary FFTs. The overdensity will be stored in a separate
 * array and one array will be needed to store the summed parts of the quadrupole in k space, so
 * that in all 2 real space arrays are needed and 2 k space array. This means that the
 * memory usage will scale as:
 * 
 *    2 x 8 bytes x N_x x N_y x N_z + 2 x 2 x 8 bytes x N_x x N_y x (N_z/2 + 1) = 
 *    34 bytes x N_x x N_y x (N_z + 1)
 * 
 * For the quadrupole, the array storing the overdensity will be used to fill the real space
 * FFT array using the formula
 *  
 *    B_ij(r) = ((r_i x r_j)/r^2)F(r)
 * 
 * where F(r) is the overdensity, B_ij(r) would be the specific FFT reals space array value,
 * and the i and j are the Cartesian coordinates x, y and z. This needs to be done six times
 * for the combinations, xx, yy, zz, xy, xz, yz, where the combinations that are mixed have
 * a multiple of 2 to account for the permutations. After a particular combination is used
 * to fill the real space array, the FFT will be performed and then the result added to a
 * separate k-space array as
 * 
 *    A_2(k) += (2 - delta_ij) x ((k_i x k_j)/k^2) x B_ij(k)
 * 
 * where delta_ij is the Kronecker delta (i.e. delta_ij = 1 if i = j, 0 otherwise). Once all
 * 6 contributions are added in the real space array will simply be filled with the overdensity,
 * then Fourier transformed and the associated k space array becomes A_0(k). This can then
 * be binned to give the results.
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>
#include <cmath>
#include <pods.h>
#include <fftw3.h>
#include <omp.h>
#include <fileFuncs.h>
#include <pfunk.h>
#include <harppi.h>

int main(int argc, char *argv[]) {
    parameters p(argv[1]);
    p.print();
    
    std::ifstream fin;
    std::ofstream fout;
    
    std::vector<std::string> ij{"xx", "yy", "zz", "xy", "xz", "yz"};
    
    // Determine the number randoms in the file
    long int randomsSize = filesize(p.gets("randomsFile"));
    std::cout << "Randoms file size: " << randomsSize << std::endl;
    std::cout << "galaxy size: " << sizeof(galaxy) << std::endl;
    int numRans = randomsSize/sizeof(galaxy);
    std::cout << "Number of Randoms: " << numRans << std::endl;
    
    // Allocate memory to read in the randoms. Use a pointer so that memory can
    // be freed after randoms are binned.
    galaxy *rans = new galaxy[numRans];
    
    // Read in the randoms
    fin.open(p.gets("randomsFile").c_str(), std::ios::in|std::ios::binary);
    fin.read((char *) rans, numRans*sizeof(galaxy));
    fin.close();
    
    // Setup some convenience variables
    int3 N = {p.geti("Nx"), p.geti("Ny"), p.geti("Nz")};
    long int N_r = N.x*N.y*N.z;
    long int N_k = N.x*N.y*(N.z/2 + 1);
    double3 L = {p.getd("Lx"), p.getd("Ly"), p.getd("Lz")};
    double3 dr = {L.x/double(N.x), L.y/double(N.y), L.z/double(N.z)};
    double3 rmin = {p.getd("x_min"), p.getd("y_min"), p.getd("z_min")};
    double3 robs = {p.getd("x_obs"), p.getd("y_obs"), p.getd("z_obs")};
    
    // Declare memory to store the binned randoms info and initialize to zero
    std::vector<double> nden_ran;
    nden_ran.reserve(N_r);
    initArray(&nden_ran[0], N_r);
    int corrType = 0;
    
    double V_ran = 0.0;
    // Bin the randoms using either nearest grid point (NGP) or cloud-in-cell (CIC).
    // Default is to use NGP if not specified in parameter file.
    std::cout << "Assigning randoms to the grid..." << std::endl;
    if (p.gets("massAssign") == "NGP" || !p.checkParam("massAssign")) {
        V_ran = binNGP(rans, &nden_ran[0], dr, numRans, N, robs);
    } else if (p.gets("massAssign") == "CIC") {
        V_ran = binCIC(rans, &nden_ran[0], dr, numRans, N, robs);
        corrType = 1;
    } else {
        std::stringstream message;
        message << "ERROR: " << p.gets("massAssign") << " is not a supported mass assignment\n";
        message << "scheme. The supported ones are NGP or CIC. Please correct the parameter\n";
        message << "file and rerun the code.\n";
        throw std::runtime_error(message.str());
    }
    double nbar_ran = double(numRans)/V_ran;
    
    // Free memory for reading in the randoms file.
    delete[] rans;
    
    int numThreads = 0;
    if (p.checkParam("numThreads")) {
        numThreads = p.geti("numThreads");
    } else {
        numThreads = omp_get_max_threads();
    }
    
    // Setup for the FFTs
    fftw_init_threads();
    double *dr3d = new double[N_r];
    fftw_complex *dk3d = new fftw_complex[N_k];
    
    fftw_import_wisdom_from_filename("FFTWWisdom.dat");
    fftw_plan_with_nthreads(numThreads);
    fftw_plan dr3d2dk3d = fftw_plan_dft_r2c_3d(N.x, N.y, N.z, dr3d, dk3d, FFTW_MEASURE);
    fftw_export_wisdom_to_filename("FFTWWisdom.dat");
    
    // Loop through all the mocks the user wishes to process specified by the values
    // of startNum and numMocks in the parameter file. HARPPI will throw an error if those
    // values are not present.
    for (int mock = p.geti("startNum"); mock < p.geti("numMocks")+p.geti("startNum"); ++mock) {
        // Dynamically get file names for input and output.
        std::string ifile = filename(p.gets("iBase"), p.geti("digits"), mock, p.gets("iExt"));
        std::string ofile = filename(p.gets("oBase"), p.geti("digits"), mock, p.gets("oExt"));
        
        std::cout << "Processing " << ifile << "..." << std::endl;
        
        long int galaxiesSize = filesize(ifile);
        int numGals = galaxiesSize/sizeof(galaxy);
        std::cout << "    Number of Galaxies: " << numGals << std::endl;
        
        std::cout << "    Allocating memory to read in galaxies..." << std::endl;
        galaxy *gals = new galaxy[numGals];
        
        std::cout << "    Reading in galaxies..." << std::endl;
        fin.open(ifile, std::ios::in|std::ios::binary);
        fin.read((char *) gals, numGals*sizeof(galaxy));
        fin.close();
        
        std::cout << "    Binning galaxies..." << std::endl;
        std::vector<double> nden_gal;
        nden_gal.reserve(N_r);
        initArray(&nden_gal[0], N_r);
        
        double V = 0.0;
        if (p.gets("massAssign") == "NGP" || !p.checkParam("massAssign")) {
            V = binNGP(gals, &nden_gal[0], dr, numGals, N, robs);
        } else if (p.gets("massAssign") == "CIC") {
            V = binCIC(gals, &nden_gal[0], dr, numGals, N, robs);
        }
        
        delete[] gals;
        
        double nbar = double(numGals)/V;
        double alpha = double(numGals)/double(numRans);
        double shotnoise = double(numGals) + alpha*alpha*double(numRans);
        double gal_nbsqwsq = alpha*double(numRans)*nbar;
        
        std::cout << "    nbar = " << nbar << std::endl;
        std::cout << "    alpha = " << alpha << std::endl;
        std::cout << "    1/nbar = " << 1.0/nbar << std::endl;
        std::cout << "    shotnoise = " << shotnoise/gal_nbsqwsq << std::endl;
        std::cout << "    gal_nbsqwsq = " << gal_nbsqwsq << std::endl;
        
        std::cout << "    Calculating A_2..." << std::endl;
        fftw_complex *A_2 = new fftw_complex[N_k];
        initArray(A_2, N_k);
        for (int i = 0; i < 6; ++i) {
            double *Br3d = new double[N_r];
            fftw_complex *Bk3d = new fftw_complex[N_k];
            fftw_plan_with_nthreads(numThreads);
            fftw_plan Br3d2Bk3d = fftw_plan_dft_r2c_3d(N.x, N.y, N.z, Br3d, Bk3d, FFTW_MEASURE);
            std::cout << "    Zeroing arrays for the FFT..." << std::endl;
            initArray(Br3d, Bk3d, N_r, N_k);
            std::cout << "    Calculating B_" << ij[i] << "(r)..." << std::endl;
            calcB_ij(Br3d, &nden_gal[0], &nden_ran[0], alpha, dr, rmin, robs, N, ij[i]);
            std::cout << "    Fourier transforming to get B_"<< ij[i] << "(k)..." << std::endl;
            fftw_execute(Br3d2Bk3d);
            std::cout << "    Adding contribution to A_2(k)..." << std::endl;
            accumulateA_2(A_2, Bk3d, N, L, ij[i]);
            delete[] Br3d;
            delete[] Bk3d;
            fftw_destroy_plan(Br3d2Bk3d);
        }
        
        std::cout << "    Calculating delta(r) from binned galaxies and randoms..." << std::endl;
        initArray(dr3d, dk3d, N_r, N_k);
        for (long int i = 0; i < N_r; ++i) {
            dr3d[i] = nden_gal[i] - alpha*nden_ran[i];
        }
        
        // Peform FFT
        std::cout << "    Performing FFT..." << std::endl;
        fftw_execute(dr3d2dk3d);
        
        // Bin frequencies
        std::cout << "    Binning frequencies..." << std::endl;
        std::vector<double> P_0(p.geti("numKBins"));
        std::vector<double> P_2(p.geti("numKBins"));
        std::vector<int> N_0(p.geti("numKBins"));
        freqBinBS(dk3d, A_2, &P_0[0], &P_2[0], &N_0[0], N, L, shotnoise, p.getd("k_min"), 
                  p.getd("k_max"), p.geti("numKBins"), p.getb("grid_cor"), corrType);
        
        normalizePk(&P_0[0], &P_2[0], &N_0[0], gal_nbsqwsq, p.geti("numKBins"));
        if (p.getb("discreteCor")) {
            correct_discreteness(p.gets("cor_file"), &P_0[0], &P_2[0], p.geti("numKBins"));
        }
        double dk = (p.getd("k_max") - p.getd("k_min"))/p.getd("numKBins");
        fout.open(ofile.c_str(), std::ios::out);
        fout.precision(15);
        for (int i = 0; i < p.geti("numKBins"); ++i) {
            double k = p.getd("k_min") + (i + 0.5)*dk;
            fout << k << " " << P_0[i] << "\n";
        }
        for (int i = 0; i < p.geti("numKBins"); ++i) {
            double k = p.getd("k_min") + (i + 0.5)*dk;
            fout << k << " " << P_2[i] << "\n";
        }
        fout.close();
        
        delete[] A_2;
    }
    
    delete[] dr3d;
    delete[] dk3d;
    fftw_destroy_plan(dr3d2dk3d);
    
    return 0;
}
