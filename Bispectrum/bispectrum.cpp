/* bispectrum.cpp
 * David W. Pearson
 * December 15, 2016
 * 
 * This code will be an implementation of the bispectrum estimator of Baldauf et al. 2015. The steps of the 
 * implementation are:
 * 
 *      1. Construct a 3D grid to store all the values for B(k_i, k_j, k_l)
 *      2. At each grid point setup and perform three FFTs of shells around k_i, k_j, and k_l
 *      3. Loop over the grids in real space, taking the product of the shells and adding it
 *         to the value for B(k_i, k_j, k_l).
 *      4. Repeat for all the grid points.
 * 
 * It is claimed that this method can be faster than other for computing the bispectrum. In the future, this
 * may be tested explicitly by creating code that uses a different estimator and comparing the computation
 * times.
 * 
 * One potential advantage of this method is that each B(k_i, k_j, k_l) can be computed independently of all
 * others meaning that it should be trivial to parallelize the calculation, potentially leading to a significant
 * speed up.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <fftw3.h>
#include <harppi.h>
#include <bfunk.h>
#include <pfunk.h>
#include <pods.h>

int main(int argc, char *argv[]) {
    parameters p(argv[1]);
    p.print();
    
    std::ifstream fin;
    std::ofstream fout;
    
    int3 N = {p.geti("Nx"), p.geti("Ny"), p.geti("Nz")};
    int N_r = N.x*N.y*N.z;
    int N_k = N.x*N.y*(N.z/2 + 1);
    int N_p = N.x*N.y*2*(N.z/2 + 1);
    
    // Read in and grid the randoms
    
    
    // Read in and grid the galaxies
    
    // Compute delta(r)
    
    // Fourier transform delta(r) to get delta(k)
    
    // Setup B(k_i,k_j,k_l) grid
    int numKBins = p.geti("numKBins");
    double k_min = p.getd("k_min");
    double k_max = p.getd("k_max");
    double dk = (k_max - k_min)/p.getd("numKBins");
    std::vector<double> Bk;
    Bk.reserve(numKBins*numKBins*numKBins);
    
    // Arrays for in-place FFTs
    double *dk_i = new double[N_p];
    double *dk_j = new double[N_p];
    double *dk_l = new double[N_p];
    
    // Loop over the grid to compute the bispectrum
    for (int i = 0; i < numKBins; ++i) {
        double k_i = (i + 0.5)*dk;
        
        for (int j = i; j < numKBins; ++j) {
            double k_j = (j + 0.5)*dk;
            
            for (int l = j; l < numKBins; ++l) {
                double k_l = (l + 0.5)*dk;
    
        // Copy the appropriate values from delta(k) onto grids for k_i, k_j, k_l
    
        // Fourier transform the grids
    
        // Loop over the real space grids computing the product and adding it to the corresponding B(k_i,k_j,k_l)
    
    // Normalize the data
    
    // Output to file
    
    return 0;
}
