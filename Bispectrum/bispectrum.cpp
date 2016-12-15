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

int main(int argc, char *argv[]) {
    parameters p(argv[1]);
    p.print();
    
    std::ifstream fin;
    std::ofstream fout;
    
    // Read in and grid the randoms
    
    // Read in and grid the galaxies
    
    // Compute delta(r)
    
    // Fourier transform delta(r) to get delta(k)
    
    // Setup B(k_i,k_j,k_l) grid
    
    // Loop over the grid to compute the bispectrum
    
        // Copy the appropriate values from delta(k) onto grids for k_i, k_j, k_l
    
        // Fourier transform the grids
    
        // Loop over the real space grids computing the product and adding it to the corresponding B(k_i,k_j,k_l)
    
    // Normalize the data
    
    // Output to file
    
    return 0;
}
