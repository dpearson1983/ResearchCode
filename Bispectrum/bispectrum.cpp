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
