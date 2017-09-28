/* bkMCMC v17
 * David W. Pearson
 * September 28, 2017
 * 
 * This version of the code will implement some improvements to make the model better fit non-linear
 * features present in the data. The algorithm is effectively that of Gil-Marin 2012/2015.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cuda.h>
#include <vector_types.h>
#include <gpuerrchk.h>
#include "include/bkmcmc.h"
#include "include/hide_harppi.h"
#include "include/make_spline.h"
#include "include/pk_slope.h"

int main(int argc, char *argv[]) {
    // Use HARPPI hidden in an object file to parse parameters
    mcmc_parameters p(argv[1]);
    
    // Generate cubic splines of the input BAO and NW power spectra
    std::vector<float4> Pk = make_spline(p.input_power);
    std::vector<float4> n;
    pkSlope<float> nSpline(p.input_linear_nw_power);
    nSpline.calculate();
    nSpline.get_device_spline(n);
    
    // Copy the splines to the allocated GPU memory
    gpuErrchk(cudaMemcpyToSymbol(d_Pk, Pk.data(), 128*sizeof(float4)));
    gpuErrchk(cudaMemcpyToSymbol(d_n, n.data(), 1000*sizeof(float4)));
    
    // Copy Gaussian Quadrature weights and evaluation point to GPU constant memory
    gpuErrchk(cudaMemcpyToSymbol(d_wi, &w_i[0], 32*sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(d_xi, &x_i[0], 32*sizeof(float)));
    
    // Copy the fitting parameters from Gil-Marin 2012/2015 to GPU memory
    gpuErrchk(cudaMemcpyToSymbol(d_af, &a_f[0], 9*sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(d_ag, &a_g[0], 9*sizeof(float)));
    
    // Copy the values of sigma_8 and k_nl to the GPU constant memory
    float sig8 = (float)p.sigma8;
    float knl = (float)p.k_nl;
    gpuErrchk(cudaMemcpyToSymbol(d_sigma8, &sig8, sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(d_knl, &knl, sizeof(float)));
    
    // Declare a pointer for the integration workspace and allocate memory on the GPU
    double *d_Bk;
    float3 *d_ks;
    
    gpuErrchk(cudaMalloc((void **)&d_Bk, p.num_data*sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&d_ks, p.num_data*sizeof(float3)));
    
    // Initialize bkmcmc object
    bkmcmc bk_fit(p.data_file, p.cov_file, p.start_params, p.var_i, d_ks, d_Bk);
    
    // Check that the initialization worked
    bk_fit.check_init();
    
    // Set any limits on the parameters
    bk_fit.set_param_limits(p.limit_params, p.min, p.max);
    
    // Run the MCMC chain
    bk_fit.run_chain(p.num_draws, p.num_burn, p.reals_file, d_ks, d_Bk, p.new_chain);
    
    // Free device pointers
    gpuErrchk(cudaFree(d_Bk));
    gpuErrchk(cudaFree(d_ks));
    
    return 0;
}
