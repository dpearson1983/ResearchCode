/* bkMCMC4.cu
 * David W. Pearson
 * July 3, 2017
 * 
 * This program makes use of the bkmcmc4.h header file in order to run the MCMC chains for fitting the
 * linear bispectrum model to input data. All of the needed functions are defined in the header file.
 * This program is just to set up the device pointers, and call the functions from the header in the 
 * appropriate order..
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cuda.h>
#include <vector_types.h>
#include <gpuerrchk.h>
#include "bkmcmc4.h"
#include "hide_harppi.h"
#include "make_spline.h"

int main(int argc, char *argv[]) {
    // Use HARPPI hidden in an object file to parse parameters
    mcmc_parameters p(argv[1]);
    
    // Generate cubic splines of the input BAO and NW power spectra
    std::vector<float4> Pk_bao = make_spline(p.input_bao_power);
    std::vector<float4> Pk_nw = make_spline(p.input_nw_power);
    
    // Copy the splines to the allocated GPU memory
    gpuErrchk(cudaMemcpyToSymbol(d_Pkbao, Pk_bao.data(), 128*sizeof(float4)));
    gpuErrchk(cudaMemcpyToSymbol(d_Pknw, Pk_nw.data(), 128*sizeof(float4)));
    
    gpuErrchk(cudaMemcpyToSymbol(d_wi, &w_i[0], 32*sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(d_xi, &x_i[0], 32*sizeof(float)));
    
    // Declare a pointer for the integration workspace and allocate memory on the GPU
    double *d_Bk;
    float3 *d_ks;
    
    gpuErrchk(cudaMalloc((void **)&d_Bk, 314*sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&d_ks, 314*sizeof(float3)));
    
    // Initialize bkmcmc object
    bkmcmc bk_fit(p.data_file, p.start_params, p.var_i, d_ks, d_Bk);
    
    // Check that the initialization worked
    bk_fit.check_init();
    
    // Set any limits on the parameters
    bk_fit.set_param_limits(p.limit_params, p.min, p.max);
    
    // Run the MCMC chain
    bk_fit.run_chain(p.num_draws, p.reals_file, d_ks, d_Bk, p.new_chain);
    
    // Free device pointers
    gpuErrchk(cudaFree(d_Bk));
    gpuErrchk(cudaFree(d_ks));
    
    return 0;
}
