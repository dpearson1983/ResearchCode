#include <iostream>
#include <fstream>
#include <cuda.h>
#include <vector_types.h>
#include "include/gpuerrchk.h"
#include "include/hide_harppi.h"
#include "include/mcmc.h"

int main(int argc, char *argv[]) {
    mcmc_parameters p(argv[1]);
    
    // Allocate some memory on the GPU
    float3 *d_ks;
    gpuErrchk(cudaMalloc((void **)&d_ks, p.num_data*sizeof(float3)));
    double *d_Bk;
    gpuErrchk(cudaMalloc((void **)&d_Bk, p.num_data*sizeof(double)));
    
    // Initialize the mcmc object
    mcmc combFit(p, d_ks, d_Bk);
    
    combFit.write_Psi("inverse_covar.dat");
    
    // Run the MCMC chain
    combFit.run_chain(d_ks, d_Bk);
    
    // Free the GPU memory
    gpuErrchk(cudaFree(d_ks));
    gpuErrchk(cudaFree(d_Bk));
    
    return 0;
}
