/* bkmcmc3.h
 * David W. Pearson
 * May 17, 2017
 * 
 * This header is going to replace bkmcmc2.h and hopefully actually work.
 * 
 * Instead of storing things on the GPU everything except for the model calculation will be done host
 * side. The device pointers will be defined in bkMCMC3.cu and passed to the functions that need them.
 * This is not as desireable, but will prevent the awkward method of initializing the pointers inside
 * of the class initializer.
 */

#ifndef _BKMCMC8_H_
#define _BKMCMC8_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <cmath>
#include <cuda.h>
#include <vector_types.h>
#include <gpuerrchk.h>

#define TWOSEVENTHS 0.285714285714
#define THREESEVENTHS 0.428571428571
#define FOURSEVENTHS 0.571428571429
#define FIVESEVENTHS 0.714285714286
#define PI 3.1415926536

__constant__ float4 d_Pkbao[128]; //  2048 bytes
__constant__ float4 d_Pknw[128]; //   2048 bytes
__constant__ float d_wi[32]; //        128 bytes
__constant__ float d_xi[32]; //        128 bytes
__constant__ float d_p[13]; //          48 bytes
//           total constant memory:   4400 bytes out of 65536 bytes

const float w_i[] = {0.096540088514728, 0.096540088514728, 0.095638720079275, 0.095638720079275,
                     0.093844399080805, 0.093844399080805, 0.091173878695764, 0.091173878695764,
                     0.087652093004404, 0.087652093004404, 0.083311924226947, 0.083311924226947,
                     0.078193895787070, 0.078193895787070, 0.072345794108849, 0.072345794108849,
                     0.065822222776362, 0.065822222776362, 0.058684093478536, 0.058684093478536,
                     0.050998059262376, 0.050998059262376, 0.042835898022227, 0.042835898022227,
                     0.034273862913021, 0.034273862913021, 0.025392065309262, 0.025392065309262,
                     0.016274394730906, 0.016274394730906, 0.007018610009470, 0.007018610009470};

const float x_i[] = {-0.048307665687738, 0.048307665687738, -0.144471961582796, 0.144471961582796,
                     -0.239287362252137, 0.239287362252137, -0.331868602282128, 0.331868602282128,
                     -0.421351276130635, 0.421351276130635, -0.506899908932229, 0.506899908932229,
                     -0.587715757240762, 0.587715757240762, -0.663044266930215, 0.663044266930215,
                     -0.732182118740290, 0.732182118740290, -0.794483795967942, 0.794483795967942,
                     -0.849367613732570, 0.849367613732570, -0.896321155766052, 0.896321155766052,
                     -0.934906075937739, 0.934906075937739, -0.964762255587506, 0.964762255587506,
                     -0.985611511545268, 0.985611511545268, -0.997263861849481, 0.997263861849481};

std::random_device seeder;
std::mt19937_64 gen(seeder());
std::uniform_real_distribution<double> dist(-1.0, 1.0);

// Evaluates the spline to get the power spectrum at k
__device__ float spline_eval_nw(float k); // done

// Calculates a single element of the sum to get the bispectrum for a particular k triplet
__device__ double bispec_model_bao(int x, float &phi, float3 k); // done

// Enables the above bispec_model to be executed on many CUDA cores to speed up the integral calculation
__global__ void bispec_bao_32(float3 *ks, double *Bk);

// Evaluates the spline to get the power spectrum at k
__device__ float spline_eval_bao(float k); // done

// Calculates a single element of the sum to get the bispectrum for a particular k triplet
__device__ double bispec_model_nw(int x, float &phi, float3 k); // done

// Enables the above bispec_model to be executed on many CUDA cores to speed up the integral calculation
__global__ void bispec_bao_32_data(float3 *ks, double *Bk);

__global__ void calc_Bk_model(float3 *ks, double *Bk_bao, double *Bk_nw, double *Bk_mod);

class bkmcmc{
    int num_data, num_pars;
    std::vector<double> data, Bk, Psi; // These should have size of num_data
    std::vector<double> theta_0, theta_i, param_vars, min, max; // These should all have size of num_pars
    std::vector<float3> k; // This should have size of num_data
    std::vector<bool> limit_pars; // This should have size of num_pars
    double chisq_0, chisq_i;
    
    // Calculates the model bispectra (bao and nw) for the input parameters, pars.
    void model_calc(std::vector<double> &pars, float3 *ks, double *Bk_bao, double *Bk_nw, double *Bk_mod); // done
    
    // Sets the values of theta_i.
    void get_param_real(); // done
    
    // Calculates the chi^2 for the current proposal, theta_i
    double calc_chi_squared(); // done
    
    // Performs one MCMC trial. Returns true if proposal accepted, false otherwise
    bool trial(float3 *ks, double *Bk_bao, double *Bk_nw, double *Bk_mod, double &L, double &R); // done
    
    // Writes the current accepted parameters to the screen
    void write_theta_screen(); // done
    
    // Burns the requested number of parameter realizations to move to a higher likelihood region
    void burn_in(int num_burn, float3 *ks, double *Bk_bao, double *Bk_nw, double *Bk_mod); // done
    
    // Changes the initial guesses for the search range around parameters until acceptance = 0.234
    void tune_vars(float3 *ks, double *Bk_bao, double *Bk_nw, double *Bk_mod); // done
    
    public:
        // Initializes most of the data members and gets an initial chisq_0
        bkmcmc(std::string data_file, std::vector<double> &pars, std::vector<double> &vars, float3 *ks, 
               double *Bk_bao, double *Bk_nw, double *Bk_mod); // done
        
        // Displays information to the screen to check that the vectors are all the correct size
        void check_init(); // done
        
        // Sets which parameters should be limited and what the limits are
        void set_param_limits(std::vector<bool> &lim_pars, std::vector<double> &min_in,
                              std::vector<double> &max_in); // done
        
        // Runs the MCMC chain for num_draws realizations, writing to reals_file
        void run_chain(int num_draws, std::string reals_file, float3 *ks, double *Bk_bao, double *Bk_nw, 
                       double *Bk_mod, bool new_chain);
        
};

__device__ float spline_eval_bao(float k) {
    int i = (k - d_Pkbao[0].x)/(d_Pkbao[1].x - d_Pkbao[0].x);
    
    float Pk = (d_Pkbao[i + 1].z*(k - d_Pkbao[i].x)*(k - d_Pkbao[i].x)*(k - d_Pkbao[i].x))/(6.0*d_Pkbao[i].w)
              + (d_Pkbao[i].z*(d_Pkbao[i + 1].x - k)*(d_Pkbao[i + 1].x - k)*(d_Pkbao[i + 1].x - k))/(6.0*d_Pkbao[i].w)
              + (d_Pkbao[i + 1].y/d_Pkbao[i].w - (d_Pkbao[i + 1].z*d_Pkbao[i].w)/6.0)*(k - d_Pkbao[i].x)
              + (d_Pkbao[i].y/d_Pkbao[i].w - (d_Pkbao[i].w*d_Pkbao[i].z)/6.0)*(d_Pkbao[i + 1].x - k);
              
    return Pk;
}

__device__ float spline_eval_nw(float k) {
    int i = (k - d_Pknw[0].x)/(d_Pknw[1].x - d_Pknw[0].x);
    
    float Pk = (d_Pknw[i + 1].z*(k - d_Pknw[i].x)*(k - d_Pknw[i].x)*(k - d_Pknw[i].x))/(6.0*d_Pknw[i].w)
              + (d_Pknw[i].z*(d_Pknw[i + 1].x - k)*(d_Pknw[i + 1].x - k)*(d_Pknw[i + 1].x - k))/(6.0*d_Pknw[i].w)
              + (d_Pknw[i + 1].y/d_Pknw[i].w - (d_Pknw[i + 1].z*d_Pknw[i].w)/6.0)*(k - d_Pknw[i].x)
              + (d_Pknw[i].y/d_Pknw[i].w - (d_Pknw[i].w*d_Pknw[i].z)/6.0)*(d_Pknw[i + 1].x - k);
              
    return Pk;
}

__device__ double bispec_model_bao(int x, float &phi, float3 k) {
    // Calculate the mu's without the AP effects
    float z = (k.x*k.x + k.y*k.y - k.z*k.z)/(2.0*k.x*k.y);
    float mu1 = d_xi[x];
    float mu2 = -d_xi[x]*z + sqrtf(1.0 - d_xi[x]*d_xi[x])*sqrtf(1.0 - z*z)*cos(phi);
    float mu3 = -(mu1*k.x + mu2*k.y)/k.z;
    
    // It's convenient to store these quantities to reduce the number of FLOP's needed later
    float sq_ratio = (d_p[4]*d_p[4])/(d_p[3]*d_p[3]) - 1.0;
    float mu1bar = 1.0 + mu1*mu1*sq_ratio;
    float mu2bar = 1.0 + mu2*mu2*sq_ratio;
    float mu3bar = 1.0 + mu3*mu3*sq_ratio;
    
    // Convert the k's and mu's to include the AP effects
    float k1 = (k.x*sqrtf(mu1bar)/d_p[4]);
    float k2 = (k.y*sqrtf(mu2bar)/d_p[4]);
    float k3 = (k.z*sqrtf(mu3bar)/d_p[4]);
    
    float P1 = spline_eval_bao(k1)/(d_p[4]*d_p[4]*d_p[3]);
    float P2 = spline_eval_bao(k2)/(d_p[4]*d_p[4]*d_p[3]);
    float P3 = spline_eval_bao(k3)/(d_p[4]*d_p[4]*d_p[3]);
    
    mu1 = (mu1*d_p[4])/(d_p[3]*sqrt(mu1bar));
    mu2 = (mu2*d_p[4])/(d_p[3]*sqrt(mu2bar));
    mu3 = (mu3*d_p[4])/(d_p[3]*sqrt(mu3bar));
    
    // More convenient things to calculate before the long expressions
    float mu12 = -(k1*k1 + k2*k2 - k3*k3)/(2.0*k1*k2);
    float mu23 = -(k2*k2 + k3*k3 - k1*k1)/(2.0*k2*k3);
    float mu31 = -(k3*k3 + k1*k1 - k2*k2)/(2.0*k3*k1);
    
    float k12 = sqrt(k1*k1 + k2*k2 + 2.0*k1*k2*mu12);
    float k23 = sqrt(k2*k2 + k3*k3 + 2.0*k2*k3*mu23);
    float k31 = sqrt(k3*k3 + k1*k1 + 2.0*k3*k1*mu31);
    
    float mu12p = (k1*mu1 + k2*mu2)/k12;
    float mu23p = (k2*mu2 + k3*mu3)/k23;
    float mu31p = (k3*mu3 + k1*mu1)/k31;
    
    float Z1k1 = (d_p[0] + d_p[2]*mu1*mu1);
    float Z1k2 = (d_p[0] + d_p[2]*mu2*mu2);
    float Z1k3 = (d_p[0] + d_p[2]*mu3*mu3);
    
    float F12 = FIVESEVENTHS + 0.5*mu12*(k1/k2 + k2/k1) + TWOSEVENTHS*mu12*mu12;
    float F23 = FIVESEVENTHS + 0.5*mu23*(k2/k3 + k3/k2) + TWOSEVENTHS*mu23*mu23;
    float F31 = FIVESEVENTHS + 0.5*mu31*(k3/k1 + k1/k3) + TWOSEVENTHS*mu31*mu31;
    
    float G12 = THREESEVENTHS + 0.5*mu12*(k1/k2 + k2/k1) + FOURSEVENTHS*mu12*mu12;
    float G23 = THREESEVENTHS + 0.5*mu23*(k2/k3 + k3/k2) + FOURSEVENTHS*mu23*mu23;
    float G31 = THREESEVENTHS + 0.5*mu31*(k3/k1 + k1/k3) + FOURSEVENTHS*mu31*mu31;
    
    float Z2k12 = (0.5*d_p[1] + d_p[0]*F12 + d_p[2]*mu12p*mu12p*G12 + 
                   0.5*d_p[2]*mu12p*k12*((mu1*Z1k2)/k1 + (mu2*Z1k1)/k2));
    float Z2k23 = (0.5*d_p[1] + d_p[0]*F23 + d_p[2]*mu23p*mu23p*G23 + 
                   0.5*d_p[2]*mu23p*k23*((mu2*Z1k3)/k2 + (mu3*Z1k2)/k3));
    float Z2k31 = (0.5*d_p[1] + d_p[0]*F31 + d_p[2]*mu31p*mu31p*G31 + 
                   0.5*d_p[2]*mu31p*k31*((mu3*Z1k1)/k3 + (mu1*Z1k3)/k1));
    
    return 2.0*(Z2k12*Z1k1*Z1k2*P1*P2 + Z2k23*Z1k2*Z1k3*P2*P3 + Z2k31*Z1k3*Z1k1*P3*P1);
}

// double smooth = d_p[5]*k1*k1 + d_p[5]*k2*k2 + d_p[5]*k3*k3 
//                     + d_p[6]*k1*k2 + d_p[6]*k1*k3 + d_p[6]*k2*k3 
//                     + d_p[7]*k1 + d_p[7]*k2 + d_p[7]*k3
//                     + d_p[8] + d_p[9]/k1 + d_p[9]/k2 + d_p[9]/k3 + d_p[10]/(k1*k2) + d_p[10]/(k1*k3) 
//                     + d_p[10]/(k2*k3) + d_p[11]/(k1*k1) + d_p[11]/(k2*k2) + d_p[11]/(k3*k3);

// GPU kernel to calculate the bispectrum model. This kernel uses a fixed 32-point Gaussian quadrature
// and utilizes constant and shared memory to speed things up by about 220x compared to the previous
// version of the code while improving accuracy.
__global__ void bispec_bao_32(float3 *ks, double *Bk) {
    int tid = threadIdx.y + blockDim.x*threadIdx.x; // Block local thread ID
    
    __shared__ double int_grid[1024]; // Shared memory for all integral values: 8192 bytes
    
    // Calculate the value for this thread
    float phi = PI*d_xi[threadIdx.y] + PI;
    int_grid[tid] = d_wi[threadIdx.x]*d_wi[threadIdx.y]*bispec_model_bao(threadIdx.x, phi, ks[blockIdx.x]);
    __syncthreads();
    
    // First step of reduction done by 32 threads
    if (threadIdx.y == 0) {
        for (int i = 1; i < 32; ++i)
            int_grid[tid] += int_grid[tid + i];
    }
    __syncthreads();
    
    // Final reduction and writing result to global memory done only on first thread
    if (tid == 0) {
        for (int i = 1; i < 32; ++i)
            int_grid[0] += int_grid[blockDim.x*i];
        Bk[blockIdx.x] = int_grid[0]/4.0;
    }
}

__device__ double bispec_model_nw(int x, float &phi, float3 k) {
    // Calculate the mu's without the AP effects
    float z = (k.x*k.x + k.y*k.y - k.z*k.z)/(2.0*k.x*k.y);
    float mu1 = d_xi[x];
    float mu2 = -d_xi[x]*z + sqrtf(1.0 - d_xi[x]*d_xi[x])*sqrtf(1.0 - z*z)*cos(phi);
    float mu3 = -(mu1*k.x + mu2*k.y)/k.z;
    
    // It's convenient to store these quantities to reduce the number of FLOP's needed later
    float sq_ratio = (d_p[4]*d_p[4])/(d_p[3]*d_p[3]) - 1.0;
    float mu1bar = 1.0 + mu1*mu1*sq_ratio;
    float mu2bar = 1.0 + mu2*mu2*sq_ratio;
    float mu3bar = 1.0 + mu3*mu3*sq_ratio;
    
    // Convert the k's and mu's to include the AP effects
    float k1 = (k.x*sqrtf(mu1bar)/d_p[4]);
    float k2 = (k.y*sqrtf(mu2bar)/d_p[4]);
    float k3 = (k.z*sqrtf(mu3bar)/d_p[4]);
    
    float P1 = spline_eval_nw(k1)/(d_p[4]*d_p[4]*d_p[3]);
    float P2 = spline_eval_nw(k2)/(d_p[4]*d_p[4]*d_p[3]);
    float P3 = spline_eval_nw(k3)/(d_p[4]*d_p[4]*d_p[3]);
    
    mu1 = (mu1*d_p[4])/(d_p[3]*sqrt(mu1bar));
    mu2 = (mu2*d_p[4])/(d_p[3]*sqrt(mu2bar));
    mu3 = (mu3*d_p[4])/(d_p[3]*sqrt(mu3bar));
    
    // More convenient things to calculate before the long expressions
    float mu12 = -(k1*k1 + k2*k2 - k3*k3)/(2.0*k1*k2);
    float mu23 = -(k2*k2 + k3*k3 - k1*k1)/(2.0*k2*k3);
    float mu31 = -(k3*k3 + k1*k1 - k2*k2)/(2.0*k3*k1);
    
    float k12 = sqrt(k1*k1 + k2*k2 + 2.0*k1*k2*mu12);
    float k23 = sqrt(k2*k2 + k3*k3 + 2.0*k2*k3*mu23);
    float k31 = sqrt(k3*k3 + k1*k1 + 2.0*k3*k1*mu31);
    
    float mu12p = (k1*mu1 + k2*mu2)/k12;
    float mu23p = (k2*mu2 + k3*mu3)/k23;
    float mu31p = (k3*mu3 + k1*mu1)/k31;
    
    float Z1k1 = (d_p[0] + d_p[2]*mu1*mu1);
    float Z1k2 = (d_p[0] + d_p[2]*mu2*mu2);
    float Z1k3 = (d_p[0] + d_p[2]*mu3*mu3);
    
    float F12 = FIVESEVENTHS + 0.5*mu12*(k1/k2 + k2/k1) + TWOSEVENTHS*mu12*mu12;
    float F23 = FIVESEVENTHS + 0.5*mu23*(k2/k3 + k3/k2) + TWOSEVENTHS*mu23*mu23;
    float F31 = FIVESEVENTHS + 0.5*mu31*(k3/k1 + k1/k3) + TWOSEVENTHS*mu31*mu31;
    
    float G12 = THREESEVENTHS + 0.5*mu12*(k1/k2 + k2/k1) + FOURSEVENTHS*mu12*mu12;
    float G23 = THREESEVENTHS + 0.5*mu23*(k2/k3 + k3/k2) + FOURSEVENTHS*mu23*mu23;
    float G31 = THREESEVENTHS + 0.5*mu31*(k3/k1 + k1/k3) + FOURSEVENTHS*mu31*mu31;
    
    float Z2k12 = (0.5*d_p[1] + d_p[0]*F12 + d_p[2]*mu12p*mu12p*G12 + 
                   0.5*d_p[2]*mu12p*k12*((mu1*Z1k2)/k1 + (mu2*Z1k1)/k2));
    float Z2k23 = (0.5*d_p[1] + d_p[0]*F23 + d_p[2]*mu23p*mu23p*G23 + 
                   0.5*d_p[2]*mu23p*k23*((mu2*Z1k3)/k2 + (mu3*Z1k2)/k3));
    float Z2k31 = (0.5*d_p[1] + d_p[0]*F31 + d_p[2]*mu31p*mu31p*G31 + 
                   0.5*d_p[2]*mu31p*k31*((mu3*Z1k1)/k3 + (mu1*Z1k3)/k1));
    
    return 2.0*(Z2k12*Z1k1*Z1k2*P1*P2 + Z2k23*Z1k2*Z1k3*P2*P3 + Z2k31*Z1k3*Z1k1*P3*P1);
}

// GPU kernel to calculate the bispectrum model. This kernel uses a fixed 32-point Gaussian quadrature
// and utilizes constant and shared memory to speed things up by about 220x compared to the previous
// version of the code while improving accuracy.
__global__ void bispec_nw_32(float3 *ks, double *Bk) {
    int tid = threadIdx.y + blockDim.x*threadIdx.x; // Block local thread ID
    
    __shared__ double int_grid[1024]; // Shared memory for all integral values: 8192 bytes
    
    // Calculate the value for this thread
    float phi = PI*d_xi[threadIdx.y] + PI;
    int_grid[tid] = d_wi[threadIdx.x]*d_wi[threadIdx.y]*bispec_model_nw(threadIdx.x, phi, ks[blockIdx.x]);
    __syncthreads();
    
    // First step of reduction done by 32 threads
    if (threadIdx.y == 0) {
        for (int i = 1; i < 32; ++i)
            int_grid[tid] += int_grid[tid + i];
    }
    __syncthreads();
    
    // Final reduction and writing result to global memory done only on first thread
    if (tid == 0) {
        for (int i = 1; i < 32; ++i)
            int_grid[0] += int_grid[blockDim.x*i];
        Bk[blockIdx.x] = int_grid[0]/4.0;
    }
}

__global__ void calc_Bk_model(int N, float3 *ks, double *Bk_bao, double *Bk_nw, double *Bk_mod) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    if (tid < N) {
        double damp = exp(-0.5*d_p[12]*d_p[12]*(ks[tid].x*ks[tid].x + ks[tid].y*ks[tid].y 
                    + ks[tid].z*ks[tid].z));
        double broadband = d_p[5]*ks[tid].x*ks[tid].x + d_p[5]*ks[tid].y*ks[tid].y 
                         + d_p[5]*ks[tid].z*ks[tid].z + d_p[6]*ks[tid].x*ks[tid].y 
                         + d_p[6]*ks[tid].x*ks[tid].z + d_p[6]*ks[tid].y*ks[tid].z 
                         + d_p[7]*ks[tid].x + d_p[7]*ks[tid].y + d_p[7]*ks[tid].z
                         + d_p[8] + d_p[9]/ks[tid].x + d_p[9]/ks[tid].y + d_p[9]/ks[tid].z 
                         + d_p[10]/(ks[tid].x*ks[tid].y) + d_p[10]/(ks[tid].x*ks[tid].z) 
                         + d_p[10]/(ks[tid].y*ks[tid].z) + d_p[11]/(ks[tid].x*ks[tid].x) 
                         + d_p[11]/(ks[tid].y*ks[tid].y) + d_p[11]/(ks[tid].z*ks[tid].z);
        
        Bk_mod[tid] = Bk_nw[tid]*(1.0 + (Bk_bao[tid]/Bk_nw[tid] - 1.0)*damp) + broadband;
    }
}

void bkmcmc::model_calc(std::vector<double> &pars, float3 *ks, double *Bk_bao, double *Bk_nw, 
                        double *Bk_mod) {
    std::vector<float> theta(bkmcmc::num_pars);
    for (int i = 0; i < bkmcmc::num_pars; ++i)
        theta[i] = float(pars[i]);
    gpuErrchk(cudaMemcpyToSymbol(d_p, theta.data(), bkmcmc::num_pars*sizeof(float)));
    
    dim3 num_threads(32,32);
    int num_blocks = ceil(bkmcmc::num_data/1024.0);
    
    bispec_bao_32<<<bkmcmc::num_data, num_threads>>>(ks, Bk_bao);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    bispec_nw_32<<<bkmcmc::num_data, num_threads>>>(ks, Bk_nw);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    calc_Bk_model<<<num_blocks, 1024>>>(bkmcmc::num_data, ks, Bk_bao, Bk_nw, Bk_mod);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    gpuErrchk(cudaMemcpy(bkmcmc::Bk.data(), Bk_mod, bkmcmc::num_data*sizeof(double), 
                         cudaMemcpyDeviceToHost));
}

void bkmcmc::get_param_real() {
    for (int i = 0; i < bkmcmc::num_pars; ++i) {
        if (bkmcmc::limit_pars[i]) {
            if (bkmcmc::theta_0[i] + bkmcmc::param_vars[i] > bkmcmc::max[i]) {
                double center = bkmcmc::max[i] - bkmcmc::param_vars[i];
                bkmcmc::theta_i[i] = center + dist(gen)*bkmcmc::param_vars[i];
            } else if (bkmcmc::theta_0[i] - bkmcmc::param_vars[i] < bkmcmc::min[i]) {
                double center = bkmcmc::min[i] + bkmcmc::param_vars[i];
                bkmcmc::theta_i[i] = center + dist(gen)*bkmcmc::param_vars[i];
            } else {
                bkmcmc::theta_i[i] = bkmcmc::theta_0[i] + dist(gen)*bkmcmc::param_vars[i];
            }
        } else {
            bkmcmc::theta_i[i] = bkmcmc::theta_0[i] + dist(gen)*bkmcmc::param_vars[i];
        }
    }
}

double bkmcmc::calc_chi_squared() {
    double chisq = 0.0;
    for (int i = 0; i < bkmcmc::num_data; ++i) {
        chisq += (bkmcmc::data[i] - bkmcmc::Bk[i])*Psi[i]*(bkmcmc::data[i] - bkmcmc::Bk[i]);
    }
    return chisq;
}

bool bkmcmc::trial(float3 *ks, double *Bk_bao, double *Bk_nw, double *Bk_mod, double &L, double &R) {
    bkmcmc::get_param_real();
    bkmcmc::model_calc(bkmcmc::theta_i, ks, Bk_bao, Bk_nw, Bk_mod);
    bkmcmc::chisq_i = bkmcmc::calc_chi_squared();
    
    L = exp(0.5*(bkmcmc::chisq_0 - bkmcmc::chisq_i));
    R = (dist(gen) + 1.0)/2.0;
    
    if (L > R) {
        for (int i = 0; i < bkmcmc::num_pars; ++i)
            bkmcmc::theta_0[i] = bkmcmc::theta_i[i];
        bkmcmc::chisq_0 = bkmcmc::chisq_i;
        return true;
    } else {
        return false;
    }
}

void bkmcmc::write_theta_screen() {
    std::cout.precision(6);
    for (int i = 0; i < 5; ++i) {
        std::cout.width(15);
        std::cout << bkmcmc::theta_0[i];
    }
    std::cout.width(15);
    std::cout << bkmcmc::chisq_0;
    std::cout.flush();
}

void bkmcmc::burn_in(int num_burn, float3 *ks, double *Bk_bao, double *Bk_nw, double *Bk_mod) {
    std::cout << "Buring the first " << num_burn << " trials to move to higher likelihood..." << std::endl;
    double L, R;
    for (int i = 0; i < num_burn; ++i) {
        bool move = bkmcmc::trial(ks, Bk_bao, Bk_nw, Bk_mod, L, R);
        if (move) {
            std::cout << "\r";
            std::cout.width(5);
            std::cout << i;
            bkmcmc::write_theta_screen();
            std::cout.width(15);
            std::cout << L;
            std::cout.width(15);
            std::cout << R;
            std::cout.flush();
        }
    }
    std::cout << std::endl;
}

void bkmcmc::tune_vars(float3 *ks, double *Bk_bao, double *Bk_nw, double *Bk_mod) {
    std::cout << "Tuning acceptance ratio..." << std::endl;
    double acceptance = 0.0;
    while (acceptance <= 0.233 || acceptance >= 0.235) {
        int accept = 0;
        double L, R;
        for (int i = 0; i < 10000; ++i) {
            bool move = bkmcmc::trial(ks, Bk_bao, Bk_nw, Bk_mod, L, R);
            if (move) {
                std::cout << "\r";
                bkmcmc::write_theta_screen();
                accept++;
            }
        }
        std::cout << std::endl;
        acceptance = double(accept)/10000.0;
        
        if (acceptance <= 0.233) {
            for (int i = 0; i < bkmcmc::num_pars; ++i)
                bkmcmc::param_vars[i] *= 0.99;
        }
        if (acceptance >= 0.235) {
            for (int i = 0; i < bkmcmc::num_pars; ++i)
                bkmcmc::param_vars[i] *= 1.01;
        }
        std::cout << "acceptance = " << acceptance << std::endl;
    }
    std::ofstream fout;
    fout.open("variances.dat", std::ios::out);
    for (int i = 0; i < bkmcmc::num_pars; ++i)
        fout << bkmcmc::param_vars[i] << " ";
    fout << "\n";
    fout.close();
}

bkmcmc::bkmcmc(std::string data_file, std::vector<double> &pars, std::vector<double> &vars, float3 *ks, 
               double *Bk_bao, double *Bk_nw, double *Bk_mod) {
    std::ifstream fin;
    std::ofstream fout;
    
    std::cout << "Reading in and storing data file..." << std::endl;
    fin.open(data_file.c_str(), std::ios::in);
    while (!fin.eof()) {
        float3 kt;
        double B, var;
        fin >> kt.x >> kt.y >> kt.z >> B >> var;
        if (!fin.eof()) {
            bkmcmc::k.push_back(kt);
            bkmcmc::data.push_back(B);
            bkmcmc::Psi.push_back(1.0/var);
            bkmcmc::Bk.push_back(0.0);
        }
    }
    fin.close();
    
    bkmcmc::num_data = bkmcmc::data.size();
    std::cout << "num_data = " << bkmcmc::num_data << std::endl;
    
    
    gpuErrchk(cudaMemcpy(ks, bkmcmc::k.data(), bkmcmc::num_data*sizeof(float3), cudaMemcpyHostToDevice));
    
    gpuErrchk(cudaMemcpy(Bk_mod, bkmcmc::Bk.data(), bkmcmc::num_data*sizeof(double), 
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(Bk_bao, bkmcmc::Bk.data(), bkmcmc::num_data*sizeof(double), 
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(Bk_nw, bkmcmc::Bk.data(), bkmcmc::num_data*sizeof(double), 
                         cudaMemcpyHostToDevice));
    
    bkmcmc::num_pars = pars.size();
    std::cout << "num_pars = " << bkmcmc::num_pars << std::endl;
    
    for (int i = 0; i < bkmcmc::num_pars; ++i) {
        bkmcmc::theta_0.push_back(pars[i]);
        bkmcmc::theta_i.push_back(0.0);
        bkmcmc::limit_pars.push_back(false);
        bkmcmc::max.push_back(0.0);
        bkmcmc::min.push_back(0.0);
        bkmcmc::param_vars.push_back(vars[i]);
    }
    
    std::cout << "Calculating initial model and chi^2..." << std::endl;
    bkmcmc::model_calc(bkmcmc::theta_0, ks, Bk_bao, Bk_nw, Bk_mod);
    bkmcmc::chisq_0 = bkmcmc::calc_chi_squared();
    
    fout.open("Bk_mod_check.dat", std::ios::out);
    for (int i =0; i < bkmcmc::num_data; ++i) {
        fout.precision(3);
        fout << bkmcmc::k[i].x << " " << bkmcmc::k[i].y << " " << bkmcmc::k[i].z << " ";
        fout.precision(15);
        fout << bkmcmc::data[i] << " " << bkmcmc::Bk[i] << "\n";
    }
    fout.close();
}

void bkmcmc::check_init() {
    std::cout << "Number of data points: " << bkmcmc::num_data << std::endl;
    std::cout << "    data.size()      = " << bkmcmc::data.size() << std::endl;
    std::cout << "    Bk.size()        = " << bkmcmc::Bk.size() << std::endl;
    std::cout << "    Psi.size()       = " << bkmcmc::Psi.size() << std::endl;
    std::cout << "Number of parameters:  " << bkmcmc::num_pars << std::endl;
    std::cout << "    theta_0.size()   = " << bkmcmc::theta_0.size() << std::endl;
    std::cout << "    theta_i.size()   = " << bkmcmc::theta_i.size() << std::endl;
    std::cout << "    limit_pars.size()= " << bkmcmc::limit_pars.size() << std::endl;
    std::cout << "    min.size()       = " << bkmcmc::min.size() << std::endl;
    std::cout << "    max.size()       = " << bkmcmc::max.size() << std::endl;
    std::cout << "    param_vars.size()= " << bkmcmc::param_vars.size() << std::endl;
}

void bkmcmc::set_param_limits(std::vector<bool> &lim_pars, std::vector<double> &min_in,
                              std::vector<double> &max_in) {
    for (int i = 0; i < bkmcmc::num_pars; ++i) {
        bkmcmc::limit_pars[i] = lim_pars[i];
        bkmcmc::max[i] = max_in[i];
        bkmcmc::min[i] = min_in[i];
    }
}

void bkmcmc::run_chain(int num_draws, std::string reals_file, float3 *ks, double *Bk_bao, double *Bk_nw,
                       double *Bk_mod, bool new_chain) {
    int num_old_rels = 0;
    if (new_chain) {
        std::cout << "Starting new chain..." << std::endl;
        bkmcmc::burn_in(10000, ks, Bk_bao, Bk_nw, Bk_mod);
        bkmcmc::tune_vars(ks, Bk_bao, Bk_nw, Bk_mod);
    } else {
        std::cout << "Resuming previous chain..." << std::endl;
        std::ifstream fin;
        fin.open("variances.dat", std::ios::in);
        for (int i = 0; i < bkmcmc::num_pars; ++i) {
            double var;
            fin >> var;
            bkmcmc::param_vars[i] = var;
        }
        fin.close();
        fin.open(reals_file.c_str(), std::ios::in);
        while (!fin.eof()) {
            num_old_rels++;
            for (int i = 0; i < bkmcmc::num_pars; ++i)
                fin >> bkmcmc::theta_0[i];
            fin >> bkmcmc::chisq_0;
        }
        fin.close();
        num_old_rels--;
    }
    
    std::ofstream fout;
    double L, R;
    fout.open(reals_file.c_str(), std::ios::app);
    fout.precision(15);
    for (int i = 0; i < num_draws; ++i) {
        bool move = bkmcmc::trial(ks, Bk_bao, Bk_nw, Bk_mod, L, R);
        for (int par = 0; par < bkmcmc::num_pars; ++par) {
            fout << bkmcmc::theta_0[par] << " ";
        }
        fout << bkmcmc::chisq_0 << "\n";
        if (move) {
            std::cout << "\r";
            std::cout.width(15);
            std::cout << i + num_old_rels;
            bkmcmc::write_theta_screen();
        }
    }
    std::cout << std::endl;
    fout.close();
}
    
#endif
