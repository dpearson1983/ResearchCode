/* bispec.h
 * David W. Pearson
 * October 24, 2017
 * 
 * This header contains all the functionality needed to calculate the bispectrum model via a slightly
 * modified version of the Scoccimarro et al. 2000 method. Instead of using a linear theory power spectrum,
 * a non-linear power spectrum is used to help better fit the non-linear features in the data. Additionally,
 * a redshift space distortion damping function is added to the model.
 * 
 * This is not as sophiticated as the model of Gil-Marin et al. 2015, but seems to adequetly model the 
 * bispectrum for the purposes of fitting to the BAO features. An object of this class will be a member of
 * the mcmc class along with an object of the powerspec class. Together, the two will enable the complete
 * calculation of the model.
 * 
 */


#ifndef _BISPEC_H_
#define _BISPEC_H_

#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cuda.h>
#include <vector_types.h>
#include <gpuerrchk.h>
#include "gaussquad32.h"
#include "file_check.h"
#include "make_spline.h"

#define TWOSEVENTHS 0.285714285714
#define THREESEVENTHS 0.428571428571
#define FOURSEVENTHS 0.571428571429
#define FIVESEVENTHS 0.714285714286
#define PI 3.1415926536

// Declare the needed device constants and track the constant memory usage 
__constant__ float4 d_Pk[128]; // Custom cubic spline for evaluation on the GPU (2048 bytes)
__constant__ float d_wi[32]; // Gaussian quadrature weights (128 bytes)
__constant__ float d_xi[32]; // Gaussian quadrature evaluation points (128 bytes)
__constant__ float d_p[14]; // Model parameters (56 bytes)
// Total constant memory used: 2360 / 65536 bytes


/* GPU function forward declarations */

// Function to evaluate the custom spline object on the GPU
__device__ float pk_spline_eval(float k);

// Function to calculate a single point on the 2D grid for the integration
__device__ double bispec_model(int x, float &phi, float3 k);

// Coordinate the calculation of the 2D integral for the bispectrum model on the GPU
__global__ void bispec_gauss_32(float3 *ks, double *Bk);

class bispec{
    size_t num_vals; // Keep track of the number of elements
    std::vector<double> B; // Storage for the bispectrum model values
    std::vector<float3> k; // The bispectrum is dependent on 3 k values
    
    // Reads the data file to determine the k triplets where the model needs to be evaluated. Also, the
    // data itself is temporarily stored in B so that it can be copied over to the mcmc object.
    void read_data_file(std::string in_file);
    
    public:
        // Default constructor
        bispec();
        
        // Constructor that automatically calls the initialize function to all the setup needed
        bispec(std::string input_data_file, std::string input_nonlin_file);
        
        // Calls the function to read in the data file and calls the functions to read in the non-linear
        // power spectrum file, create the custom spline object and then copy to the device constant memory
        void initialize(std::string input_data_file, std::string input_nonlin_file);
        
        // Calculates the model bispectrum and stores it in B. NOTE: This will overwrite the current
        // data in B.
        void calculate(std::vector<double> &pars, float3 *ks, double *Bk);
        
        // Returns the value stored at B[i]
        double get(int i);
        
        float3 getx(int i);
        
        // Return the number of values
        size_t size();
};

void bispec::read_data_file(std::string in_file) {
    if (check_file_exists(in_file)) {
        std::ifstream fin(in_file);
        while (!fin.eof()) {
            float3 kt;
            double bt, N;
            fin >> kt.x >> kt.y >> kt.z >> bt >> N;
            if (!fin.eof()) {
                bispec::k.push_back(kt);
                bispec::B.push_back(bt);
            }
        }
        fin.close();
        bispec::num_vals = bispec::B.size();
    }
}

bispec::bispec() {
    bispec::num_vals = 1;
}

bispec::bispec(std::string input_data_file, std::string input_nonlin_file) {
    bispec::initialize(input_data_file, input_nonlin_file);
}

void bispec::initialize(std::string input_data_file, std::string input_nonlin_file) {
    bispec::read_data_file(input_data_file);
    std::vector<float4> Pk_nonlin = make_spline(input_nonlin_file);
    gpuErrchk(cudaMemcpyToSymbol(d_Pk, Pk_nonlin.data(), 128*sizeof(float4)));
    gpuErrchk(cudaMemcpyToSymbol(d_wi, &w_i[0], 32*sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(d_xi, &x_i[0], 32*sizeof(float)));
}

void bispec::calculate(std::vector<double> &pars, float3 *ks, double *Bk) {
    std::vector<float> theta(pars.size());
    for (size_t i = 0; i < pars.size(); ++i)
        theta[i] = float(pars[i]);
    gpuErrchk(cudaMemcpyToSymbol(d_p, theta.data(), theta.size()*sizeof(float)));
    
    dim3 num_threads(32, 32);
    
    bispec_gauss_32<<<bispec::num_vals, num_threads>>>(ks, Bk);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(bispec::B.data(), Bk, bispec::num_vals*sizeof(double), 
                         cudaMemcpyDeviceToHost));
}

double bispec::get(int i) {
    return bispec::B[i];
}

float3 bispec::getx(int i) {
    return bispec::k[i];
}

size_t bispec::size() {
    return bispec::num_vals;
}

__device__ float pk_spline_eval(float k) {
    int i = (k - d_Pk[0].x)/(d_Pk[1].x - d_Pk[0].x);
    
    float Pk = (d_Pk[i + 1].z*(k - d_Pk[i].x)*(k - d_Pk[i].x)*(k - d_Pk[i].x))/(6.0*d_Pk[i].w)
              + (d_Pk[i].z*(d_Pk[i + 1].x - k)*(d_Pk[i + 1].x - k)*(d_Pk[i + 1].x - k))/(6.0*d_Pk[i].w)
              + (d_Pk[i + 1].y/d_Pk[i].w - (d_Pk[i + 1].z*d_Pk[i].w)/6.0)*(k - d_Pk[i].x)
              + (d_Pk[i].y/d_Pk[i].w - (d_Pk[i].w*d_Pk[i].z)/6.0)*(d_Pk[i + 1].x - k);
              
    return Pk;
}

__device__ double bispec_model(int x, float &phi, float3 k) {
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
    
    float P1 = pk_spline_eval(k1)/(d_p[4]*d_p[4]*d_p[3]);
    float P2 = pk_spline_eval(k2)/(d_p[4]*d_p[4]*d_p[3]);
    float P3 = pk_spline_eval(k3)/(d_p[4]*d_p[4]*d_p[3]);
    
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
    
    float Z2k12 = 0.5*d_p[1] + d_p[0]*(F12 + 0.5*mu12p*k12*(mu1/k1 + mu2/k2)) + d_p[2]*mu12p*mu12p*G12
                  + 0.5*d_p[2]*d_p[2]*mu12p*k12*mu1*mu2*(mu1/k1 + mu2/k2) 
                  + 0.5*(-FOURSEVENTHS*(d_p[0] - 1.0))*(mu12*mu12 - 1.0/3.0);
    float Z2k23 = 0.5*d_p[1] + d_p[0]*(F23 + 0.5*mu23p*k23*(mu2/k2 + mu3/k3)) + d_p[2]*mu23p*mu23p*G23
                  + 0.5*d_p[2]*d_p[2]*mu23p*k23*mu2*mu3*(mu2/k2 + mu3/k3)
                  + 0.5*(-FOURSEVENTHS*(d_p[0] - 1.0))*(mu23*mu23 - 1.0/3.0);
    float Z2k31 = 0.5*d_p[1] + d_p[0]*(F31 + 0.5*mu31p*k31*(mu3/k3 + mu1/k1)) + d_p[2]*mu31p*mu31p*G31
                  + 0.5*d_p[2]*d_p[2]*mu31p*k31*mu3*mu1*(mu3/k3 + mu1/k1)
                  + 0.5*(-FOURSEVENTHS*(d_p[0] - 1.0))*(mu31*mu31 - 1.0/3.0);
                  
    float den = 1.0 + 0.5*(k1*k1*mu1*mu1 + k2*k2*mu2*mu2 + k3*k3*mu3*mu3)*(k1*k1*mu1*mu1 + k2*k2*mu2*mu2 + k3*k3*mu3*mu3)*d_p[5]*d_p[5];
    float FoG = 1.0/(den*den);
    
    return 2.0*(Z2k12*Z1k1*Z1k2*P1*P2 + Z2k23*Z1k2*Z1k3*P2*P3 + Z2k31*Z1k3*Z1k1*P3*P1)*FoG;
}

// GPU kernel to calculate the bispectrum model. This kernel uses a fixed 32-point Gaussian quadrature
// and utilizes constant and shared memory to speed things up by about 220x compared to the previous
// version of the code while improving accuracy.
__global__ void bispec_gauss_32(float3 *ks, double *Bk) {
    int tid = threadIdx.y + blockDim.x*threadIdx.x; // Block local thread ID
    
    __shared__ double int_grid[1024]; 
    
    // Calculate the value for this thread
    float phi = PI*d_xi[threadIdx.y] + PI;
    int_grid[tid] = d_wi[threadIdx.x]*d_wi[threadIdx.y]*bispec_model(threadIdx.x, phi, ks[blockIdx.x]);
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

#endif
