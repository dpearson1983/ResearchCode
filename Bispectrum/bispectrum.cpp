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
#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include <fftw3.h>
#include <omp.h>
#include <harppi.h>
#include <bfunk.h>
#include <pfunk.h>
#include <fileFuncs.h>
#include <pods.h>

struct in_gal{
    double x, y, z, nbar, bias;
};

struct bispec{
    double k1, k2, k3, val;
};

// This function will pull out the values for the different shell FFTs. The arguments are as follows:
//      1. dk3d - The Fourier transformed density field in an array from an in-place transform with FFTW3
//      2. dk3d_shell - Pointer to the memory to store the shell, same size as dk3d
//      3. kBins - An integer array the same size as dk3d that stores which output k bin the array location
//                 corresponds to, allowing for a single for loop to assign the values.
//      4. N_p - The number of values in dk3d, i.e. N_x*N_y*2*(N_z/2 + 1), where the N_i are the grid sizes
//               in the corresponding coordinate directions
//      5. kBin - The number of the k bin the shell corresponds to.
void get_shell(double *dk3d, double *dk3d_shell, int *kBins, int N_p, int kBin) {
    for (int i = 0; i < N_p; ++i) {
        if (kBins[i] == kBin) dk3d_shell[i] = dk3d[i];
        else dk3d_shell[i] = 0;
    }
}
    

int main(int argc, char *argv[]) {
    parameters p(argv[1]);
    p.print();
    
    std::ifstream fin;
    std::ofstream fout;
    
    // Setup some basic stuff
    std::cout << "Basic setup..." << std::endl;
    int3 N = {p.geti("Nx"), p.geti("Ny"), p.geti("Nz")};
    double3 L = {p.getd("Lx"), p.getd("Ly"), p.getd("Lz")};
    double3 dr = {L.x/double(N.x), L.y/double(N.y), L.z/double(N.z)};
    double3 r_obs = {p.getd("x_obs"), p.getd("y_obs"), p.getd("z_obs")};
    double3 dk = {(2.0*pi)/L.x, (2.0*pi)/L.y, (2.0*pi)/L.z};
    double V_f = dk.x*dk.y*dk.z;
    int N_r = N.x*N.y*N.z;
    int N_k = N.x*N.y*(N.z/2 + 1);
    int N_p = N.x*N.y*2*(N.z/2 + 1);
    
    // Determine the number of galaxies and randoms based on the input file sizes.
    std::cout << "Determining the number of galaxies and randoms..." << std::endl;
    long gal_in_size = filesize(p.gets("infile"));
    int numGals = gal_in_size/sizeof(in_gal);
    std::cout << "    numGals = " << numGals << std::endl;
    
    long ran_in_size = filesize(p.gets("ranfile"));
    int numRans = ran_in_size/sizeof(in_gal);
    std::cout << "    numRans = " << numRans << std::endl;
    
    // Read in and bin the randoms
    std::cout << "Reading in and binning randoms..." << std::endl;
    in_gal *in_rans = new in_gal[numRans];
    galaxy *rans = new galaxy[numRans];
    fin.open(p.gets("ranfile").c_str(), std::ios::in|std::ios::binary);
    fin.read((char *) in_rans, numRans*sizeof(in_gal));
    fin.close();
    for (int i = 0; i < numRans; ++i) {
        rans[i].x = in_rans[i].x;
        rans[i].y = in_rans[i].y;
        rans[i].z = in_rans[i].z;
        rans[i].nbar = in_rans[i].nbar;
        rans[i].bias = in_rans[i].bias;
        rans[i].ra = 0.0;
        rans[i].dec = 0.0;
        rans[i].red = 0.0;
    }
    delete[] in_rans;
    
    double *nden_ran = new double[N_r];
    double V = binNGP(rans, nden_ran, dr, numRans, N, r_obs);
    
    delete[] rans;
    
    std::cout << "Reading in and binning galaxies..." << std::endl;
    in_gal *in_gals = new in_gal[numGals];
    galaxy *gals = new galaxy[numGals];
    fin.open(p.gets("infile").c_str(), std::ios::in|std::ios::binary);
    fin.read((char *) in_gals, numGals*sizeof(in_gal));
    fin.close();
    for (int i = 0; i < numGals; ++i) {
        gals[i].x = in_gals[i].x;
        gals[i].y = in_gals[i].y;
        gals[i].z = in_gals[i].z;
        gals[i].nbar = in_gals[i].nbar;
        gals[i].bias = in_gals[i].bias;
        gals[i].ra = 0.0;
        gals[i].dec = 0.0;
        gals[i].red = 0.0;
    }
    delete[] in_gals;
    
    double *nden_gal = new double[N_r];
    V = binNGP(gals, nden_gal, dr, numGals, N, r_obs);
    
    delete[] gals;
    
    double nbar = double(numGals)/V;
    double alpha = double(numGals)/double(numRans);
    double shotnoise = double(numGals) + alpha*alpha*double(numRans);
    double gal_nbsqwsq = alpha*double(numRans)*nbar;
    std::cout << "V = " << V << std::endl;
    std::cout << "numGals^2 = " << double(numGals)*double(numGals) << std::endl;
    std::cout << "gal_nbsqwsq = " << gal_nbsqwsq << std::endl;
    
    // Setup for FFT of delta(r)
    std::cout << "Planning FFT..." << std::endl;
    fftw_init_threads();
    double *dr3d = new double[N_p];
    
    fftw_plan_with_nthreads(omp_get_max_threads());
    fftw_import_wisdom_from_filename("Wisdom.dat");
    fftw_plan dr3d2dk3d = fftw_plan_dft_r2c_3d(N.x, N.y, N.z, dr3d, (fftw_complex *)dr3d, FFTW_MEASURE);
    
    // Compute delta(r)
    std::cout << "Computing delta(r)..." << std::endl;
    std::vector<int> drs;
    for (int i = 0; i < N.x; ++i) {
        for (int j = 0; j < N.y; ++j) {
            for (int k = 0; k < N.z; ++k) {
                int index1 = k + N.z*(j + N.y*i);
                int index2 = k + 2*(N.z/2 + 1)*(j + N.y*i);
                drs.push_back(index2);
                dr3d[index2] = nden_gal[index1] - alpha*nden_ran[index1];
            }
        }
    }
    
    delete[] nden_gal;
    delete[] nden_ran;
    
    // Fourier transform delta(r) to get delta(k)
    std::cout << "Performing FFT..." << std::endl;
    fftw_execute(dr3d2dk3d);
    
//     displayFFT((fftw_complex *) dr3d);
//     
//     fout.open("dk3d.dat", std::ios::out);
//     fout.precision(15);
//     for (int i = 0; i < 1; ++i) {
//         for (int j = 0; j < N.y; ++j) {
//             for (int k = 0; k <= N.z/2; ++k) {
//                 fout << dr3d[(2*k    ) + 2*(N.z/2 + 1)*(j + N.y*i)] << " ";
//                 fout << dr3d[(2*k + 1) + 2*(N.z/2 + 1)*(j + N.y*i)] << "\n";
//             }
//         }
//     }
//     fout.close();
    
    // Setup B(k_i,k_j,k_l) grid
    int numKBins = p.geti("numKBins");
    double k_min = p.getd("k_min");
    double k_max = p.getd("k_max");
    double delta_k = (p.getd("k_max") - p.getd("k_min"))/p.getd("numKBins");
    std::vector<bispec> Bk;
    
    double *P_0 = new double[numKBins];
    int *N_0 = new int[numKBins];
    initArray(P_0, numKBins);
    initArray(N_0, numKBins);
    freqBinMono((fftw_complex *)dr3d, P_0, N_0, N, L, shotnoise, k_min, k_max, numKBins, true, 0);
    normalizePk(P_0, N_0, gal_nbsqwsq, numKBins);
    fout.open("Pk_gal.dat", std::ios::out);
    fout.precision(15);
    for (int i = 0; i < numKBins; ++i) {
        double k = k_min + (i + 0.5)*delta_k;
        fout << k << " " << P_0[i] << " " << N_0[i] << "\n";
    }
    fout.close();
    delete[] N_0;
    
    // Arrays and plans for in-place FFTs
    std::cout << "Planning more FFTs..." << std::endl;
    double *dk_i = new double[N_p];
    double *dk_j = new double[N_p];
    double *dk_l = new double[N_p];
    int *kBins = new int[N_p];
    
    fftw_plan dki2dri = fftw_plan_dft_c2r_3d(N.x, N.y, N.z, (fftw_complex *)dk_i, dk_i, FFTW_MEASURE);
    fftw_plan dkj2drj = fftw_plan_dft_c2r_3d(N.x, N.y, N.z, (fftw_complex *)dk_j, dk_j, FFTW_MEASURE);
    fftw_plan dkl2drl = fftw_plan_dft_c2r_3d(N.x, N.y, N.z, (fftw_complex *)dk_l, dk_l, FFTW_MEASURE);
    fftw_export_wisdom_to_filename("Wisdom.dat");
    
    // One time calculation of kBin for all grid points
    std::cout << "Mapping k bins..." << std::endl;
    for (int i = 0; i < N.x; ++i) {
        double kx = i*dk.x;
        for (int j = 0; j < N.y; ++j) {
            double ky = j*dk.y;
            for (int k = 0; k <= N.z/2; ++k) {
                double kz = k*dk.z;
                double k_tot = sqrt(kx*kx + ky*ky + kz*kz);
                int kBin = (k_tot - k_min)/delta_k;
                kBins[(2*k    ) + 2*(N.z/2 + 1)*(j + N.y*i)] = kBin;
                kBins[(2*k + 1) + 2*(N.z/2 + 1)*(j + N.y*i)] = kBin;
            }
        }
    }
    
    double coeff = 1.0/(8*pi*pi*pi*pi);
    double delta_k_cube = delta_k*delta_k*delta_k;
    
    // Loop over the grid to compute the bispectrum
    std::cout << "Calculating bispectrum..." << std::endl;
    double start = omp_get_wtime();
    for (int i = 0; i < numKBins; ++i) {
        double k_i = k_min + (i + 0.5)*delta_k;
        int k_iBin = (k_i - k_min)/delta_k;
        get_shell(dr3d, dk_i, kBins, N_p, k_iBin);
        fftw_execute(dki2dri);
        for (int j = i; j < numKBins; ++j) {
            double k_j = k_min + (j + 0.5)*delta_k;
            if (j != i) {
                int k_jBin = (k_j - k_min)/delta_k;
                get_shell(dr3d, dk_j, kBins, N_p, k_jBin);
                fftw_execute(dkj2drj);
            } else {
                #pragma omp parallel for
                for (int q = 0; q < N_p; ++q)
                    dk_j[q] = dk_i[q];
            }
            int stop = (k_i + k_j - k_min)/delta_k;
            stop = std::min(stop, numKBins);
            for (int l = j; l < stop; ++l) {
                double k_l = k_min + (l + 0.5)*delta_k;
                if (l != j) {
                    int k_lBin = (k_l - k_min)/delta_k;
                    get_shell(dr3d, dk_l, kBins, N_p, k_lBin);
                    fftw_execute(dkl2drl);
                } else {
                    #pragma omp parallel for
                    for (int q = 0; q < N_p; ++q)
                        dk_l[q] = dk_j[q];
                }
                
                bispec Btemp = {k_i, k_j, k_l, 0.0};
                double sum = 0.0;
                #pragma omp parallel for reduction(+:sum)
                for (int q = 0; q < N_r; ++q) {
                    sum += dk_i[drs[q]]*dk_j[drs[q]]*dk_l[drs[q]];
                }
                Btemp.val = sum - (P_0[i] + P_0[j] + P_0[l])/nbar - 1.0/(nbar*nbar);
                double V = V_f/(coeff*k_i*k_j*k_l*delta_k_cube);
                Btemp.val *= V;
                //std::cout << Btemp.k1 << ", " << Btemp.k2 << ", " << Btemp.k3 << ", " << Btemp.val << std::endl;
                Bk.push_back(Btemp);
            }
        }
    }
    std::cout << "Time to calculate bispectrum: " << omp_get_wtime() - start << " s" << std::endl;
    
    delete[] kBins;
    delete[] dk_i;
    delete[] dk_j;
    delete[] dk_l;
    delete[] dr3d;
    delete[] P_0;
    
    fftw_destroy_plan(dr3d2dk3d);
    fftw_destroy_plan(dki2dri);
    fftw_destroy_plan(dkj2drj);
    fftw_destroy_plan(dkl2drl);
    
    // Output to file
    double norm = double(N_r)*double(N_r)*double(N_r);
    std::cout << "Writing bispectrum to file..." << std::endl;
    int numOut = Bk.size();
    fout.open(p.gets("outfile").c_str(), std::ios::out);
    fout.precision(15);
    for (int i = 0; i < numOut; ++i) {
        fout << Bk[i].k1 << " " << Bk[i].k2 << " " << Bk[i].k3 << " " << Bk[i].val/norm << "\n";
    }
    fout.close();
    
    return 0;
}
