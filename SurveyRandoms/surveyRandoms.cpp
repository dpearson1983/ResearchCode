#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <vector>
#include <random>
#include <cmath>
#include <pods.h>
#include <harppi.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_integration.h>
#include <chealpix.h>
#include <pfunk.h>
#include <omp.h>

struct min_max{
    double min, max;
};

int main(int argc, char *argv[]) {
    parameters p(argv[1]);
    p.print();
    
    std::ifstream fin;
    std::ofstream fout;
    
    double timesRan = p.getd("timesRan");
    double red_min = p.getd("red_min");
    double red_max = p.getd("red_max");
    double Omega_M = p.getd("Omega_M");
    double Omega_L = p.getd("Omega_L");
    double sky_frac = p.getd("surveyArea")*pi/129600.0;
    int numZBins = p.geti("numZBins");
    double dz = (red_max - red_min)/p.getd("numZBins");
    int numThreads = p.geti("numThreads");
    
    long nside = p.geti("nside");
    long npix = nside2npix(nside);
    std::cout << "npix = " << npix << std::endl;
    std::cout << "Initializing mask..." << std::endl;
    int *mask = new int[npix];
    double *nz = new double[p.geti("numZBins") + 2];
    double *red = new double[p.geti("numZBins") + 2];
#pragma omp parallel for
    for (int i = 0; i < npix; ++i) {
        mask[i] = 0;
    }
#pragma omp parallel for
    for (int i = 0; i < numZBins; ++i) {
        nz[i] = 0.0;
    }
    
    int numMockGals = 0;
    std::vector<std::vector<double>> masses;
    masses.reserve(numZBins);
    std::cout << "Reading in full mock and creating mask..." << std::endl;
    fin.open(p.gets("fullMock").c_str(), std::ios::in);
    while (!fin.eof()) {
        double ra, dec, red, mass;
        fin >> ra >> dec >> red >> mass;
        if (red >= red_min && red <= red_max) {
            ra *= pi/180.0;
            dec -= 90.0;
            dec = fabs(dec)*pi/180.0;
            
            long pix;
            ang2pix_nest(nside, dec, ra, &pix);
            mask[pix] = 1;
            
            int zbin = (red - red_min)/dz;
            nz[zbin + 1]++;
            numMockGals++;
            masses[zbin].push_back(log10(mass));
        }
    }
    fin.close();
    
    std::cout << "numMockGals = " << numMockGals << std::endl;
    int numNeeded = timesRan*numMockGals;
    
    std::cout << "Setting up spline for redshift probability..." << std::endl;
    nz[0] = 1.0/numMockGals;
    nz[1] -= 1.0;
    red[0] = red_min;
    nz[numZBins + 1] = 1.0/numMockGals;
    nz[numZBins] -= 1.0
    red[numZBins + 1] = red_max;
    for (int i = 1; i <= numZBins; ++i) {
        nz[i] /= numMockGals;
        red[i] = (i + 0.5)*dz;
    }
    
    gsl_spline *n_of_z = gsl_spline_alloc(gsl_interp_cspline, numZBins + 2);
    gsl_interp_accel *acc_n_of_z = gsl_interp_accel_alloc();
    gsl_spline_init(n_of_z, red, nz, numZBins + 2);
    
    std::cout << "Setting up splines for halo mass probability..." << std::endl;
    
    std::vector<gsl_spline *> massSplines;
    std::vector<gsl_interp_accel *> accs;
    std::vector<min_max> M_minmax;
    M_minmax.reserve(numZBins);
    massSplines.reserve(numZBins);
    accs.reserve(numZBins);
    for (int i = 0; i < numZBins; ++i) {
        std::vector<double> nm(p.geti("numMassBins") + 2);
        std::vector<double> m(p.geti("numMassBins") + 2);
        int numGalsZbin = masses[i].size();
        auto min = std::min_element(std::begin(masses[i]), std::end(masses[i]));
        auto max = std::max_element(std::begin(masses[i]), std::end(masses[i]));
        min_max temp = {*min, *max};
        M_minmax[i] = temp;
        double dm = (*max - *min)/p.getd("numMassBins");
        for (int j = 0; j < numGalsZbin; ++j) {
            int bin = ((masses[i][j] - *min)/dm) + 1;
            nm[bin] += 1.0;
        }
        nm[0] = 1.0;
        m[0] = *min;
        nm[1] -= 1.0;
        nm[p.geti("numMassBins") + 1] = 1.0;
        m[p.geti("numMassBins") + 1] = *max;
        nm[p.geti("numMassBins")] -= 1.0;
        for (int j = 1; j <= p.geti("numMassBins"); ++j) {
            nm[j] = log10(nm[j]/double(numGalsZbin));
            m[j] = *min + (j + 0.5)*dm;
        }
        massSplines[i] = gsl_spline_alloc(gsl_interp_cspline, p.geti("numMassBins") + 2);
        accs[i] = gsl_interp_accel_alloc();
        gsl_spline_init(massSplines[i], &m[0], &nm[0], p.geti("numMassBins") + 2);
    }

    std::cout << "totalRans = " << totalRans << std::endl;
    std::cout << "file size = " << double(totalRans*sizeof(galaxy))/1073741824.0 << " GiB" << std::endl;
    
    std::random_device seeder;
    std::mt19937_64 gen(seeder());
    std::uniform_real_distribution<double> xdist(p.getd("x_min"), p.getd("x_max"));
    std::uniform_real_distribution<double> ydist(p.getd("y_min"), p.getd("y_max"));
    std::uniform_real_distribution<double> zdist(p.getd("z_min"), p.getd("z_max"));
    std::uniform_real_distribution<double> keep(0.0, 1.0);
    
    std::vector<int> numGals(nin.size());
    long numRans = 0;
    bool moreRans = true;
    int numDraws = p.geti("numDraws");
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(10000000);
    std::cout << "Generating randoms..." << std::endl;
    fout.open(p.gets("ransFile").c_str(), std::ios::out|std::ios::binary);
    while (moreRans) {
        std::vector<std::vector<galaxyf>> threadRans;
        #pragma omp parallel num_threads(numThreads) reduction(+:numRans)
        {
            int numAccepted = 0;
            std::vector<galaxy> rans;
            for (int i = 0; i < numDraws; ++i) {
                galaxy ran;
                ran.x = xdist(gen);
                ran.y = ydist(gen);
                ran.z = zdist(gen);
                cartesian2equatorial(&ran, 1, Omega_M, Omega_L, w);
                double phi = ran.ra*pi/180.0;
                double theta = fabs(ran.dec - 90.0)*pi/180.0;
                long pix;
                ang2pix_nest(nside, theta, phi, &pix);
                if (mask[pix] == 1 && ran.red >= red_min && ran.red <= red_max) {
                    // Determine probability of keeping galaxy based on number density profile
                    
                }
            }
            numRans += numAccepted;
        }
        for (int i = 0; i < numThreads; ++i)
            fout.write((char *) &threadRans[i][0], threadRans[i].size()*sizeof(galaxyf));
        
        if (numRans >= numNeeded) moreRans = false;
    }
    fout.close();
    gsl_integration_workspace_free(w);
    std::cout << std::endl;
    
    delete[] mask;
    delete[] numGalvsz;
    
    return 0;
}
