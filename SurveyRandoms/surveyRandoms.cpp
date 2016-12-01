#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
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

int main(int argc, char *argv[]) {
    parameters p(argv[1]);
    p.print();
    
    std::ifstream fin;
    std::ofstream fout;
    
    long nside = p.geti("nside");
    long npix = nside2npix(nside);
    std::cout << "npix = " << npix << std::endl;
    std::cout << "Initializing mask..." << std::endl;
    int *mask = new int[npix];
#pragma omp parallel for
    for (int i = 0; i < npix; ++i) {
        mask[i] = 0;
    }
    
    std::cout << "Reading in full mock and creating mask..." << std::endl;
    fin.open(p.gets("fullMock").c_str(), std::ios::in);
    while (!fin.eof()) {
        double ra, dec, red, mass;
        fin >> ra >> dec >> red >> mass;
        
        ra *= pi/180.0;
        dec -= 90.0;
        dec = fabs(dec)*pi/180.0;
        
        long pix;
        ang2pix_nest(nside, dec, ra, &pix);
        mask[pix] = 1;
    }
    fin.close();
    
    std::cout << "Reading in number density profile..." << std::endl;
    std::vector<double> zin;
    std::vector<double> nin;
    fin.open(p.gets("nbarvszfile").c_str(), std::ios::in);
    while (!fin.eof()) {
        double z, n;
        fin >> z >> n;
        if (!fin.eof()) {
            nin.push_back(n);
            zin.push_back(z);
        }
    }
    fin.close();
    
    int numZbins = nin.size();
    double timesRan = p.getd("timesRan");
    double dz = zin[1] - zin[0];
    double red_min = p.getd("red_min");
    double red_max = p.getd("red_max");
    double Omega_M = p.getd("Omega_M");
    double Omega_L = p.getd("Omega_L");
    double sky_frac = p.getd("surveyArea")*pi/129600.0;
    
    std::cout << "Determining the number of randoms for each redshift bin..." << std::endl;
    int *numGalvsz = new int[nin.size()];
    int totalRans = 0;
#pragma omp parallel for
    for (int i = 0; i < numZbins; ++i) {
        gsl_integration_workspace *w = gsl_integration_workspace_alloc(10000000);
        double red_low = red_min + i*dz;
        double red_high = red_min + (i + 1.0)*dz;
        double r_low = rz(red_low, Omega_M, Omega_L, w);
        double r_high = rz(red_high, Omega_M, Omega_L, w);
        double V = (4.0*pi*(r_high*r_high*r_high - r_low*r_low*r_low))/3.0;
        numGalvsz[i] = V*sky_frac*nin[i]*timesRan;
        totalRans += numGalvsz[i];
        gsl_integration_workspace_free(w);
    }
//     std::string check;
    std::cout << "totalRans = " << totalRans << std::endl;
    std::cout << "file size = " << double(totalRans*sizeof(galaxy))/1073741824.0 << " GiB" << std::endl;
//     std::cout << "Continue? (y or n): ";
//     std::cin >> check;
//     if (check == "n") {
//         return 0;
//     }
    
    std::random_device seeder;
    std::mt19937_64 gen(seeder());
    std::uniform_real_distribution<double> xdist(p.getd("x_min"), p.getd("x_max"));
    std::uniform_real_distribution<double> ydist(p.getd("y_min"), p.getd("y_max"));
    std::uniform_real_distribution<double> zdist(p.getd("z_min"), p.getd("z_max"));
    
    std::vector<int> numGals(nin.size());
    long numRans = 0;
    bool moreRans = true;
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(10000000);
    std::cout << "Generating randoms..." << std::endl;
    fout.open(p.gets("ransFile").c_str(), std::ios::out|std::ios::binary);
    while (moreRans) {
        galaxy ran;
        ran.x = xdist(gen);
        ran.y = ydist(gen);
        ran.z = zdist(gen);
        cartesian2equatorial(&ran, 1, Omega_M, Omega_L, w);
        double phi = ran.ra*pi/180.0;
        double theta = fabs(ran.dec - 90.0)*pi/180.0;
        long pix;
        ang2pix_nest(nside, theta, phi, &pix);
        if (mask[pix] == 1 && ran.red <= red_max && ran.red >= red_min) {
            int zbin = (ran.red - red_min)/dz;
            if (numGals[zbin] < numGalvsz[zbin] && numRans < totalRans) {
                fout.write((char *) &ran, sizeof(galaxy));
                ++numGals[zbin];
                ++numRans;
            }
        }
        moreRans = false;
        for (int i = 0; i < numZbins; ++i) {
            if (numGals[i] != numGalvsz[i]) {
                moreRans = true;
                break;
            }
        }
        if (numRans == totalRans) moreRans = false;
        std::cout << numRans << "\r";
        std::cout.flush();
    }
    fout.close();
    gsl_integration_workspace_free(w);
    std::cout << std::endl;
    
    delete[] mask;
    delete[] numGalvsz;
    
    return 0;
}
