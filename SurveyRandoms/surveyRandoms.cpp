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
    double *nz = new double[p.geti("numZBins")];
    double *red = new double[p.geti("numZBins") + 2];
#pragma omp parallel for
    for (int i = 0; i < npix; ++i) {
        mask[i] = 0;
    }
#pragma omp parallel for
    for (int i = 0; i < numZBins; ++i) {
        nz[i] = 0.0;
    }
    
    double mass_max = 0.0;
    double mass_min = 1000000.0;
    int numMockGals = 0;
    std::vector<std::vector<double>> masses(numZBins);
    std::vector<double> reds;
    //masses.reserve(numZBins);
    std::cout << "Reading in full mock and creating mask..." << std::endl;
    fin.open(p.gets("fullMock").c_str(), std::ios::in);
    while (!fin.eof()) {
        double ra, dec, redt, mass;
        fin >> ra >> dec >> redt >> mass;
        if (redt >= red_min && redt <= red_max && !fin.eof()) {
            ra *= pi/180.0;
            dec -= 90.0;
            dec = fabs(dec)*pi/180.0;
            
            long pix;
            ang2pix_nest(nside, dec, ra, &pix);
            mask[pix] = 1;
            
            int zbin = (redt - red_min)/dz;
            nz[zbin]++;
            numMockGals++;
            masses[zbin].push_back(log10(mass));
            reds.push_back(redt);
            if (mass < mass_min) mass_min = mass;
            if (mass > mass_max) mass_max = mass;
        }
    }
    fin.close();
    std::cout << std::endl;
    
    std::cout << "mass_min = " << mass_min << std::endl;
    std::cout << "mass_max = " << mass_max << std::endl;
    
    std::cout << "numMockGals = " << numMockGals << std::endl;
    int totalRans = timesRan*numMockGals;
    
//     std::cout << "Setting up spline for redshift probability..." << std::endl;
//     double sumNbar = 0.0;
//     red[0] = red_min;
//     red[numZBins + 1] = red_max;
//     gsl_integration_workspace *w = gsl_integration_workspace_alloc(10000000);
//     for (int i = 0; i < numZBins; ++i) {
//         nz[i] /= numMockGals;
//         red[i + 1] = red_min + (i + 0.5)*dz;
//         sumNbar += nz[i];
//     }
//     sumNbar += nz[numZBins - 1] + (nz[numZBins - 1] - nz[numZBins - 2])/2.0;
//     double *n_cdf = new double[numZBins + 2];
//     double accumulate = 0.0;
//     n_cdf[0] = 0.0;
//     n_cdf[numZBins + 1] = 1.0;
//     fout.open("NvszCDF.dat",std::ios::out);
//     fout.precision(15);
//     fout << red[0] << " " << n_cdf[0] << std::endl;
//     std::cout << red[0] << " " << n_cdf[0] << std::endl;
//     for (int i = 0; i < numZBins; ++i) {
//         nz[i] /= sumNbar;
//         accumulate += nz[i];
//         n_cdf[i + 1] = accumulate;
//         fout << red[i + 1] << " " << n_cdf[i + 1] << std::endl;
//         std::cout << red[i + 1] << " " << n_cdf[i + 1] << std::endl;
//     }
//     delete[] nz;
//     fout << red[numZBins + 1] << " " << n_cdf[numZBins + 1] << std::endl;
//     std::cout << red[numZBins + 1] << " " << n_cdf[numZBins + 1] << std::endl;
//     fout.close();
//     
//     gsl_spline *n_of_z = gsl_spline_alloc(gsl_interp_steffen, numZBins + 2);
//     gsl_interp_accel *acc_n_of_z = gsl_interp_accel_alloc();
//     gsl_spline_init(n_of_z, n_cdf, red, numZBins + 2);
//     
//     for (int i = 0; i < 10; ++i) {
//         double val = (i + 1.0)/10.0;
//         double result = gsl_spline_eval(n_of_z, val, acc_n_of_z);
//         std::cout << val << " " << result << std::endl;
//     }
    
//     std::cout << "Setting up splines for halo mass probability..." << std::endl;
//     
//     std::vector<gsl_spline *> massSplines;
//     std::vector<gsl_interp_accel *> accs;
//     std::vector<min_max> M_minmax;
//     M_minmax.reserve(numZBins);
//     massSplines.reserve(numZBins);
//     accs.reserve(numZBins);
//     fout.open("massBinInfo.dat", std::ios::out);
//     for (int i = 0; i < numZBins; ++i) {
//         std::vector<double> nm(p.geti("numMassBins") + 2);
//         std::vector<double> m(p.geti("numMassBins") + 2);
//         int numGalsZbin = masses[i].size();
//         auto min = std::min_element(std::begin(masses[i]), std::end(masses[i]));
//         auto max = std::max_element(std::begin(masses[i]), std::end(masses[i]));
//         min_max temp = {*min, *max};
//         M_minmax[i] = temp;
//         double dm = (*max - *min)/p.getd("numMassBins");
//         for (int j = 0; j < numGalsZbin; ++j) {
//             //if (masses[i][j] == *max) continue;
//             int bin = ((masses[i][j] - *min)/dm) + 1;
//             if (bin == p.geti("numMassBins") + 1) --bin;
//             nm[bin] += 1.0;
//         }
//         nm[0] = 0.0;
//         m[0] = *min;
//         m[p.geti("numMassBins") + 1] = *max;
//         double accum = 0.0;
//         nm[p.geti("numMassBins") + 1] = 0.01*nm[p.geti("numMassBins")];
//         if (nm[p.geti("numMassBins") + 1] < 0) {
//             std::cout << "Bad trend..." << std::endl;
//             std::cout << "nm[11] = " << nm[p.geti("numMassBins") + 1] << std::endl;
//         }
//         for (int j = 1; j <= p.geti("numMassBins"); ++j) {
//             if (nm[j] == 0) std::cout << "nm[" << j << "] = 0 for zbin = " << i << std::endl;
//             accum += nm[j];
//             nm[j] = accum;
//             m[j] = *min + (j - 0.5)*dm;
//         }
//         accum += nm[p.geti("numMassBins") + 1];
//         nm[p.geti("numMassBins") + 1] = accum;
//         for (int j = 0; j <= p.geti("numMassBins") + 1; ++j) {
//             nm[j] /= accum;
//             fout << m[j] << " " << nm[j] << "\n";
//         }
//         fout << "\n" << std::endl;
//         massSplines[i] = gsl_spline_alloc(gsl_interp_steffen, p.geti("numMassBins") + 2);
//         accs[i] = gsl_interp_accel_alloc();
//         gsl_spline_init(massSplines[i], &nm[0], &m[0], p.geti("numMassBins") + 2);
//     }
//     fout.close();
//     
//     fout.open("massStuff.dat", std::ios::out);
//     for (int i = 0; i < numZBins; ++i) {
//         fout << "m_min = " << M_minmax[i].min << "\n";
//         fout << "m_max = " << M_minmax[i].max << "\n";
//         for (int j = 0; j < 11; ++j) {
//             double x = j*0.1;
//             fout << x << " " << gsl_spline_eval(massSplines[i], x, accs[i]) << " ";
//             fout << pow(10.0,gsl_spline_eval(massSplines[i], x, accs[i])) << "\n";
//         }
//         fout << "\n\n";
//     }
//     fout.close();

    std::cout << "totalRans = " << totalRans << std::endl;
    std::cout << "file size = " << double(totalRans*sizeof(galaxyf))/1073741824.0 << " GiB" << std::endl;
    
    std::random_device seeder;
    std::mt19937_64 gen(seeder());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    long numRans = 0;
    bool moreRans = true;
    int numDraws = p.geti("numDraws");
    std::cout << "Generating randoms..." << std::endl;
    fout.open(p.gets("ransFile").c_str(), std::ios::out|std::ios::binary);
    while (moreRans) {
        std::vector<galaxyf> rans;
        for (int i = 0; i < numDraws; ++i) {
            double RA = 2.0*pi*dist(gen);
            double DEC = acos(2.0*dist(gen) - 1.0);
            long pix;
            ang2pix_nest(nside, DEC, RA, &pix);
            if (pix >= npix) std::cout << "pix = " << pix << " >= " << npix << std::endl;
            if (mask[pix] == 1) {
                galaxyf ran;
                ran.ra = RA*(180.0/pi);
                DEC *= (180.0/pi);
                ran.dec = fabs(DEC - 180.0) - 90.0;
                //double splineNum = dist(gen);
                //ran.red = gsl_spline_eval(n_of_z, splineNum, acc_n_of_z);
                int draw = int(dist(gen)*reds.size());
                if (draw == reds.size()) draw -= 1;
                ran.red = reds[draw];
                int zbin = (ran.red - red_min)/dz;
                if (ran.red > red_max || ran.red < red_min) {
                    std::cout << "ERROR: Invalid redshift." << std::endl;
                    std::cout << "    ran.red = " << ran.red << std::endl;
                    std::cout << "    red_max = " << red_max << std::endl;
//                     std::cout << "  splineNum = " << splineNum << std::endl;
                    continue;
                }
                if (ran.red == red_max) --zbin;
                //ran.bias = pow(10.0,gsl_spline_eval(massSplines[zbin], dist(gen), accs[zbin]));
                draw = int(dist(gen)*masses[zbin].size());
                if (draw == masses[zbin].size()) draw -= 1;
                ran.bias = pow(10.0,masses[zbin][draw]);
                ran.nbar = 0.0;
                rans.push_back(ran);
                numRans += 1;
            }
        }
        fout.write((char *) &rans[0], rans.size()*sizeof(galaxyf));
        if (numRans >= totalRans) moreRans = false;
        std::cout << "\r";
        std::cout.width(20);
        std::cout << numRans << "/";
        std::cout.width(20);
        std::cout << totalRans;
        std::cout.width(20);
        std::cout << 100.0*(double(numRans)/double(totalRans)) << "%";
        std::cout.flush();
    }
    fout.close();
//     gsl_integration_workspace_free(w);
    std::cout << std::endl;
    
    delete[] mask;
//     delete[] n_cdf;
    delete[] red;
    
//     for (int i = 0; i < numZBins; ++i) {
//         gsl_spline_free(massSplines[i]);
//         gsl_interp_accel_free(accs[i]);
//     }
//     
//     gsl_spline_free(n_of_z);
//     gsl_interp_accel_free(acc_n_of_z);
    
    return 0;
}
