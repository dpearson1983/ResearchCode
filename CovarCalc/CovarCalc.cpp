/* CovarCalc.cpp v2
 * David W. Pearson
 * 3/10/2016
 * 
 * This code will calculate the sample covariance given a number of data files. This code
 * is based on a previous piece of code I wrote to do the same thing, but has the advantage
 * of reading in values from a parameter file to prevent having to recompile every time
 * there is a minor change, such as the number of data files, or the base of those data
 * file names. To keep the code as general as possible, there are options to specify certain
 * file formats, such as having x-values and y-values in the file or y-values only. There
 * are also options to select from different output formats.
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cmath>

struct parameters{
    std::string inbase, ext, avgbase, covbase, corbase, xvals, outformat;
    int xbins, numMocks, startNum, datapoints, digits;
    double xmin, xmax, xbinWidth;
};

void assignParams(parameters *p, std::string name, std::string val) {
    if (name == "inbase") p[0].inbase = val;
    else if (name == "ext") p[0].ext = val;
    else if (name == "avgbase") p[0].avgbase = val;
    else if (name == "covbase") p[0].covbase = val;
    else if (name == "corbase") p[0].corbase = val;
    else if (name == "xbins") p[0].xbins = atof(val.c_str());
    else if (name == "numMocks") p[0].numMocks = atof(val.c_str());
    else if (name == "startNum") p[0].startNum = atof(val.c_str());
    else if (name == "datapoints") p[0].datapoints = atof(val.c_str());
    else if (name == "outformat") p[0].outformat = val;
    else if (name == "xvals") p[0].xvals = val;
    else if (name == "digits") p[0].digits = atof(val.c_str());
    else if (name == "xmin") p[0].xmin = atof(val.c_str());
    else if (name == "xbinWidth") p[0].xbinWidth = atof(val.c_str());
    else if (name == "xmax") p[0].xmax = atof(val.c_str());
    else {
        std::cout << "WARNING: Unrecognamized value in parameter file\n";
        std::cout << "    " << name << " is not a value parameter\n";
    }
}

parameters readParams(char *file) {
    std::ifstream fin;
    std::string name;
    std::string val;
    parameters p;
    
    fin.open(file, std::ios::in);
    while (!fin.eof()) {
        fin >> name >> val;
        assignParams(&p, name, val);
    }
    fin.close();
    
    return p;
}

std::string filename(std::string filebase, int filenum, int digits, std::string ext) {
    std::string file;
    
    std::stringstream ss;
    ss << filebase << std::setw(digits) << std::setfill('0') << filenum << ext;
    file = ss.str();
    
    return file;
}

int main(int argc, char *argv[]) {
    parameters p = readParams(argv[1]);
    p.xbinWidth = (p.xmax - p.xmin)/double(p.xbins);
    
    std::cout << "inbase        " << p.inbase << "\n";
    std::cout << "ext           " << p.ext << "\n";
    std::cout << "avgbase       " << p.avgbase << "\n";
    std::cout << "covbase       " << p.covbase << "\n";
    std::cout << "corbase       " << p.corbase << "\n";
    std::cout << "datapoints    " << p.datapoints << "\n";
    std::cout << "xvals         " << p.xvals << "\n";
    std::cout << "outformat     " << p.outformat << "\n";
    std::cout << "xmin          " << p.xmin << "\n";
    std::cout << "xbinWidth     " << p.xbinWidth << "\n";
    std::cout << "numMocks      " << p.numMocks << "\n";
    std::cout << "startNum      " << p.startNum << "\n";
    std::cout << "digits        " << p.digits << "\n";
    
    std::ifstream fin;
    std::ofstream fout;
    
    std::vector< double > mu(p.datapoints);
    std::vector< double > xvalues(p.datapoints);
    const int N = p.datapoints;
    double cov[N][N] = {0.0};
    
    std::cout << "Finding the average values...\n";
    for (int mock = p.startNum; mock <= p.numMocks; ++mock) {
        std::string infile = filename(p.inbase, mock, p.digits, p.ext);
        
        fin.open(infile.c_str(),std::ios::in);
        if (p.xvals == "true") {
            int i = 0;
            while (!fin.eof()) {
                double x, y;
                fin >> x >> y;
                if (x >= p.xmin && !fin.eof()) {
                    mu[i] += y/p.numMocks;
                    xvalues[i] = x;
                    ++i;
                }
            }
        } else {
            for (int i = 0; i < N; ++i) {
                double y;
                fin >> y;
                mu[i] += y/p.numMocks;
            }
        }
        fin.close();
    }
    std::cout << "\n";
    
    std::cout << "Calculating the sample covariance...\n";
    for (int mock = p.startNum; mock <= p.numMocks; ++mock) {
        std::string infile = filename(p.inbase, mock, p.digits, p.ext);
        std::vector< double > x;
        
        fin.open(infile.c_str(), std::ios::in);
        if (p.xvals == "true") {
            while (!fin.eof()) {
                double xval, y;
                fin >> xval >> y;
                if (xval >= p.xmin && !fin.eof()) {
                    x.push_back(y);
                }
            }
        } else {
            for (int i = 0; i < N; ++i) {
                double y;
                fin >> y;
                x.push_back(y);
            }
        }
        
        if (x.size() != p.datapoints) {
            std::cout << "ERROR: Not enough/too much data input!\n";
            return 0;
        }
        
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                cov[i][j] += ((x[i]-mu[i])*(x[j]-mu[j]))/(p.numMocks-1.0);
            }
        }
        fin.close();
    }
    std::cout << "\n";
    std::string avgout = filename(p.avgbase, p.numMocks, p.digits, p.ext);
    std::string covout = filename(p.covbase, p.numMocks, p.digits, p.ext);
    std::string corout = filename(p.corbase, p.numMocks, p.digits, p.ext);
    
    std::cout << "Outputting the average data to " << avgout << "\n";
    fout.open(avgout.c_str(),std::ios::out);
    fout.precision(15);
    if (p.xvals == "true") {
        for (int i = 0; i < N; ++i) {
            fout << xvalues[i] << " " << mu[i] << " " << sqrt(cov[i][i]) << "\n";
        }
    } else {
        for (int i = 0; i < N; ++i) {
            fout << mu[i] << " " << sqrt(cov[i][i]) << "\n";
        }
    }
    fout.close();
    
    std::cout << "Outputting the covariance matrix to " << covout << "\n";
    fout.open(covout.c_str(),std::ios::out);
    fout.precision(15);
    if (p.outformat == "square") {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                fout.width(35);
                fout << cov[i][j];
            }
            fout << "\n";
        }
    } else {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                fout << i << " " << j << " " << cov[i][j] << "\n";
            }
        }
    }
    fout.close();
    
    std::cout << "Outputting the correlation matrix to " << corout << "\n";
    fout.open(corout.c_str(),std::ios::out);
    fout.precision(15);
    if (p.outformat == "square") {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                fout.width(35);
                fout << cov[i][j]/sqrt(cov[i][i]*cov[j][j]);
            }
            fout << "\n";
        }
    } else {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                fout << i << " " << j << " " << cov[i][j]/sqrt(cov[i][i]*cov[j][j]) << "\n";
            }
        }
    }
    fout.close();
    
    return 0;
}
