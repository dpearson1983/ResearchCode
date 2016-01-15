#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <cmath>

using namespace std;

const int numSpec = 20;
const int startNum = 1;
const double k_min = 0.0;
const double k_binwidth = 0.008;
const int kBins = 64;
const string inbase = "Power_SpectrumLRG";
const string outbase = "Avg_Pk_LRG_";
const string cbase = "Covariance_Matrix_";
const string rbase = "Reduced_Covariance_Matrix_";
const string ext = ".dat";

string filename(string filebase, int filenum, string fileext) {
    string file;
    
    stringstream ss;
    ss << filebase << setw(4) << setfill('0') << filenum << fileext;
    file = ss.str();
    
    return file;
}

int main() {
    ofstream fout;
    ifstream fin;
    
    double P_avg[kBins] = {0.0};
    double P2r_avg[kBins] = {0.0};
    double P2i_avg[kBins] = {0.0};
    double P2_avg[kBins] = {0.0};
    double P2P0_avg[kBins] = {0.0};
    double C[kBins][kBins] = {0.0};
    double C2r[kBins][kBins] = {0.0};
    double C2i[kBins][kBins] = {0.0};
    double C2[kBins][kBins] = {0.0};
    //double C4r[kBins][kBins] = {0.0};
    //double C4i[kBins][kBins] = {0.0};
    double C2C0[kBins][kBins] = {0.0};
    
    for (int spec = 0; spec < numSpec; spec++) {
        double P[kBins] = {0.0};
        double P2[kBins] = {0.0};
        
        string infile = filename(inbase, spec+startNum, ext);
        
        fin.open(infile.c_str(), ios::in);
        for (int i = 0; i < kBins; i++) {
            double k = 0.0;
            
            fin >> k >> P[i] >> P2[i]; // >> P2i[i] >> P2[i];
            P_avg[i] += P[i]/numSpec;
            //P2r_avg[i] += P2r[i]/numSpec;
            //P2i_avg[i] += P2i[i]/numSpec;
            P2_avg[i] += P2[i]/numSpec;
            if (P2[i] != 0) P2P0_avg[i] += (P[i]/P2[i])/numSpec;
        }
        fin.close();
    }
    
    for (int spec = 0; spec < numSpec; spec++) {
        double P[kBins] = {0.0};
        string P2r[kBins];
        string P2i[kBins];
        double P2[kBins] = {0.0};
        
        string infile = filename(inbase, spec+startNum, ext);
        
        fin.open(infile.c_str(), ios::in);
        for (int i = 0; i < kBins; i++) {
            double k = 0.0;
            
            fin >> k >> P[i] >> P2[i]; // >> P2i[i] >> P2[i];
        }
        fin.close();
        
        for (int i = 0; i < kBins; i++) {
            for (int j = 0; j < kBins; j++) {
                    C[i][j] += ((P[i]-P_avg[i])*(P[j]-P_avg[j]))/(numSpec-1);
                    //C2r[i][j] += ((P2r[i]-P2r_avg[i])*(P2r[j]-P2r_avg[j]))/(numSpec-1);
                    //C2i[i][j] += ((P2i[i]-P2i_avg[i])*(P2i[j]-P2i_avg[j]))/(numSpec-1);
                    C2[i][j] += ((P2[i]-P2_avg[i])*(P2[j]-P2_avg[j]))/(numSpec-1);
                    if (P2[i] != 0) {
                        C2C0[i][j] += ((P[i]/P2[i]-P2P0_avg[i])*(P[j]/P2[j]-P2P0_avg[j]))/(numSpec-1);
                    }
            }
        }
    }
    
    string outfile = filename(outbase, numSpec, ext);
    
    fout.open(outfile.c_str(),ios::out);
    for (int i = 0; i < kBins; i++) {
        double k = (i+0.5)*k_binwidth+k_min;
        fout.width(5);
        fout << k;
        fout.width(15);
        fout << P_avg[i];
        fout.width(15);
        fout << sqrt(C[i][i]);
//         fout.width(15);
//         fout << P2r_avg[i];
//         fout.width(15);
//         fout << sqrt(C2r[i][i]);
//         fout.width(15);
//         fout << P2i_avg[i];
//         fout.width(15);
//         fout << sqrt(C2i[i][i]);
        fout.width(15);
        fout << P2_avg[i];
        fout.width(15);
        fout << sqrt(C2[i][i]);
        fout.width(15);
        fout << P2P0_avg[i];
        fout.width(15);
        fout << sqrt(C2C0[i][i]) << "\n";
    }
    fout.close();
    
//     outfile = filename(cbase, numSpec, ext);
//     
//     fout.open(outfile.c_str(),ios::out);
//     for (int i = 0; i < kBins; i++) {
//         for (int j = 0; j < kBins; j++) {
//             fout.width(2);
//             fout << i+1;
//             fout.width(4);
//             fout << j+1;
//             fout.width(15);
//             fout << C[i][j];
//             fout.width(15);
//             fout << C2r[i][j];
//             fout.width(15);
//             fout << C2i[i][j] << "\n";
//         }
//     }
//     fout.close();
//     
//     outfile = filename(rbase, numSpec, ext);
//     
//     fout.open(outfile.c_str(),ios::out);
//     for (int i = 0; i < kBins; i++) {
//         for (int j = 0; j < kBins; j++) {
//             fout.width(2);
//             fout << i+1;
//             fout.width(4);
//             fout << j+1;
//             fout.width(15);
//             fout << C[i][j]/sqrt(C[i][i]*C[j][j]);
//             fout.width(15);
//             fout << C2r[i][j]/sqrt(C2r[i][i]*C2r[j][j]);
//             fout.width(15);
//             fout << C2i[i][j]/sqrt(C2i[i][i]*C2i[j][j]) << "\n";
//         }
//     }
//     fout.close();
    
    return 0;
}