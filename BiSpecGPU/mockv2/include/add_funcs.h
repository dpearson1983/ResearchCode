#ifndef _ADD_FUNCS_H_
#define _ADD_FUNCS_H_

#include <vector>
#include <constants.h>

std::vector<double> fftfreq(int N, double L) {
    double dk = (2.0*pi)/L;
    std::vector<double> k;
    k.reserve(N);
    for (int i = 0; i <= N/2; ++i)
        k.push_back(i*dk);
    for (int i = N/2 + 1; i < N; ++i)
        k.push_back((i - N)*dk);
    return k;
}

std::vector<double> myfreq(int N, double L) {
    double dk = (2.0*pi)/L;
    std::vector<double> k;
    k.reserve(N);
    for (int i = 0; i < N; ++i)
        k.push_back((int(i - N/2))*dk);
    return k;
}

double gridCorCIC(vec3<double> k, vec3<double> dr) {
    double sincx = sin(0.5*k.x*dr.x + 1E-17)/(0.5*k.x*dr.x + 1E-17);
    double sincy = sin(0.5*k.y*dr.y + 1E-17)/(0.5*k.y*dr.y + 1E-17);
    double sincz = sin(0.5*k.z*dr.z + 1E-17)/(0.5*k.z*dr.z + 1E-17);
    double prodsinc = sincx*sincy*sincz;
    
    return 1.0/(prodsinc*prodsinc);
}

int kMatch(double ks, std::vector<double> &kb, double L) {
    double dk = (2.0*pi)/L;
    int i = 0;
    bool found = false;
    int N = kb.size();
    int index1 = floor(ks/dk + 0.5);
    for (int j = 0; j < N; ++j) {
        int index2 = floor(kb[j]/dk + 0.5);
        if (index2 == index1) {
            i = j;
            found = true;
        }
    }
    
    if (found) return i;
    
    return -10000;
}

std::string filename(std::string base, int digits, int num, std::string ext) {
    std::stringstream file;
    file << base << std::setw(digits) << std::setfill('0') << num << ext;
    return file.str();
}
