#ifndef _PODS_H_
#define _PODS_H_

const long double pi = acos(-1);

struct kinfo{
    double min, max, width;
};

struct double2{
    double x, y;
};

struct double3{
    double x, y, z;
};

struct int3{
    int x, y, z;
};

struct int4{
    int x, y, z, w;
};

struct int8{
    int cic1, cic2, cic3, cic4, cic5, cic6, cic7, cic8;
};

struct galaxy{
    double x, y, z, nbar, bias;
};

struct galaxies{
    double x, y, z, vx, vy, vz;
};

struct powerspec{
    double M, Q, QSN, H, HSN;
    int N;
};

struct Pk{
    double k, P;
};

#endif
