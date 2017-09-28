#include <iostream>
#include <fstream>
#include <vector>
#include <vector_types.h>
#include <harppi.h>
#include "include/pk_slope.h"

int main(int argc, char *argv[]) {
    parameters p(argv[1]);
    p.print();
    
    std::vector<double4> n_spline;
    
    pkSlope<double> n(p.gets("in_pk_file"));
    
    n.calculate();
    
    n.get_device_spline(n_spline);
    
    std::ofstream fout(p.gets("out_file"));
    for (int i = 0; i < n_spline.size(); ++i) {
        fout << n_spline[i].x << " " << n_spline[i].y << " " << n_spline[i].z << " " << n_spline[i].w << "\n";
    }
    fout.close();
    
    return 0;
}
