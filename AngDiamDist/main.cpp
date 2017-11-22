#include <iostream>
#include <harppi.h>
#include "include/cosmology.h"

int main(int argc, char *argv[]) {
    parameters p(argv[1]);
    p.print();
    
    cosmology cosmo(p.getd("H_0"), p.getd("Omega_M"), p.getd("Omega_L"), p.getd("Omega_b"), 
                    p.getd("Omega_c"), p.getd("Tau"), p.getd("T_cmb"));
    
    std::cout.precision(15);
    std::cout << "D_A(z = " << p.getd("z") << ") = " << cosmo.D_A(p.getd("z")) << std::endl;
    std::cout << "D_V(z = " << p.getd("z") << ") = " << cosmo.D_V(p.getd("z")) << std::endl;
    
    return 0;
}
