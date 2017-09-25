#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include "include/harppi.h"
#include "include/bkmcmc.h"

int main(int argc, char *argv[]) {
    parameters p(argv[1]);
    p.print();
    
    bkmcmc bkFit(p);
    
    bkFit.run_chain();
    
    return 0;
}
