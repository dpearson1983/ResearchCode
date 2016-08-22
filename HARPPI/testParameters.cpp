#include <iostream>
#include <harppi.h>

int main(int argc, char *argv[]) {
    if (argc == 1) {
        std::cout << "ERROR: No parameter file specified.\n";
        return 0;
    }
    
    parameters p;
    
    p.readParams(argv[1]);
    
    std::vector<typekey> minParams(5);
    minParams[0] = {"double", "bias"};
    minParams[1] = {"double", "growth"};
    minParams[2] = {"int", "numGals"};
    minParams[3] = {"string", "inFile"};
    minParams[4] = {"string", "outFile"};
    
    p.check_min(minParams);
    p.print();
    
    std::cout << "The input galaxy bias is: " << p.getd("bias") << std::endl;
    std::cout << "The input growth factor is: " << p.getd("growth") << std::endl;
    std::cout << "The input number of galaxies is: " << p.geti("numGals") << std::endl;
    
    if (p.checkParam("test")) {
        std::cout << "There is a parameter named test in the file." << std::endl;
    }
    
    double mono = p.getd("bias")*p.getd("bias") 
                  + (2.0/3.0)*p.getd("growth")*p.getd("bias")
                  + (1.0/5.0)*p.getd("growth")*p.getd("growth");
    double quad = (4.0/3.0)*p.getd("bias")*p.getd("growth")
                  + (4.0/7.0)*p.getd("growth")*p.getd("growth");
    
    std::cout << "Monopole scaling = " << mono << std::endl;
    std::cout << "Quadrupole scaling = " << quad << std::endl;
    
    std::cout << "The test parameter file contains a vector of 3 doubles, they are:\n";
    for (int i = 0; i < 3; ++i)
        std::cout << p.getd("bs", i) << std::endl;
    
    std::cout << p.gets("outFile") << std::endl;
    std::cout << "The bias as an integer is: " << p.geti("bias") << std::endl;
    std::cout << "This should throw an error: " << p.geti("inFile") << std::endl;
    
    return 0;
}
