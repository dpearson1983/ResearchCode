#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>
#include <cmath>
#include <harppi.h>

std::string filename(std::string filebase, int digits, int filenum, std::string fileext) {
    std::stringstream ss;
    ss << filebase << std::setw(digits) << std::setfill('0') << filenum << fileext;
    return ss.str();
}

int main(int argc, char *argv[]) {
    if (argc == 1) {
        std::cout << "ERROR: Expected parameter file to be passed with program execution.\n";
        std::cout << "    Usage - " << argv[0] << " ParameterFileName.params\n";
        std::cout << "    ParameterFileName.params can have virtually any extension so long\n";
        std::cout << "    the file is a plain text file. Please rerun the program making\n";
        std::cout << "    sure to pass a parameter file name at runtime.";
        
        throw std::runtime_error("No parameter file.");
    }
    
    std::ifstream fin;
    std::ofstream fout;
    std::ofstream mout;
    
    parameters p(argv[1]);
    p.print();
    
    double Hz = sqrt(p.getd("Omega_M")*(1.0+p.getd("z"))*(1.0+p.getd("z"))*(1.0+p.getd("z"))
                     +p.getd("Omega_L"))*p.getd("H_0");
    double a = 1.0/(1.0+p.getd("z"));
    
    mout.open(p.gets("maxShiftFile").c_str(), std::ios::out);
    for (int mock = p.geti("startNum"); mock < p.geti("numMocks")+p.geti("startNum"); ++mock) {
        std::string infile = filename(p.gets("inBase"), p.geti("digits"), mock, p.gets("ext"));
        std::string outfile = filename(p.gets("outBase"), p.geti("digits"), mock, p.gets("ext"));
        
        std::cout << "Adding distortions to " << infile << "..." << std::endl;
        
        double maxShift = 0.0;
        
        fin.open(infile.c_str(), std::ios::in);
        fout.open(outfile.c_str(), std::ios::out);
        fout.precision(15);
        while (!fin.eof()) {
            double x, y, z, vx, vy, vz, b;
            
            fin >> x >> y >> z >> vx >> vy >> vz >> b;
            x += vx/(Hz);
            double shift = fabs(vx/(Hz));
            if (!fin.eof() && x >= 0 && x <= p.getd("Lx")) {
                fout << x << " " << y << " " << z << " " << b << " " << p.getd("nbar") << "\n";
                if (shift > maxShift) maxShift = shift;
            }
        }
        fin.close();
        fout.close();
        
        mout << maxShift << "\n";
    }
    mout.close();
    
    return 0;
}
