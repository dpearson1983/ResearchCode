#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cmath>
#include <random>
#include <harppi.h>

std::string filename(std::string filebase, int digits, int filenum, std::string fileext) {
    std::stringstream file;
    file << filebase << std::setw(digits) << std::setfill('0') << filenum << fileext;
    return file.str();
}

int main(int argc, char *argv[]) {
    parameters p;
    p.readParams(argv[1]);
    p.print();
    
    std::ofstream fout;
    std::random_device seeder;
    std::mt19937_64 gen(seeder());
    std::normal_distribution<double> dist(0.0,p.getd("sigma"));
    int numFiles = p.geti("numFiles");
    int dataPoints = p.geti("dataPoints");
    int startNum = p.geti("startNum");
    double dx = (p.getd("xmax")-p.getd("xmin"))/p.getd("dataPoints");
    for (int file = startNum; file < numFiles+startNum; ++file) {
        std::string outfile = filename(p.gets("fileBase"), p.geti("digits"), file, p.gets("ext"));
        
        fout.open(outfile.c_str(), std::ios::out);
        for (int i = 0; i < dataPoints; ++i) {
            double x = p.getd("xmin") + (i + 0.5)*dx;
            double y = p.getd("m")*x + p.getd("b");
            y += dist(gen)*y;
            fout << x << " " << y << std::endl;
        }
        fout.close();
    }
    
    
    return 0;
}
            
