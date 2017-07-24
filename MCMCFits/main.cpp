#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <CCfits/CCfits>
#include <harppi.h>

int main(int argc, char *argv[]) {
    parameters p(argv[1]);
    p.print();
    
    std::ifstream fin;
    
    std::unique_ptr<CCfits::FITS> pFits;
    pFits.reset(new CCfits::FITS(p.gets("out_file"), CCfits::Write));
    
    std::string hduName("MCMC_CHAIN");
    
    std::vector<std::string> col_name(p.geti("num_cols"));
    std::vector<std::string> col_form(p.geti("num_cols"));
//     std::vector<std::string> col_unit(p.geti("num_cols"));
    
    std::cout << "Getting column info from parameters..." << std::endl;
    for (int i = 0; i < p.geti("num_cols"); ++i) {
        col_name[i] = p.gets("col_name", i);
        col_form[i] = p.gets("col_form", i);
//         col_unit[i] = p.gets("col_unit", i);
    }
    
    std::vector<std::vector<double>> data(p.geti("num_cols"));
    
    std::cout << "Reading in the data file..." << std::endl;
    if (std::ifstream(p.gets("in_file"))) {
        fin.open(p.gets("in_file").c_str(), std::ios::in);
        while (!fin.eof()) {
            for (int i = 0; i < p.geti("num_cols"); ++i) {
                double temp;
                fin >> temp;
                if (!fin.eof()) {
                    data[i].push_back(temp);
                }
            }
        }
        fin.close();
    } else {
        std::stringstream message;
        message << " Could not open " << p.gets("in_file") << std::endl;
        throw std::runtime_error(message.str());
    }
    
    int rows = data[0].size();
    std::cout << "rows = " << rows << std::endl;
    
    CCfits::Table *newTable = pFits->addTable(hduName, rows, col_name, col_form);
    
    std::cout << "Writing fits file..." << std::endl;
    for (int i = 0; i < p.geti("num_cols"); ++i) {
        newTable->column(col_name[i]).write(data[i], 1);
    }
    
    return 0;
}
