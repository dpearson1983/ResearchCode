#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <CCfits/CCfits>
#include "include/harppi.h"

int main(int argc, char *argv[]) {
    parameter p(argv[1]);
    p.print();
    
    std::unique_ptr<FITS> pFits(0);
    pFits.reset(new FITS(p.gets("out_file"), Write));
    
    std::string hduName("MCMC_CHAIN");
    
    std::vector<std::string> col_name(p.geti("num_cols"));
    std::vector<std::string> col_form(p.geti("num_cols"));
    std::vector<std::string> col_unit(p.geti("num_cols"));
    
    for (int i = 0; i < p.geti("num_cols"); ++i) {
        col_name[i] = p.gets("col_name", i);
        col_form[i] = p.gets("col_form", i);
        col_unit[i] = p.gets("col_unit", i);
    }
    
    Table *newTable = pFits->addTable(hduName, rows, col_name, col_form, col_unit);
    
    
