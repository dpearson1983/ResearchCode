#include <iostream>
#include <fstream>
#include <vector>
#include <harppi.h>
#include "include/file_check.h"
#include "include/pkMod.h"

int main(int argc, char *argv[]) {
    parameters p(argv[1]);
    p.print();
    
    std::vector<double> params(p.geti("num_params"));
    std::vector<std::vector<double>> covar(p.geti("num_data"),std::vector<double>(p.geti("num_data")));
    
    for (int i = 0; i < p.geti("num_params"); ++i)
        params[i] = p.getd("params", i);
    
    if (check_file_exists(p.gets("covar_file"))) {
        std::ifstream fin(p.gets("covar_file"));
        for (int i = 0; i < p.geti("num_data"); ++i) {
            for (int j = 0; j < p.geti("num_data"); ++j) {
                fin >> covar[i][j];
            }
        }
        fin.close();
    }
    
    pk_model mod(p.gets("pk_file"), p.gets("data_file"), params);
    
    mod.calculate_model();
    
    mod.write_model_to_file(p.gets("model_file"));
    
    mod.write_normalized_data_to_file(p.gets("data_norm_file"));
    
    mod.normalize_covariance(covar);
    
    std::ofstream fout(p.gets("covar_norm_file"));
    fout.precision(15);
    for (int i = 0; i < p.geti("num_data"); ++i) {
        for (int j = 0; j < p.geti("num_data"); ++j) {
            fout.width(25);
            fout << covar[i][j];
        }
        fout << "\n";
    }
    fout.close();
    
    return 0;
}
