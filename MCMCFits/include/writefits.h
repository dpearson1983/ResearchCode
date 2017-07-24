#ifndef _WRITEFITS_H_
#define _WRITEFITS_H_

#include <vector>
#include <string>

bool write_data_to_fits(std::vector<std::string> col_name, std::vector<std::string> col_form, 
                        std::vector<std::string> col_unit, unsigned long rows, std::string in_file);
