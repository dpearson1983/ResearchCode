#ifndef _FILE_CHECK_H_
#define _FILE_CHECK_H_

#include <fstream>
#include <sstream>
#include <string>

bool check_file_exists(const std::string &file);

bool check_file_exists(const std::string &file, std::string &error_message);

#endif
