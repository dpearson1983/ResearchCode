#ifndef _FILEFUNCS_H_
#define _FILEFUNCS_H_

#include <fstream>
#include <string>

std::string filename(std::string base, int digits, int num, std::string ext);

long int filesize(std::string file);

#endif
