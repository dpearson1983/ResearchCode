#include <fstream>
#include <iomanip>
#include <string>

std::string filename(std::string base, int digits, int num, std::string ext) {
    std::stringstream file;
    file << base << std::setw(digits) << std::setfill('0') << num << ext;
    return file.str();
}

long int filesize(std::string file) {
    std::ifstream fin;
    std::streampos begin, end;
    fin.open(file.c_str(), std::ios::binary);
    begin = fin.tellg();
    fin.seekg(0, std::ios::end);
    end = fin.tellg();
    fin.close();
    return end-begin;
}
