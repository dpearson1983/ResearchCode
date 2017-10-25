#include <fstream>
#include <sstream>
#include <string>

bool check_file_exists(const std::string &file) {
    if (std::ifstream(file)) {
        return true;
    } else {
        std::stringstream message;
        message << "Cannot open " << file << ". Check that the file exists in the appropriate directory.";
        message << std::endl;
        throw std::runtime_error(message.str());
    }
}

bool check_file_exists(const std::string &file, std::string &error_message) {
    if (std::ifstream(file)) {
        return true;
    } else {
        std::stringstream message;
        message << "Cannot open " << file << ". Check that the file exists in the appropriate directory.";
        message << std::endl;
        error_message = message.str();
        return false;
    }
}
