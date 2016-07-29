#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <typeinfo>
#include "harppi.h"

bool parameters::assignParams(std::string type, std::string key, std::string val) {
    if (type == "string") {
        parameters::strings.insert(std::pair<std::string, std::string>(key,val));
    } else if (type == "int") {
        parameters::ints.insert(std::pair<std::string, int>(key,atof(val.c_str())));
    } else if (type == "double") {
        parameters::doubles.insert(std::pair<std::string, double>(key,atof(val.c_str())));
    } else if (type == "bool") {
        bool b;
        std::istringstream(val) >> std::boolalpha >> b;
        parameters::bools.insert(std::pair<std::string, bool>(key, b));
    } else if (type == "vector<double>") {
        std::vector<double> vec;
        std::istringstream iss(val);
        std::string s;
        while (std::getline(iss, s, ',')) {
            vec.push_back(atof(s.c_str()));
        }
        parameters::dvectors.insert(std::pair<std::string, std::vector<double> >(key, vec));
    } else if (type == "vector<int>") {
        std::vector<int> vec;
        std::istringstream iss(val);
        std::string s;
        while (std::getline(iss, s, ',')) {
            vec.push_back(atof(s.c_str()));
        }
        parameters::ivectors.insert(std::pair<std::string, std::vector<int> >(key, vec));
    } else if (type == "vector<string>") {
        std::vector<std::string> vec;
        std::istringstream iss(val);
        std::string s;
        while (std::getline(iss, s, ',')) {
            vec.push_back(s);
        }
        parameters::svectors.insert(std::pair<std::string, std::vector<std::string> >(key, vec));
    } else {
        std::cout << "WARNING: Unrecognized type specified in parameter file.\n";
        std::cout << "    Type " << type << " is not currently supported.\n";
        std::cout << "    Currently supported types are:\n";
        std::cout << "        string, int, double, bool, vector<string>\n";
        std::cout << "        vector<double>, and vector<int>" << std::endl;
        return false;
    }
    
    return true;
}

void parameters::readParams(char *file) {
    std::ifstream fin;
    std::string line, type, key, equal, val;
    bool check = true;
    
    fin.open(file, std::ios::in);
    while (std::getline(fin, line) && check) {
        std::istringstream iss(line);
        iss >> type >> key >> equal >> val;
        if (type != "#") {
            check = parameters::assignParams(type, key, val);
        } 
    }
    fin.close();
    
    if (!check) {
        std::stringstream message;
        message << "ERROR: All parameters have not been assigned" << std::endl;
        throw std::runtime_error(message.str());
    }
}

void parameters::print() {
    std::map<std::string, std::string>::iterator it_string = parameters::strings.begin();
    std::map<std::string, int>::iterator it_int = parameters::ints.begin();
    std::map<std::string, double>::iterator it_double = parameters::doubles.begin();
    std::map<std::string, bool>::iterator it_bool = parameters::bools.begin();
    std::map<std::string, std::vector<double> >::iterator it_dvectors = 
                                                        parameters::dvectors.begin();
    std::map<std::string, std::vector<int> >::iterator it_ivectors = 
                                                        parameters::ivectors.begin();
    std::map<std::string, std::vector<std::string> >::iterator it_svectors =
                                                        parameters::svectors.begin();
    
    for (it_string = parameters::strings.begin(); it_string != parameters::strings.end(); ++it_string)
        std::cout << "string " << it_string->first << " = " << it_string->second << std::endl;
    for (it_int = parameters::ints.begin(); it_int != parameters::ints.end(); ++it_int)
        std::cout << "int " << it_int->first << " = " << it_int->second << std::endl;
    for (it_double = parameters::doubles.begin(); it_double != parameters::doubles.end(); ++it_double)
        std::cout << "double " << it_double->first << " = " << it_double->second << std::endl;
    for (it_bool = parameters::bools.begin(); it_bool != parameters::bools.end(); ++it_bool)
        std::cout << "bool " << it_bool->first << " = " << it_bool->second << std::endl;
    for (it_dvectors = parameters::dvectors.begin(); it_dvectors != parameters::dvectors.end(); ++it_dvectors) {
        int numVals = parameters::dvectors[it_dvectors->first].size();
        std::cout << "vector<double> " << it_dvectors->first << " = ";
        for (int i = 0; i < numVals; ++i) {
            std::cout << parameters::dvectors[it_dvectors->first][i];
            if (i != numVals-1) std::cout << ",";
        }
        std::cout << std::endl;
    }
    for (it_ivectors = parameters::ivectors.begin(); it_ivectors != parameters::ivectors.end(); ++it_ivectors) {
        int numVals = parameters::ivectors[it_ivectors->first].size();
        std::cout << "vector<int> " << it_ivectors->first << " = ";
        for (int i = 0; i < numVals; ++i) {
            std::cout << parameters::ivectors[it_ivectors->first][i];
            if (i != numVals-1) std::cout << ",";
        }
        std::cout << std::endl;
    }
    for (it_svectors = parameters::svectors.begin(); it_svectors != parameters::svectors.end(); ++it_svectors) {
        int numVals = parameters::svectors[it_svectors->first].size();
        std::cout << "vector<string> " << it_svectors->first << " = ";
        for (int i = 0; i < numVals; ++i) {
            std::cout << parameters::svectors[it_svectors->first][i];
            if (i != numVals-1) std::cout << ",";
        }
        std::cout << std::endl;
    }
}

void parameters::check_min(std::vector<typekey> neededParams) {
    int minNum = neededParams.size();
    int count = 0;
                                                        
    for (int i = 0; i < minNum; ++i) {
        if (neededParams[i].type == "int") {
            if (parameters::ints.count(neededParams[i].key) == 1) ++count;
            else std::cout << neededParams[i].type << " " << neededParams[i].key << " not found." << std::endl;
        } else if (neededParams[i].type == "double") {
            if (parameters::doubles.count(neededParams[i].key) == 1) ++count;
            else std::cout << neededParams[i].type << " " << neededParams[i].key << " not found." << std::endl;
        } else if (neededParams[i].type == "string") {
            if (parameters::strings.count(neededParams[i].key) == 1) ++count;
            else std::cout << neededParams[i].type << " " << neededParams[i].key << " not found." << std::endl;
        } else if (neededParams[i].type == "bool") {
            if (parameters::bools.count(neededParams[i].key) == 1) ++count;
            else std::cout << neededParams[i].type << " " << neededParams[i].key << " not found." << std::endl;
        } else if (neededParams[i].type == "vector<double>") {
            if (parameters::dvectors.count(neededParams[i].key) == 1) ++count;
            else std::cout << neededParams[i].type << " " << neededParams[i].key << " not found." << std::endl;
        } else if (neededParams[i].type == "vector<int>") {
            if (parameters::ivectors.count(neededParams[i].key) == 1) ++count;
            else std::cout << neededParams[i].type << " " << neededParams[i].key << " not found." << std::endl;
        } else if (neededParams[i].type == "vector<string>") {
            if (parameters::svectors.count(neededParams[i].key) == 1) ++count;
            else std::cout << neededParams[i].type << " " << neededParams[i].key << " not found." << std::endl;
        }
    }
    
    if (count != minNum) {
        std::stringstream message;
        message << "ERROR: Minimum parameters not found." << std::endl;
        throw std::runtime_error(message.str());
    }
}

double parameters::getd(std::string key, int element) {
    if (parameters::ints.count(key) == 1) {
        return double(parameters::ints[key]);
    } else if (parameters::doubles.count(key) == 1) {
        return parameters::doubles[key];
    } else if (parameters::dvectors.count(key) == 1) {
        return parameters::dvectors[key][element];
    } else if (parameters::ivectors.count(key) == 1) {
        return double(parameters::ivectors[key][element]);
    } else {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " is not a numeric type." << std::endl;
        throw std::runtime_error(message.str());
    }       
}

int parameters::geti(std::string key, int element) {
    if (parameters::ints.count(key) == 1) {
        return parameters::ints[key];
    } else if (parameters::doubles.count(key) == 1) {
        return int(parameters::doubles[key]);
    } else if (parameters::dvectors.count(key) == 1) {
        return int(parameters::dvectors[key][element]);
    } else if (parameters::ivectors.count(key) == 1) {
        return parameters::ivectors[key][element];
    } else {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " is not a numeric type." << std::endl;
        throw std::runtime_error(message.str());
    }
}

bool parameters::getb(std::string key, int element) {
    if (parameters::bools.count(key) == 1) {
        return parameters::bools[key];
    } else {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " is not a boolean type." << std::endl;
        throw std::runtime_error(message.str());
    }
}

std::string parameters::gets(std::string key, int element) {
    if (parameters::strings.count(key) == 1) {
        return parameters::strings[key];
    } else if (parameters::svectors.count(key) == 1) {
        return parameters::svectors[key][element];
    } else {
        std::stringstream message;
        message << "ERROR: Parameter " << key << " is not a string type." << std::endl;
        throw std::runtime_error(message.str());
    }
}
