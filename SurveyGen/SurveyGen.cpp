/* 
 * SurveyGen.cpp
 * David W. Pearson
 * 1/15/2016
 * 
 * This code is intended to create a mock galaxy catalog much more similar to what would
 * result from a survey. It takes as input a cube (or more general cuboid) mock, a number
 * density profile, redshift limits and the size of the survey on the sky. Using this, the
 * code will find all galaxies within the survey volume and then randomly select them 
 * according to the input number density profile.
 * 
 * In the future the code will be updated to use more general survey masks, but for now
 * the survey area will be taken to be square/rectangular region for simplicity.
 * 
 * Compile with:
 * g++ -std=c++11 -lgsl -lgslcblas -lm -fopenmp -O2 SurveyGen.cpp -o SurveyGen
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>
#include <random>
#include <cstdlib>
#include <cmath>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>

