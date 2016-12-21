CXX=g++ -std=c++14
CXXFLAGS= -march=native -mtune=native -w -O3
LDLIBS= -lharppi -lpfunk -lfileFuncs -lfftw3 -lfftw3_omp -lgsl -lgslcblas -fopenmp
DEPS=$(HOME)/ResearchCode/PowerSpecCalc/pfunk.h $(HOME)/ResearchCode/file/fileFuncs.h $(HOME)/ResearchCode/HARPPI/harppi.h

localinstall: bispectrum.cpp
	mkdir -p $(HOME)/bin
	$(CXX) $(CXXFLAGS) $(LDLIBS) -o $(HOME)/bin/bispectrum bispectrum.cpp
