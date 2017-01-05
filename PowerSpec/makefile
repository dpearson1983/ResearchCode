CXX = g++
CXXFLAGS = -march=native -mtune=native -O3 -w
LDLIBS = -lfftw3 -lfftw3_omp -fopenmp
DEPS = powerspec.h

localinstall: powerspec.cpp
	mkdir -p $(HOME)/include
	mkdir -p $(HOME)/lib
	cp powerspec.h $(HOME)/include/
	$(CXX) $(CXXFLAGS) $(LDLIBS) -c -o $(HOME)/lib/libpowerspec.a powerspec.cpp
