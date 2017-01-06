CXX = g++ -std=c++14
CXXFLAGS = -march=native -mtune=native -w -O3
LDLIBS = -lpowerspec -lfftw3 -lfftw3_omp -fopenmp
DEPS = bispec.h $(HOME)/include/tpods.h $(HOME)/include/constants.h $(HOME)/include/powerspec.h

localinstall: bispec.cpp
	mkdir -p $(HOME)/include
	mkdir -p $(HOME)/lib
	cp bispec.h $(HOME)/include/
	$(CXX) $(CXXFLAGS) $(LDLIBS) -c -o $(HOME)/lib/libbispec.a bispec.cpp
