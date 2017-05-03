CXX = g++
CXXFLAGS = -lgalaxy -lmpfr -lgmp -lCGAL -lgsl -lgslcblas -lm -fopenmp -march=native -O3
DEPS = dtfe.h

build: dtfe.cpp
	mkdir -p $(HOME)/include
	mkdir -p $(HOME)/lib
	cp dtfe.h $(HOME)/include
	$(CXX) $(CXXFLAGS) -c -o $(HOME)/lib/libdtfe.a dtfe.cpp
