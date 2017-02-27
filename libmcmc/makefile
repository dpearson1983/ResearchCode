CXX = g++
CXXFLAGS = -fopenmp -march=native -mtune=native -O3
DEPS = mcmc.h

build: mcmc.cpp
	mkdir -p $(HOME)/include
	mkdir -p $(HOME)/lib
	cp mcmc.h $(HOME)/include/
	$(CXX) $(CXXFLAGS) -c -o $(HOME)/lib/libmcmc.a mcmc.cpp
