CXX = g++
CXXLIBS = -lharppi -lgsl -lgslcblas -lm
CXXFLAGS = -march=native -O3
DEPS = include/pkmcmc.h

build: main.cpp source/pkmcmc.cpp
	mkdir -p obj
	$(CXX) $(CXXLIBS) $(CXXFLAGS) -c -o obj/pkmcmc.o source/pkmcmc.cpp
	$(CXX) $(CXXLIBS) $(CXXFLAGS) -o $(HOME)/bin/pkMCMC2 main.cpp obj/pkmcmc.o
