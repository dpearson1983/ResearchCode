CXX = g++
CXXLIBS = -lharppi -lCCfits -lcfitsio
CXXFLAGS = -march=native -O3

build: main.cpp
	$(CXX) $(CXXLIBS) $(CXXFLAGS) -o $(HOME)/bin/MCMCfits main.cpp
