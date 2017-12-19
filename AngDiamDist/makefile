CXX = g++
CXXLIBS = -lgsl -lgslcblas -lm
CXXOPTS = -march=native -O3

build: cosmology main.cpp
	$(CXX) -lharppi $(CXXLIBS) $(CXXOPTS) -o $(HOME)/bin/angDiamDist main.cpp obj/cosmology.o
	
cosmology: source/cosmology.cpp
	mkdir -p obj
	$(CXX) $(CXXLIBS) $(CXXOPTS) -c -o obj/cosmology.o source/cosmology.cpp
