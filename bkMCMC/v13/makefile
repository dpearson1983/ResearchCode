CXX = g++
CXXLIBS = -lgsl -lgslcblas -lm -fopenmp
CXXFLAGS = -march=native -O3

build: harppi bkmcmc main.cpp
	mkdir -p $(HOME)/bin
	$(CXX) $(CXXLIBS) $(CXXFLAGS) -o $(HOME)/bin/bkMCMC13 main.cpp obj/harppi.o obj/bkmcmc.o
	
harppi: source/harppi.cpp
	mkdir -p obj
	$(CXX) $(CXXFLAGS) -c -o obj/harppi.o source/harppi.cpp
	
bkmcmc: source/bkmcmc.cpp
	mkdir -p obj
	$(CXX) $(CXXLIBS) $(CXXFLAGS) -c -o obj/bkmcmc.o source/bkmcmc.cpp
