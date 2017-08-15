CXX = g++
CXXFLAGS = -march=native -O3
CXXLIBS = -lharppi -lfftw3 -lfftw3_omp -fopenmp

build: main.cpp source/bispec.cpp
	$(CXX) $(CXXLIBS) $(CXXFLAGS) -c -o obj/bispec.o source/bispec.cpp
	$(CXX) $(CXXLIBS) $(CXXFLAGS) -o bispecCPU main.cpp obj/bispec.o
