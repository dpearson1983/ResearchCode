CXX = g++
ARCHS = sm_52
VXX = nvcc -std=c++11 -arch=$(ARCHS)
CXXFLAGS = -lharppi -lgsl -lgslcblas -lm
CXXOPTS = -march=native -O3
VXXFLAGS = --compiler-options "$(CXXFLAGS) $(CXXOPTS)"

build: bkMCMC6.cu hide_harppi.cpp make_spline.cpp
	$(CXX) -lgsl -lgslcblas -lm $(CXXOPTS) -c -o make_spline.o make_spline.cpp
	$(CXX) $(CXXFLAGS) $(CXXOPTS) -c -o hide_harppi.o hide_harppi.cpp
	$(VXX) $(VXXFLAGS) -O3 -o $(HOME)/bin/bkMCMC6 bkMCMC6.cu make_spline.o hide_harppi.o
	
debugbuild: bkMCMC6.cu hide_harppi.cpp make_spline.cpp
	$(CXX) -lgsl -lgslcblas -lm $(CXXOPTS) -c -o make_spline.o make_spline.cpp
	$(CXX) $(CXXFLAGS) $(CXXOPTS) -c -o hide_harppi.o hide_harppi.cpp
	$(VXX) $(VXXFLAGS) -G -g -o bkMCMC6 bkMCMC6.cu make_spline.o hide_harppi.o
