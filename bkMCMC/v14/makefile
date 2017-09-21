CXX = g++
ARCHS = sm_50
VXX = nvcc -std=c++11 -arch=$(ARCHS)
CXXLIBS = -lgsl -lgslcblas -lm
CXXFLAGS = -march=native -O3
VXXFLAGS = --compiler-options "$(CXXLIBS) $(CXXFLAGS)"

build: hide_harppi make_spline main.cu
	$(VXX) $(VXXFLAGS) -O3 -o $(HOME)/bin/bkMCMC14 main.cu obj/make_spline.o obj/hide_harppi.o obj/harppi.o
	
hide_harppi: harppi source/hide_harppi.cpp
	mkdir -p obj
	$(CXX) $(CXXLIBS) $(CXXFLAGS) -c -o obj/hide_harppi.o source/hide_harppi.cpp
	
harppi: source/harppi.cpp
	mkdir -p obj
	$(CXX) $(CXXFLAGS) -c -o obj/harppi.o source/harppi.cpp

make_spline: source/make_spline.cpp
	mkdir -p obj
	$(CXX) $(CXXLIBS) $(CXXFLAGS) -c -o obj/make_spline.o source/make_spline.cpp
