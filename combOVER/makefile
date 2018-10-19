CXX = cuda-g++
VXX = nvcc -arch=sm_52 -ccbin=cuda-g++
CXXFLAGS = -lharppi -lgsl -lgslcblas -lm
CXXOPTS = -march=native -mtune=native -O3
VXXFLAGS = -maxrregcount=64 --compiler-options "$(CXXFLAGS) $(CXXOPTS)" -O3

build: hide_harppi make_spline file_check main.cu
	$(VXX) $(VXXFLAGS) -o $(HOME)/bin/combOVER main.cu obj/hide_harppi.o obj/make_spline.o obj/file_check.o
	
hide_harppi: source/hide_harppi.cpp
	mkdir -p obj
	$(CXX) $(CXXFLAGS) $(CXXOPTS) -c -o obj/hide_harppi.o source/hide_harppi.cpp

make_spline: source/make_spline.cpp
	mkdir -p obj
	$(CXX) $(CXXFLAGS) $(CXXOPTS) -c -o obj/make_spline.o source/make_spline.cpp
	
file_check: source/file_check.cpp
	mkdir -p obj
	$(CXX) $(CXXFLAGS) $(CXXOPTS) -c -o obj/file_check.o source/file_check.cpp
