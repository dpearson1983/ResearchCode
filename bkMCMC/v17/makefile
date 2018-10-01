CXX = cuda-g++
ARCHS = sm_52
VXX = nvcc -arch=$(ARCHS) -ccbin=cuda-g++
CXXFLAGS = -lharppi -lgsl -lgslcblas -lm
CXXOPTS = -march=native -mtune=native -O3
VXXFLAGS = -Xptxas="-v" -maxrregcount=64 --compiler-options "$(CXXFLAGS) $(CXXOPTS)"

build: hide_harppi make_spline main.cu
	$(VXX) $(VXXFLAGS) -O3 -o $(HOME)/bin/bkMCMC17 main.cu obj/make_spline.o obj/hide_harppi.o
	
hide_harppi: source/hide_harppi.cpp
	mkdir -p obj
	$(CXX) $(CXXFLAGS) $(CXXOPTS) -c -o obj/hide_harppi.o source/hide_harppi.cpp
	
make_spline: source/make_spline.cpp
	mkdir -p obj
	$(CXX) -lgsl -lgslcblas -lm $(CXXOPTS) -c -o obj/make_spline.o source/make_spline.cpp
