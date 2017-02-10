CXX = g++
CXXFLAGS = -std=c++98 -lharppi -lgalaxy -lgsl -lgslcblas -lCCfits -lcfitsio
NVXX = nvcc
NVXXFLAGS = -arch=sm_52 --compiler-options "$(CXXFLAGS) -lpowerspec -lfftw3 -lfftw3_omp -fopenmp -march=native -mtune=native -O3" -O3

localinstall: bispecGPU.cu source/fileReader.cpp
	mkdir -p $(HOME)/bin
	$(CXX) $(CXXFLAGS) -c -o fileReader.o source/fileReader.cpp
	$(NVXX) $(NVXXFLAGS) -o $(HOME)/bin/bispecGPU bispecGPU.cu fileReader.o

powerspec: pkGPU.cu source/fileReader.cpp
	mkdir -p $(HOME)/bin
	$(CXX) $(CXXFLAGS) -c -o fileReader.o source/fileReader.cpp
	$(NVXX) $(NVXXFLAGS) -o $(HOME)/bin/pkGPU pkGPU.cu fileReader.o

shell: bispecGPUShell.cu source/fileReader.cpp
	mkdir -p $(HOME)/bin
	$(CXX) $(CXXFLAGS) -c -o fileReader.o source/fileReader.cpp
	$(NVXX) $(NVXXFLAGS) -o $(HOME)/bin/bispecGPUShell bispecGPUShell.cu fileReader.o
