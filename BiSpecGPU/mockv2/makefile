VXX = nvcc
CXX = g++
CXXLIBS = -lgsl -lgslcblas -lfftw3 -lfftw3_omp -lm -fopenmp
CXXOPTS = -march=native -O3
VXXOPTS = -arch=sm_50 --compiler-options "-std=c++98 $(CXXLIBS) $(CXXOPTS)" -O3
OBJ_FILES = obj/add_funcs.o obj/cosmology.o obj/density_field.o obj/file_check.o obj/file_reader.o obj/galaxy.o obj/harppi.o

build: add_funcs cosmology galaxy density_field file_check file_reader harppi main.cu
	$(VXX) $(VXXOPTS) -o $(HOME)/bin/bispecGPUMock2 main.cu $(OBJ_FILES)
	
add_funcs: source/add_funcs.cpp
	mkdir -p obj
	$(CXX) $(CXXOPTS) -c -o obj/add_funcs.o source/add_funcs.cpp
	
cosmology: source/cosmology.cpp
	mkdir -p obj
	$(CXX) $(CXXOPTS) -c -o obj/cosmology.o source/cosmology.cpp
	
galaxy: source/galaxy.cpp
	mkdir -p obj
	$(CXX) $(CXXLIBS) $(CXXOPTS) -c -o obj/galaxy.o source/galaxy.cpp
	
density_field: source/density_field.cpp
	mkdir -p obj
	$(CXX) $(CXXOPTS) -c -o obj/density_field.o source/density_field.cpp
	
file_check: source/file_check.cpp
	mkdir -p obj
	$(CXX) $(CXXOPTS) -c -o obj/file_check.o source/file_check.cpp
	
file_reader: source/file_reader.cpp
	mkdir -p obj
	$(CXX) $(CXXOPTS) -c -o obj/file_reader.o source/file_reader.cpp
	
harppi: source/harppi.o
	mkdir -p obj
	$(CXX) $(CXXOPTS) -c -o obj/harppi.o source/harppi.cpp
