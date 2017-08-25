CXX = g++
CXXLIBS = -lgsl -lgslcblas -lm
CXXFLAGS = -march=native -O3

build: file_check pkMod main.cpp
	$(CXX) -lharppi $(CXXLIBS) $(CXXFLAGS) -o $(HOME)/bin/pkModel main.cpp obj/file_check.o obj/pkMod.o

file_check: source/file_check.cpp
	mkdir -p obj
	$(CXX) $(CXXFLAGS) -c -o obj/file_check.o source/file_check.cpp

pkMod: source/pkMod.cpp
	mkdir -p obj
	$(CXX) $(CXXLIBS) $(CXXFLAGS) -c -o obj/pkMod.o source/pkMod.cpp
