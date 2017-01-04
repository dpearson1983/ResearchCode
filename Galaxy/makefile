CXX = g++
CXXFLAGS = -march=native -mtune=native -O3
DEPS = galaxy.h

localinstall: galaxy.cpp
	mkdir -p $(HOME)/include
	mkdir -p $(HOME)/lib
	cp galaxy.h $(HOME)/include/
	$(CXX) $(CXXFLAGS) -w -c -o $(HOME)/lib/libgalaxy.a galaxy.cpp
