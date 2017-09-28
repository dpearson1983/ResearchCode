CXX = g++
CXX_LIBS = -lharppi -lgsl -lgslcblas -lm
CXX_OPTS = -march=native -O3

build: main.cpp
	mkdir -p $(HOME)/bin
	$(CXX) $(CXX_LIBS) $(CXX_OPTS) -o $(HOME)/bin/pkSlope main.cpp
