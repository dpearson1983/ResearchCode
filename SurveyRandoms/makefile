CXX=g++ -std=c++14
CXXFLAGS= -march=native -mtune=native -w -O3
LDLIBS= -lharppi -lpfunk -lgsl -lgslcblas -lchealpix -lcfitsio -fopenmp
DEPS=pfunk.h

localinstall: surveyRandoms.cpp
	mkdir -p $(HOME)/bin
	$(CXX) $(CXXFLAGS) $(LDLIBS) -o $(HOME)/bin/surveyRandoms surveyRandoms.cpp
