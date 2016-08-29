 CXX=g++ -std=c++14
 CXXFLAGS= -fopenmp -march=native -mtune=native -O3
 LDFLAGS=-lgsl -lgslcblas -lfftw3 -lfftw3_omp -lm
 DEPS= lognormal.h pods.h
 
 LNKNLogsmake: lognormal.cpp LNKNLogs.cpp redshiftDistort.cpp
	$(CXX) $(LDFLAGS) $(CXXFLAGS) -c -o lognormal.o lognormal.cpp
	$(CXX) $(LDFLAGS) $(CXXFLAGS) -o $(HOME)/bin/LNKNLogs LNKNLogs.cpp lognormal.o
	$(CXX) -lharppi -march=native -mtune=native -O3 -o $(HOME)/bin/redshiftDistort redshiftDistort.cpp
