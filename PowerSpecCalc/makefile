CXX=g++ -std=c++14
CXXFLAGS= -march=native -mtune=native -w -O3
LDLIBS= -lharppi -lfileFuncs -lfftw3 -lfftw3_omp -lgsl -lgslcblas -fopenmp
DEPS=pfunk.h $(HOME)/ResearchCode/file/fileFuncs.h

localinstall: pfunk.cpp LNPk-PP.cpp
	mkdir -p $(HOME)/include
	mkdir -p $(HOME)/lib
	mkdir -p $(HOME)/bin
	cp pfunk.h $(HOME)/include/
	$(CXX) $(CXXFLAGS) $(LDLIBS) -c -o $(HOME)/lib/libpfunk.a pfunk.cpp
	$(CXX) $(CXXFLAGS) $(LDLIBS) -lpfunk -o $(HOME)/bin/LNPk-PP LNPk-PP.cpp
	$(CXX) $(CXXFLAGS) $(LDLIBS) -lpfunk -o $(HOME)/bin/LNPk-rad LNPk-rad.cpp
	$(CXX) $(CXXFLAGS) $(LDLIBS) -lpfunk -o $(HOME)/bin/LNPk-radPP LNPk-radPP.cpp
