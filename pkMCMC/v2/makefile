CXX = g++
CXXLIBS = -lharppi -lgsl -lgslcblas -lm
CXXFLAGS = -march=native -O3
DEPS = include/pkmcmc.h

build: file_check pkmcmc main.cpp
	$(CXX) $(CXXLIBS) $(CXXFLAGS) -o $(HOME)/bin/pkMCMC2 main.cpp obj/pkmcmc.o obj/file_check.o

file_check: source/file_check.cpp
	mkdir -p obj
	$(CXX) $(CXXFLAGS) -c -o obj/file_check.o source/file_check.cpp
	
pkmcmc: source/pkmcmc.cpp
	mkdir -p obj
	$(CXX) $(CXXLIBS) $(CXXFLAGS) -c -o obj/pkmcmc.o source/pkmcmc.cpp
