CXX=g++ -std=c++14
CXXFLAGS= -march=native -mtune=native -O3
DEPS=fileFuncs.h

localinstall: fileFuncs.cpp
	mkdir -p $(HOME)/include
	mkdir -p $(HOME)/lib
	cp fileFuncs.h $(HOME)/include/
	$(CXX) $(CXXFLAGS) -c -o $(HOME)/lib/libfileFuncs.a fileFuncs.cpp
	
fileFuncsmake: fileFuncs.cpp
	$(CXX) $(CXXFLAGS) -c -o libfileFuncs.a fileFuncs.cpp
