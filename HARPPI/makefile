CXX=g++ -std=c++14
CXXFLAGS= -march=native -mtune=native -O3
DEPS=harppi.h

harppimake: harppi.cpp
	$(CXX) $(CXXFLAGS) -c -o harppi.a harppi.cpp
	
localinstall: harppi.cpp
	cp harppi.h $(HOME)/include/
	$(CXX) $(CXXFLAGS) -c -o $(HOME)/lib/libharppi.a harppi.cpp
	
globalinstall: harppi.cpp
	cp harrpi.h /usr/include
	mkdir /usr/lib/harppi
	$(CXX) $(CXXFLAGS) -c -o /usr/lib/harppi/harppi.a harppi.cpp
	cd /usr/lib
	ls -l harppi/harppi.a
	cd -
