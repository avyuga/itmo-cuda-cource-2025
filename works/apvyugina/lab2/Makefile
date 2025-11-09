EIGEN_INCLUDE_DIR := /home/anastasia/DEV/cpp_libs/eigen-5.0.0/
CXXFLAGS := -Wall -std=c++17 -O2

build:
	g++ $(CXXFLAGS) src/main.cpp -o build/cpu_test -I include/ -I $(EIGEN_INCLUDE_DIR)

clean:
	rm -rf build/* 

.PHONY: clean build
