#!/bin/make
# Makefile
# (c) Mohammad Hasanzadeh Mofrad, 2019
# (e) m.hasanzadeh.mofrad@gmail.com

CXX = g++
CXX_MPI = mpicxx
#CXX_MPI = mpicxx.mpich
#DEBUG = -fsanitize=address
CXX_OPTIMIZED = -DNDEBUG -O3 -flto -fwhole-program -march=native -ftree-vectorize -ffast-math -funroll-loops
CXX_SKIPPED_WARNINGS = -Wall -Wextra -Werror -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable -Wno-maybe-uninitialized
CXX_FLAGS = -std=c++17 $(CXX_OPTIMIZED) $(CXX_SKIPPED_WARNINGS)
THREADED = -fopenmp -D_GLIBCXX_PARALLEL -pthread
NUMACTL = /ihome/rmelhem/moh18/numactl/libnuma
LIBNUMA = /ihome/rmelhem/moh18/numactl/libnuma/usr/local/lib
SYSLIBS = -lnuma -I $(NUMACTL) -L$(LIBNUMA)

OBJS = radixnet

all: dir $(OBJS)

dir:
	@mkdir -p bin

$(OBJS): %: src/apps/%.cpp
	$(CXX_MPI) $(CXX_FLAGS) $(THREADED) $(DEBUG) -o bin/$@ -I src $< $(SYSLIBS)

clean:
	rm -rf bin 
