/*
 * radixnet.cpp: Radix-Net sparse DNN inference for MNIST dataset
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com
 */
 
// make clean && make && time mpirun.mpich -np 4 bin/./radixnet -m 60000 -n 1024 -l 120 data1/bin/MNIST data1/bin/DNN 

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <memory>

#include "radixnet.h"
#include "env.hpp"
#include "log.hpp"
#include "triple.hpp"
#include "io.hpp"

#include "tiling.hpp"
#include "net.hpp"
#include "allocator.hpp"

int main(int argc, char **argv) {

    Logging::enabled = true;
    int status = Env::init();
    if(status) {
        Logging::print(Logging::LOG_LEVEL::FATAL, "Failure to initialize MPI environment\n");
        std::exit(Env::finalize());   
    }

    if(argc != 9) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "USAGE = %s -m <NinputInstances> -n <Nneurons> -l <maxLayers> <path_to_input> <path_to_dnn>\n", argv[0]);
        std::exit(Env::finalize());     
    }
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Radix-Net sparse DNN for MNIST dataset Implementation\n");
    Logging::print(Logging::LOG_LEVEL::INFO, "MPI ranks = %d, Threads per rank = %d\n", Env::nranks, Env::nthreads);
    Logging::print(Logging::LOG_LEVEL::INFO, "Sockets   = %d, Processors = %d\n", Env::nsockets, Env::ncores);
    if(Env::NUMA_ALLOC) {
        Logging::print(Logging::LOG_LEVEL::INFO, "NUMA is enabled.\n", Env::NUMA_ALLOC);
    }
    else {
        Logging::print(Logging::LOG_LEVEL::WARN, "NUMA is disabled.\n", Env::NUMA_ALLOC);
    }
    
    Net<WGT> N(atoi(argv[2]), atoi(argv[4]), ((std::string) argv[7]), atoi(argv[6]), ((std::string) argv[8]));//, INPUT_TYPE::_TEXT_) ;
    
    return(Env::finalize());
}
