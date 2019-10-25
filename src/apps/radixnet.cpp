/*
 * radixnet.cpp: Radix-Net sparse DNN inference for MNIST dataset
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com
 */

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
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

    if(argc != 7) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "USAGE = %s -n <Nneurons> -l <maxLayers> <path_to_input> <path_to_dnn> <Ninput_instances> <Ninput_features>\n", argv[0]);
        std::exit(Env::finalize());     
    }
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Radix-Net sparse DNN for MNIST dataset Implementation\n");
    Logging::print(Logging::LOG_LEVEL::INFO, "MPI ranks = %d, Threads per rank = %d\n", Env::nranks, Env::nthreads);
    Net<WGT> N(atoi(argv[2]), ((std::string) argv[5]), atoi(argv[4]), ((std::string) argv[6])) ;
    Logging::print(Logging::LOG_LEVEL::INFO, "Total IO time %f\n", Env::io_time);
    return(Env::finalize());
}
