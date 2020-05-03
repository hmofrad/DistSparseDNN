/*
 * radixnet.cpp: Radix-Net sparse DNN inference for MNIST dataset
 * [http://graphchallenge.mit.edu/]
 * (c) Mohammad Hasanzadeh Mofrad, 2020
 * (e) m.hasanzadeh.mofrad@gmail.com
 */
 
// make clean && make && time mpirun.mpich -np 4 bin/./radixnet -m 60000 -n 1024 -l 120 data1/bin/MNIST data1/bin/DNN -p 0

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <memory>

#include "env.hpp"
#include "log.hpp"
#include "triple.hpp"
#include "io.hpp"
#include "tiling.hpp"
#include "net.hpp"
#include "allocator.hpp"

using WGT = double;

int main(int argc, char **argv) {
    Logging::enabled = true;
    int status = Env::init();
    if(status) {
        Logging::print(Logging::LOG_LEVEL::FATAL, "Failure to initialize MPI environment\n");
        std::exit(Env::finalize());   
    }

    if(argc != 11) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "USAGE = %s -m <NinputInstances> -n <nneurons> -l <nmax_layers> <path_to_input> <path_to_dnn> -p <parallelism_type>\n", argv[0]);
        std::exit(Env::finalize());     
    }
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Radix-Net sparse DNN for MNIST dataset Implementation\n");
    Logging::print(Logging::LOG_LEVEL::INFO, "Machines = %d, MPI ranks  = %d, Threads per rank = %d\n", Env::nmachines, Env::nranks, Env::nthreads);
    Logging::print(Logging::LOG_LEVEL::INFO, "Sockets  = %d, Processors = %d\n", Env::nsockets, Env::ncores);
    if(Env::NUMA_ALLOC) {
        Logging::print(Logging::LOG_LEVEL::INFO, "NUMA is enabled.\n", Env::NUMA_ALLOC);
    }
    else {
        Logging::print(Logging::LOG_LEVEL::WARN, "NUMA is disabled.\n", Env::NUMA_ALLOC);
    }
    
	std::string script = "radixnet";
	uint32_t ninstances = atoi(argv[2]);
	uint32_t nneurons = atoi(argv[4]);
	uint32_t nmax_layers = atoi(argv[6]);
	std::string layerFile_prefix = ((std::string) argv[8]);
	
	INPUT_TYPE input_type = INPUT_TYPE::_BINARY_;
	
    std::vector<uint32_t> nneurons_vector = {1024, 4096, 16384, 65536};    
    uint32_t idxN = std::distance(nneurons_vector.begin(), std::find(nneurons_vector.begin(), nneurons_vector.end(), nneurons));
    if(idxN >= nneurons_vector.size()) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Invalid number of neurons %d\n", nneurons);
        std::exit(Env::finalize());
    }    
    
    std::string inputFile_prefix = ((std::string) argv[7]);
    std::string feature_file = inputFile_prefix + "/sparse-images-" + std::to_string(nneurons);
    feature_file += (input_type == INPUT_TYPE::_TEXT_) ? ".tsv" : ".bin";
    
	
    std::vector<uint32_t> nmax_layers_vector = {120, 480, 1920};
    uint32_t idxL = std::distance(nmax_layers_vector.begin(), std::find(nmax_layers_vector.begin(), nmax_layers_vector.end(), nmax_layers));
    if(idxL >= nmax_layers_vector.size()) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Invalid number of layers %d\n", nmax_layers);
        std::exit(Env::finalize());
    }
    std::string category_file = layerFile_prefix + "/neuron" + std::to_string(nneurons) + "-l" + std::to_string(nmax_layers);
    category_file += (input_type == INPUT_TYPE::_TEXT_) ? "-categories.tsv" : "-categories.bin";
	CATEGORY_TYPE category_type = CATEGORY_TYPE::_NONZERO_INSTANCES_ONLY_;
	
	std::vector<std::string> layer_files;
	for(uint32_t i = 0; i < nmax_layers; i++) {
		std::string layer_file = layerFile_prefix + "/neuron" + std::to_string(nneurons) + "/n" + std::to_string(nneurons) + "-l" + std::to_string(i+1);
		layer_file += (input_type == INPUT_TYPE::_TEXT_) ? ".tsv" : ".bin";
		layer_files.push_back(layer_file);
	}
	
	std::vector<std::string> bias_files;
	std::vector<WGT> bias_vector = {-0.3,-0.35,-0.4,-0.45};
	WGT biasValue = bias_vector[idxN];
	BIAS_TYPE bias_type = BIAS_TYPE::_CONSTANT_;
	
	int x = atoi(argv[10]);
    PARALLELISM_TYPE parallelism_type = (PARALLELISM_TYPE)x;
    if(parallelism_type >= (PARALLELISM_TYPE::_SIZE_)) {
        Logging::print(Logging::LOG_LEVEL::FATAL, "Incorrect parallelism type\n");
        std::exit(Env::finalize());
    }
	
	COMPRESSED_FORMAT compression_type = COMPRESSED_FORMAT::_CSC_;
	HASHING_TYPE hashing_type = HASHING_TYPE::_BOTH_;
	
    Net<WGT> N(ninstances, nneurons, nmax_layers, 
			   feature_file,
			   category_type, category_file, 
			   layer_files, 
			   bias_type, bias_files, biasValue,
			   parallelism_type, compression_type, hashing_type, input_type);
    
    return(Env::finalize());
}