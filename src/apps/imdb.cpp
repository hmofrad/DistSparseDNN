/*
 * imdb.cpp: Sparse DNN inference for IMDB Large Movie Review Dataset
 * [https://www.tensorflow.org/datasets/catalog/imdb_reviews]
 * (c) Mohammad Hasanzadeh Mofrad, 2020
 * (e) m.hasanzadeh.mofrad@gmail.com
 */
 
// make clean && make && time mpirun.mpich -np 1 bin/./imdb -m 25000 10000 -n 1024 -l 3 -c 2 scripts/data/imdb/bin/ scripts/data/imdb/bin/ -p 0

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

using WGT = float;
WGT noop(WGT w) {return w;}
WGT relu(WGT w) {return (w < 0) ? 0 : w;}
const std::string classifier = "softmax";

int main(int argc, char **argv) {
    Logging::enabled = true;
    int status = Env::init();
    if(status) {
        Logging::print(Logging::LOG_LEVEL::FATAL, "Failure to initialize MPI environment\n");
        std::exit(Env::finalize());   
    }

    if(argc != 14) {
        Logging::print(Logging::LOG_LEVEL::FATAL, "USAGE = %s -m <input_ninstances input_nfeatures> -n <nneurons> -l <nmax_layers> -c <ncategories> <path_to_input> <path_to_dnn> -p <parallelism_type>\n", argv[0]);
        std::exit(Env::finalize());     
    }
    
	std::string script = "sparse_mnist";
    Logging::print(Logging::LOG_LEVEL::INFO, "%s sparse DNN for MNIST dataset\n", script.c_str());
    Logging::print(Logging::LOG_LEVEL::INFO, "Machines = %d, MPI ranks  = %d, Threads per rank = %d\n", Env::nmachines, Env::nranks, Env::nthreads);
    Logging::print(Logging::LOG_LEVEL::INFO, "Sockets  = %d, Processors = %d\n", Env::nsockets, Env::ncores);
    if(Env::NUMA_ALLOC) Logging::print(Logging::LOG_LEVEL::INFO, "NUMA is enabled.\n", Env::NUMA_ALLOC);
    else Logging::print(Logging::LOG_LEVEL::WARN, "NUMA is disabled.\n", Env::NUMA_ALLOC);
    
	uint32_t input_ninstances = atoi(argv[2]);
	uint32_t input_nfeatures = atoi(argv[3]);
	uint32_t nneurons = atoi(argv[5]);
	uint32_t nmax_layers = atoi(argv[7]);
	uint32_t ncategories = atoi(argv[9]);
	std::string feature_file_prefix = ((std::string) argv[10]);
	std::string layer_file_prefix = ((std::string) argv[11]);
	INPUT_TYPE input_type = INPUT_TYPE::_BINARY_;
	
    std::vector<uint32_t> nneurons_vector = {2048};    
    uint32_t idxN = std::distance(nneurons_vector.begin(), std::find(nneurons_vector.begin(), nneurons_vector.end(), nneurons));
    if(idxN >= nneurons_vector.size()) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Invalid number of neurons %d\n", nneurons);
        std::exit(Env::finalize());
    }    
    
    std::string feature_file = feature_file_prefix + "/input";
    feature_file += (input_type == INPUT_TYPE::_TEXT_) ? ".txt" : ".bin";
	
    std::vector<uint32_t> nmax_layers_vector = {30};
    uint32_t idxL = std::distance(nmax_layers_vector.begin(), std::find(nmax_layers_vector.begin(), nmax_layers_vector.end(), nmax_layers));
    if(idxL >= nmax_layers_vector.size()) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Invalid number of layers %d\n", nmax_layers);
        std::exit(Env::finalize());
    }
    std::string category_file = layer_file_prefix;
    category_file += (input_type == INPUT_TYPE::_TEXT_) ? "predictions.txt" : "predictions.bin";
    VALUE_TYPE category_type = VALUE_TYPE::_INSTANCE_AND_VALUE_PAIRS_;
	
    std::vector<std::string> layer_files;
    for(uint32_t i = 0; i < nmax_layers; i++) {
	std::string layer_file = layer_file_prefix + "/weights" + std::to_string(i);
	layer_file += (input_type == INPUT_TYPE::_TEXT_) ? ".txt" : ".bin";
	layer_files.push_back(layer_file);
    }
	
    std::vector<std::string> bias_files;
    for(uint32_t i = 0; i < nmax_layers; i++) {
	std::string bias_file = layer_file_prefix + "/bias" + std::to_string(i);
	bias_file += (input_type == INPUT_TYPE::_TEXT_) ? ".txt" : ".bin";
	bias_files.push_back(bias_file);
    }
    WGT bias_value = 0;
    VALUE_TYPE bias_type = VALUE_TYPE::_INSTANCE_AND_VALUE_PAIRS_;
	
    int x = atoi(argv[13]);
    PARALLELISM_TYPE parallelism_type = (PARALLELISM_TYPE)x;
    if(parallelism_type >= (PARALLELISM_TYPE::_SIZE_)) {
        Logging::print(Logging::LOG_LEVEL::FATAL, "Incorrect parallelism type\n");
        std::exit(Env::finalize());
    }
	
    COMPRESSED_FORMAT compression_type = COMPRESSED_FORMAT::_CSR_;
    HASHING_TYPE hashing_type = HASHING_TYPE::_NO_;
	
    Net<WGT> N(input_ninstances, input_nfeatures, feature_file,
			   nneurons, nmax_layers, layer_files, 
			   bias_value, bias_type, bias_files, 
			   ncategories, category_type, category_file, 
			   noop, relu, classifier,
			   input_type, parallelism_type, compression_type, hashing_type);
    
    return(Env::finalize());
}
