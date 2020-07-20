/*
 * radixnet.cpp: Radix-Net sparse DNN inference for MNIST dataset
 * [http://graphchallenge.mit.edu/]
 * (c) Mohammad Hasanzadeh Mofrad, 2020
 * (e) m.hasanzadeh.mofrad@gmail.com
 */
 
// make clean && make && time mpirun.mpich -np 4 bin/./radixnet -i 1000 1024 0 data/radixnet/bin/MNIST -n 1024 120 data/radixnet/bin/DNN -c 1 1 -m 0 -p 0 -h 1

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
WGT relu(WGT w) {return (w < 0) ? 0 : (w > 32) ? 32 : w;}
const std::string classifier = "softmax";
std::vector<uint32_t> nneurons_vector = {1024, 4096, 16384, 65536};
std::vector<uint32_t> nmax_layers_vector = {120, 480, 1920};
std::vector<WGT> bias_vector = {-0.3,-0.35,-0.4,-0.45};

int main(int argc, char** argv) {
    Logging::enabled = true;
    int status = Env::init();
    if(status) {
        Logging::print(Logging::LOG_LEVEL::FATAL, "Failure to initialize MPI environment\n");
        std::exit(Env::finalize());   
    }
    uint32_t input_ninstances, input_nfeatures, ncategories;
    uint32_t nneurons, nmax_layers;
    std::string input_path, layers_path;
    uint32_t ci, cl, p, h;
    Env::read_options(argc, argv, "MNIST", "Radix-Net", 
                      input_ninstances, input_nfeatures, ncategories, input_path, 
                      nneurons, nmax_layers, layers_path, 
                      ci, cl, p, h);
    FILE_TYPE file_type = FILE_TYPE::_BINARY_;

    std::string input_file = input_path + "/sparse-images-" + std::to_string(nneurons);
    input_file += (file_type == FILE_TYPE::_TEXT_) ? ".tsv" : ".bin";

    std::string category_file = layers_path + "/neuron" + std::to_string(nneurons) + "-l" + std::to_string(nmax_layers);
    category_file += (file_type == FILE_TYPE::_TEXT_) ? "-categories.tsv" : "-categories.bin";
    VALUE_TYPE category_type = VALUE_TYPE::_NONZERO_INSTANCES_ONLY_;
    
    std::vector<std::string> layer_files;
    for(uint32_t i = 0; i < nmax_layers; i++) {
        std::string layer_file = layers_path + "/neuron" + std::to_string(nneurons) + "/n" + std::to_string(nneurons) + "-l" + std::to_string(i+1);
        layer_file += (file_type == FILE_TYPE::_TEXT_) ? ".tsv" : ".bin";
        layer_files.push_back(layer_file);
    }
    
    std::vector<std::string> bias_files;
    uint32_t index = std::distance(nneurons_vector.begin(), std::find(nneurons_vector.begin(), nneurons_vector.end(), nneurons));
    WGT bias_value = bias_vector[index];
    VALUE_TYPE bias_type = VALUE_TYPE::_CONSTANT_;
    
    COMPRESSED_FORMAT input_compression_type = (COMPRESSED_FORMAT)ci;
    COMPRESSED_FORMAT layer_compression_type = (COMPRESSED_FORMAT)cl;
    PARALLELISM_TYPE parallelism_type = (PARALLELISM_TYPE)p;
    HASHING_TYPE hashing_type = (HASHING_TYPE)h;
    
    if((input_compression_type >= (COMPRESSED_FORMAT::_C_SIZE_)) or 
       (layer_compression_type >= (COMPRESSED_FORMAT::_C_SIZE_)) or
       (hashing_type >= (HASHING_TYPE::_H_SIZE_)) or
       (parallelism_type >= (PARALLELISM_TYPE::_P_SIZE_))) {
        Logging::print(Logging::LOG_LEVEL::FATAL, "Incorrect parallelism,compression, or hashing type\n");
        std::exit(Env::finalize());
    }

    Net<WGT> N(input_ninstances, input_nfeatures, input_file,
               ncategories, category_type, category_file, 
               nneurons, nmax_layers, layer_files, 
               bias_value, bias_type, bias_files,
               noop, relu, classifier,
               file_type, 
               input_compression_type, layer_compression_type, parallelism_type, hashing_type);
    
    return(Env::finalize());
}
