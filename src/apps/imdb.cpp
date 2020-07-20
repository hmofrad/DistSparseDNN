/*
 * imdb.cpp: Sparse DNN inference for IMDB Large Movie Review Dataset
 * [https://www.tensorflow.org/datasets/catalog/imdb_reviews]
 * (c) Mohammad Hasanzadeh Mofrad, 2020
 * (e) m.hasanzadeh.mofrad@gmail.com
 */
 
// make clean && make && time mpirun.mpich -np 2 bin/./cifar -i 25000 10000 2 /zfs1/cs3580_2017F/moh18/dnn/imdb/bin/ -n 2048 30 /zfs1/cs3580_2017F/moh18/dnn/imdb/bin/ -c 1 1 -p 0 -h 3

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
const std::string classifier = "sigmoid";

int main(int argc, char **argv) {
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
    Env::read_options(argc, argv, "IMDB", "Customized DNN", 
                      input_ninstances, input_nfeatures, ncategories, input_path, 
                      nneurons, nmax_layers, layers_path, 
                      ci, cl, p, h);
    FILE_TYPE file_type = FILE_TYPE::_BINARY_;
    
    std::string input_file = input_path + "/input";
    input_file += (file_type == FILE_TYPE::_TEXT_) ? ".txt" : ".bin";
    
    std::string category_file = layers_path;
    category_file += (file_type == FILE_TYPE::_TEXT_) ? "predictions.txt" : "predictions.bin";
    VALUE_TYPE category_type = VALUE_TYPE::_INSTANCE_AND_VALUE_PAIRS_;
    
    std::vector<std::string> layer_files;
    for(uint32_t i = 0; i < nmax_layers; i++) {
        std::string layer_file = layers_path + "/weights" + std::to_string(i);
        layer_file += (file_type == FILE_TYPE::_TEXT_) ? ".txt" : ".bin";
        layer_files.push_back(layer_file);
    }
    
    std::vector<std::string> bias_files;
    for(uint32_t i = 0; i < nmax_layers; i++) {
    std::string bias_file = layers_path + "/bias" + std::to_string(i);
    bias_file += (file_type == FILE_TYPE::_TEXT_) ? ".txt" : ".bin";
    bias_files.push_back(bias_file);
    }
    WGT bias_value = 0;
    VALUE_TYPE bias_type = VALUE_TYPE::_INSTANCE_AND_VALUE_PAIRS_;

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