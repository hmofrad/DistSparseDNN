/*
 * radixnet.cpp: Radix-Net sparse DNN for MNIST dataset
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

/*


#include <chrono>

#include "Triple.hpp"
#include "DenseVec.hpp"
#include "SparseMat.hpp"
#include "InferenceReLU.cpp"

*/


//#define HAS_EDGE_WEIGHT
// make clean && make && mpirun.mpich -np 2 bin/./radixnet -n 1024 -l 120 data/MNIST data/DNN


int main(int argc, char **argv) {
    Logging::enabled = true;
    int status = Env::init();
    if(status) {
        Logging::print(Logging::LOG_LEVEL::FATAL, "Failure to initialize MPI environment\n");
        std::exit(Env::finalize());
        //int ret = MPI_Finalize();
        //std::exit(1);         
    }

    if(argc != 7) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "USAGE = %s -n <Nneurons> -l <maxLayers> <path_to_input> <path_to_dnn> <Ninput_instances> <Ninput_features>\n", argv[0]);
        std::exit(Env::finalize());     
    }
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Radix-Net sparse DNN for MNIST dataset Implementation\n");
    Logging::print(Logging::LOG_LEVEL::INFO, "MPI ranks = %d, Threads per rank = %d\n", Env::nranks, Env::nthreads);

    Net<WGT> N(atoi(argv[2]), ((std::string) argv[5]), atoi(argv[4]), ((std::string) argv[6])) ;
    
    //delete N.tiling;
    //printf("Net is done\n");
    
    return(Env::finalize());

        /*
    struct CSC<WGT> *featuresSpMat = new struct CSC<WGT>((nrowsFeatures + 1), (Nneurons + 1), featuresTriples.size(), featuresTriples);
    featuresTriples.clear();
    featuresTriples.shrink_to_fit();
    
    uint32_t maxLayers = atoi(argv[4]);
    std::vector<uint32_t> maxLayersVector = {120, 480, 1920};
    std::ptrdiff_t idxL = std::distance(maxLayersVector.begin(), std::find(maxLayersVector.begin(), maxLayersVector.end(), maxLayers));
    if(idxL >= maxLayersVector.size()) {
        fprintf(stderr, "Invalid number of layers %d\n", maxLayers);
        exit(1);
    }    
    
    std::string categoryFile = ((std::string) argv[6]) + "/neuron" + std::to_string(Nneurons) + "-l" + std::to_string(maxLayers) + "-categories.tsv";
    printf("INFO: Start reading the category file %s\n", categoryFile.c_str());
    
    fin.clear();
    fin.open(categoryFile.c_str());
    if(!fin.is_open()) {
        fprintf(stderr, "Error: Opening %s\n", categoryFile.c_str());
        exit(1);
    }
    std::vector<uint32_t> trueCategories;
    uint32_t category = 0;
    while (std::getline(fin, line)) {
        iss.clear();
        iss.str(line);
        iss >> category;
        trueCategories.push_back(category);
    }
    fin.close();
    printf("INFO: Done  reading the category file %s\n", categoryFile.c_str());
    uint64_t Ncategories = trueCategories.size();
    printf("INFO: Number of categories %lu\n", Ncategories);

    uint64_t DNNedges = 0;
    
    std::vector<struct Triple<WGT>> layerTriples;
    struct Triple<WGT> layerTriple;  
    std::vector<struct CSC<WGT>*> layersSpMat;
    //std::vector<struct CompressedSpMat<WGT>*> layersSpMat;
    std::vector<struct DenseVec<WGT>*> biasesDenseVec;
    //maxLayers = 1;
    printf("INFO: Start reading %d layer files\n", maxLayers);
    auto start = std::chrono::high_resolution_clock::now();
    for(uint32_t i = 0; i < maxLayers; i++) {  
        std::string layerFile = ((std::string) argv[6]) + "/neuron" + std::to_string(Nneurons) + "/n" + std::to_string(Nneurons) + "-l" + std::to_string(i+1) + ".tsv";
        
        fin.clear();
        fin.open(layerFile.c_str());
        if(!fin.is_open()) {
            fprintf(stderr, "Error: Opening %s\n", layerFile.c_str());
            exit(1);
        }

        uint64_t nrows = 0;
        uint64_t ncols = 0;

        while (std::getline(fin, line)) {
            iss.clear();
            iss.str(line);
            iss >> layerTriple.row >> layerTriple.col >> layerTriple.weight;
            layerTriples.push_back(layerTriple);
            if(layerTriple.row > nrows)
                nrows = layerTriple.row;
            if(layerTriple.col > ncols)
                ncols = layerTriple.col;
        }
        fin.close();
        DNNedges += layerTriples.size();
        struct CSC<WGT> *layerSpMat = new struct CSC<WGT>((Nneurons + 1), (ncols + 1), layerTriples.size(), layerTriples);
        layersSpMat.push_back(layerSpMat);
        layerTriples.clear();
        layerTriples.shrink_to_fit();
        
        struct DenseVec<WGT> *biaseDenseVec = new struct DenseVec<WGT>((Nneurons + 1));
        auto &bias_A = biaseDenseVec->A;
        for(uint32_t j = 1; j < Nneurons+1; j++) {
            bias_A[j] = biasValue;
        }
        
        biasesDenseVec.push_back(biaseDenseVec);
    } 

    auto finish = std::chrono::high_resolution_clock::now();
    printf("INFO: Done  reading %d layer files\n", maxLayers);
    WGT readLayerTime = (WGT)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish-start).count())/1e9;
    WGT readLayerRate = (WGT) DNNedges/readLayerTime;
    printf("INFO: DNN neurons/layer: %d, layers:%d, edges:%lu\n", Nneurons, maxLayers, DNNedges);
    printf("INFO: Read time (sec): %f, read rate (edges/sec): %f\n", readLayerTime, readLayerRate);
    
    Env::init();
    std::vector<struct DenseVec<WGT>*> spa_VEC;
    for(uint32_t i = 0; i < Env::nthreads; i++) {
        struct DenseVec<WGT> *spa_DVEC = new struct DenseVec<WGT>(nrowsFeatures + 1);
        spa_VEC.push_back(spa_DVEC);
    }
    
    
    
    start = std::chrono::high_resolution_clock::now();
        inferenceReLU<WGT>(layersSpMat, biasesDenseVec, featuresSpMat, spa_VEC); // Train DNN 
    finish = std::chrono::high_resolution_clock::now();
    WGT challengeRunTime = (WGT)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish-start).count())/1e9;
    WGT challengeRunRate = NfeatureVectors * (DNNedges/challengeRunTime);
    printf("INFO: Run time (sec): %f, run rate (edges/sec): %f\n", challengeRunTime, challengeRunRate);
    
    validate_prediction<WGT>(featuresSpMat, trueCategories); // Test DNN
    
    delete featuresSpMat;
    for(uint32_t i = 0; i < maxLayers; i++) {  
        delete layersSpMat[i];
        delete biasesDenseVec[i];
    }
    layersSpMat.clear();
    layersSpMat.shrink_to_fit();
    biasesDenseVec.clear();
    biasesDenseVec.shrink_to_fit();
    
    for(uint32_t i = 0; i < Env::nthreads; i++) {
        delete spa_VEC[i];
    }
    spa_VEC.clear();
    spa_VEC.shrink_to_fit();
    
    return(0);
    */
}
