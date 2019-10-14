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


#include "radixnet.h"
#include "env.hpp"
#include "log.hpp"
#include "triple.hpp"

/*


#include <chrono>

#include "Triple.hpp"
#include "DenseVec.hpp"
#include "SparseMat.hpp"
#include "InferenceReLU.cpp"

*/


//#define HAS_EDGE_WEIGHT
// make clean && make && mpirun -np 2 bin/./radixnet -n 1024 -l 120 data/MNIST data/DNN


int main(int argc, char **argv) {
    Logging::enabled = true;
    int status = Env::init();
    if(status) {
        Logging::print(Logging::LOGLEVELS::FATAL, "Failure to initialize MPI environment\n");
        std::exit(Env::finalize());
        //int ret = MPI_Finalize();
        //std::exit(1);         
    }

    if(argc != 7) {
        Logging::print(Logging::LOGLEVELS::ERROR, "USAGE = %s -n <Nneurons> -l <maxLayers> <path_to_input> <path_to_dnn>\n", argv[0]);
        std::exit(Env::finalize());     
    }
    
    Logging::print(Logging::LOGLEVELS::INFO, "Number of MPI ranks = %d, number of threads per rank = %d\n", Env::nranks, Env::nthreads);
    Logging::print(Logging::LOGLEVELS::INFO, "Radix-Net sparse DNN for MNIST dataset Implementation\n");
    
    std::vector<WGT> neuralNetBias = {-0.3,-0.35,-0.4,-0.45};
    uint32_t Nneurons = atoi(argv[2]);
    std::vector<uint32_t> NneuronsVector = {1024, 4096, 16384, 65536};
    uint32_t idxN = std::distance(NneuronsVector.begin(), std::find(NneuronsVector.begin(), NneuronsVector.end(), Nneurons));
    if(idxN >= NneuronsVector.size()) {
        Logging::print(Logging::LOGLEVELS::ERROR, "Invalid number of neurons/layer %d", Nneurons);
        //fprintf(stderr, "Invalid number of neurons/layer %d\n", Nneurons);
        std::exit(Env::finalize());
    }    
    WGT biasValue = neuralNetBias[idxN];
    
    std::string featureFile = ((std::string) argv[5]) + "/sparse-images-" + std::to_string(Nneurons) + ".tsv";
    Logging::print(Logging::LOGLEVELS::INFO, "Start reading the feature file %s\n", featureFile.c_str());
    
    /*
    std::ifstream fin(featureFile.c_str());
    if(not fin.is_open()) {
        Logging::print(Logging::LOGLEVELS::ERROR, "Opening %s\n", featureFile.c_str());
        
        std::exit(Env::finalize());
    }
    
    
    uint64_t nrowsFeatures = 0; 
    uint64_t ncolsFeatures = 0;
    
    struct Triple<WGT> featuresTriple;
    std::string line;
    std::istringstream iss;
    
    uint64_t nlines = 0;
    while (std::getline(fin, line)) {
        nlines++;
    }
    
    fin.clear();
    fin.seekg(0, std::ios_base::beg);
    
    uint64_t share = nlines / Env::nranks;
    uint64_t start_line = Env::rank * share;
    uint64_t end_line = (Env::rank != Env::nranks - 1) ? ((Env::rank + 1) * share) : nlines;
    share = (Env::rank == Env::nranks - 1) ? end_line - start_line : share;
    uint64_t curr_line = 0;
    
    
    while(curr_line < start_line) {
        std::getline(fin, line);
        curr_line++;
    }
    
    //printf("%d: share=%lu start=%lu end=%lu curr=%lu\n", Env::rank, share, start_line, end_line, curr_line);
    
    std::vector<struct Triple<WGT>> featuresTriples(share);
    
    #pragma omp parallel reduction(max : nrowsFeatures, ncolsFeatures)
    {
        int nthreads = Env::nthreads; 
        int tid = omp_get_thread_num();
        
        uint64_t share_t = share / nthreads; 
        uint64_t start_line_t = curr_line + (tid * share_t);
        uint64_t end_line_t = (tid != Env::nthreads - 1) ? curr_line + ((tid + 1) * share_t) : end_line;
        share_t = (tid == Env::nthreads - 1) ? end_line_t - start_line_t : share_t;
        uint64_t curr_line_t = 0;
        std::string line_t;
        std::ifstream fin_t(featureFile.c_str());
        if(not fin_t.is_open()) {
            Logging::print(Logging::LOGLEVELS::ERROR, "Opening %s\n", featureFile.c_str());
            //fprintf(stderr, "Error: Opening %s\n", featureFile.c_str());
            std::exit(Env::finalize());
        }
        fin_t.seekg(fin.tellg(), std::ios_base::beg);
        curr_line_t = curr_line;
        
        while(curr_line_t < start_line_t) {
            std::getline(fin_t, line_t);
            curr_line_t++;
        }
        
        //printf("%d: %d / %d share=%lu start=%lu end=%lu curr=%lu\n", Env::rank, tid, nthreads, share_t, start_line_t, end_line_t, curr_line_t);
        
        struct Triple<WGT> featuresTriple1;
        std::istringstream iss_t;
        while (curr_line_t < end_line_t) {
            std::getline(fin_t, line_t);
            iss_t.clear();
            iss_t.str(line_t);
            iss_t >> featuresTriple.row >> featuresTriple.col >> featuresTriple.weight;
            //long int d = (curr_line_t - curr_line);
            //printf("%d %d %lu %lu %lu\n", Env::rank, tid, curr_line, curr_line_t, d);
            featuresTriples[curr_line_t - curr_line] = featuresTriple;
            
            if(featuresTriple.row > nrowsFeatures)
                nrowsFeatures = featuresTriple.row;
            if(featuresTriple.col > ncolsFeatures)
                ncolsFeatures = featuresTriple.col;
            
           // if(!Env::rank)
             //   printf("%d %d %f\n", featuresTriple.row, featuresTriple.col, featuresTriple.weight);
            curr_line_t++;
        }
        
        fin_t.close();
    }

    fin.close();
    
        
        
    
    //printf("%d %lu\n", Env::rank, current );
    
    
    uint64_t reducer = 0;
    MPI_Allreduce(&ncolsFeatures, &reducer, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);
    ncolsFeatures = reducer;
    //uint64_t global_max1 = nrowsFeatures;
    MPI_Allreduce(&nrowsFeatures, &reducer, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);
    nrowsFeatures = reducer;
    
    uint64_t nnzFeatures = featuresTriples.size();
    MPI_Allreduce(&nnzFeatures, &reducer, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
    nnzFeatures = reducer;
    
    //  printf("Rank=%d curr=%lu nrowsFeatures=%lu ncolsFeatures=%lu global_max=%lu\n", Env::rank, curr_line, nrowsFeatures, ncolsFeatures, global_max);
    //sleep(2);
    //MPI_Barrier(MPI_COMM_WORLD);
    
    Logging::print(Logging::LOGLEVELS::INFO, "Done  reading the feature file %s\n", featureFile.c_str());
    Logging::print(Logging::LOGLEVELS::INFO, "Feature file is [%lu x %lu], nnz=%lu\n", nrowsFeatures, ncolsFeatures, nnzFeatures);
    */
    
    
    return(Env::finalize());
    
    
    
    //fin.seekg(start_line, std::ios::cur);
    
    //Env::finalize(0);
    /*
    while (std::getline(fin, line)) {
        iss.clear();
        iss.str(line);
        iss >> featuresTriple.row >> featuresTriple.col >> featuresTriple.weight;
        featuresTriples.push_back(featuresTriple);
        if(featuresTriple.row > nrowsFeatures)
            nrowsFeatures = featuresTriple.row;
        if(featuresTriple.col > ncolsFeatures)
            ncolsFeatures = featuresTriple.col;
    }
    fin.close();
    printf("INFO: Done  reading the features file %s\n", featureFile.c_str());
    printf("INFO: Features file is %lu x %lu, nnz=%lu\n", nrowsFeatures, ncolsFeatures, featuresTriples.size());
    uint64_t NfeatureVectors = nrowsFeatures;
    
    
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
