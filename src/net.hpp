/*
 * net.hpp: Neural network base class 
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */
 
#ifndef NET_HPP
#define NET_HPP

#include "triple.hpp"
#include "tiling.hpp"
#include "dvec.hpp"

template<typename Weight>
class Net {
    public:
        Net() {};
        ~Net() {};
        
        Net(const uint32_t Nneurons_, const std::string inputFile_prefix, 
            const uint32_t maxLayers_, const std::string layerFile_prefix,
            const INPUT_TYPE input_type = INPUT_TYPE::_BINARY_,
            const TILING_TYPE tiling_type_ = TILING_TYPE::_1D_ROW_,
            const COMPRESSED_FORMAT compression_type = COMPRESSED_FORMAT::_CSR_);
        
        std::unique_ptr<struct Tiling<Weight>> tiling = nullptr;
        std::vector<uint32_t> trueCategories;
        std::vector<std::unique_ptr<struct Tiling<Weight>>> layers;
        //std::vector<std::unique_ptr<struct DVec<Weight>>> biasDenseVecs;
        std::vector<std::vector<Weight>> biasDenseVecs;
        
        uint32_t Nneurons;
        Weight biasValue;
        uint32_t maxLayers;
};

template<typename Weight>
Net<Weight>::Net(const uint32_t Nneurons_, const std::string inputFile_prefix, 
                 const uint32_t maxLayers_, const std::string layerFile_prefix,
                 const INPUT_TYPE input_type, const TILING_TYPE tiling_type, 
                 const COMPRESSED_FORMAT compression_type) : Nneurons(Nneurons_), maxLayers(maxLayers_) {
                     
    std::vector<Weight> neuralNetBias = {-0.3,-0.35,-0.4,-0.45};
    std::vector<uint32_t> NneuronsVector = {1024, 4096, 16384, 65536};    
    uint32_t idxN = std::distance(NneuronsVector.begin(), std::find(NneuronsVector.begin(), NneuronsVector.end(), Nneurons));
    if(idxN >= NneuronsVector.size()) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Invalid number of neurons %d", Nneurons);
        std::exit(Env::finalize());
    }    
    Weight biasValue = neuralNetBias[idxN];
    
    std::string feature_file = inputFile_prefix + "/sparse-images-" + std::to_string(Nneurons);
    feature_file += (input_type == INPUT_TYPE::_TEXT_) ? ".tsv" : ".bin";
    
    tiling = std::move(std::make_unique<Tiling<Weight>>((Env::nranks * Env::nranks), Env::nranks, Env::nranks, Env::nranks,
                        feature_file, input_type, tiling_type, compression_type));
    
    
    std::vector<uint32_t> maxLayersVector = {120, 480, 1920};
    uint32_t idxL = std::distance(maxLayersVector.begin(), std::find(maxLayersVector.begin(), maxLayersVector.end(), maxLayers));
    if(idxL >= maxLayersVector.size()) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Invalid number of layers %d", maxLayers);
        std::exit(Env::finalize());
    }

    std::string categoryFile = layerFile_prefix + "/neuron" + std::to_string(Nneurons) + "-l" + std::to_string(maxLayers);
    categoryFile += (input_type == INPUT_TYPE::_TEXT_) ? "-categories.tsv" : "-categories.bin";
    //printf("INFO: Start reading the category file %s\n", categoryFile.c_str());
    
    if(INPUT_TYPE::_TEXT_ == input_type) {
        IO::text_file_categories(categoryFile, trueCategories, tiling->tile_height);
    }
    else {
        IO::binary_file_categories(categoryFile, trueCategories, tiling->tile_height);
    }
    
    /*
    int k = 0;
   // if(Env::rank == 3) {
        for(auto& c: trueCategories) {
            if(c == 1) {
                k++;  
               // printf("%d %d %d\n", c, c + (Env::rank * tiling->tile_height), k);
            }
        }
    //}
    */
    //printf("2. %d %d\n", Env::rank, k);
    


    Logging::enabled = false; 
    layers.resize(maxLayers);
    biasDenseVecs.resize(maxLayers);
    for(uint32_t i = 0; i < maxLayers; i++) {
        std::string layerFile = layerFile_prefix + "/neuron" + std::to_string(Nneurons) + "/n" + std::to_string(Nneurons) + "-l" + std::to_string(i+1);
        layerFile += (input_type == INPUT_TYPE::_TEXT_) ? ".tsv" : ".bin";
        
        //layers.push_back(std::move(std::make_unique<Tiling<Weight>>((Env::nranks * Env::nranks), Env::nranks, Env::nranks, Env::nranks,
          //               layerFile, input_type, tiling_type, compression_type)));
        layers[i] = std::move(std::make_unique<Tiling<Weight>>((Env::nranks * Env::nranks), Env::nranks, Env::nranks, Env::nranks,
                         layerFile, input_type, tiling_type, compression_type)); 
        
        //for(int32_t j = 0; j < Env::nranks; j++) {
        //biasDenseVecs.push_back(std::move(std::make_unique<struct DVec<Weight>>(layers[i]->tile_height, biasValue)));
        //biasDenseVecs.push_back(std
        biasDenseVecs[i] = std::vector<Weight>(tiling->tile_height, biasValue);
        
       //if(!Env::rank) printf("%s %lu\n", layerFile.c_str(), biasDenseVecs[i]->nitems);
       
    //   break;
    }
    Logging::enabled = true;
    
    


    
    
    //printf("INFO: Start reading %d layer files\n", maxLayers);
    //auto start = std::chrono::high_resolution_clock::now();
    //for(uint32_t i = 0; i < maxLayers; i++) {  
      //  std::string layerFile = ((std::string) argv[6]) + "/neuron" + std::to_string(Nneurons) + "/n" + std::to_string(Nneurons) + "-l" + std::to_string(i+1) + ".tsv";
    //}
      //           }
    
    /*
    std::string categoryFile = ((std::string) argv[6]) + "/neuron" + std::to_string(Nneurons) + "-l" + std::to_string(maxLayers) + "-categories.tsv";
    printf("INFO: Start reading the category file %s\n", categoryFile.c_str());
    */
    
    
}

#endif 