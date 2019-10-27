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
#include "spops.hpp"

template<typename Weight>
class Net {
    public:
        Net() {};
        ~Net() {};
        
        Net(const uint32_t NinputInstanses_, const uint32_t Nneurons_, 
            const std::string inputFile_prefix, const uint32_t maxLayers_, const std::string layerFile_prefix,
            const INPUT_TYPE input_type = INPUT_TYPE::_BINARY_,
            const TILING_TYPE tiling_type_ = TILING_TYPE::_1D_ROW_,
            const COMPRESSED_FORMAT compression_type = COMPRESSED_FORMAT::_CSR_);
        
        void inferenceReLU(TILING_TYPE tiling_type);
        
        std::unique_ptr<struct Tiling<Weight>> inputFeatures = nullptr;
        std::vector<uint32_t> trueCategories;
        std::vector<std::unique_ptr<struct Tiling<Weight>>> layers;
        std::vector<std::vector<Weight>> biasDenseVecs;
        std::vector<std::vector<Weight>> spaDenseVec;

        uint32_t NinputInstanses;        
        uint32_t Nneurons;
        Weight biasValue;
        uint32_t maxLayers;
};

template<typename Weight>
Net<Weight>::Net(const uint32_t NinputInstanses_, const uint32_t Nneurons_, const std::string inputFile_prefix, 
                 const uint32_t maxLayers_, const std::string layerFile_prefix,
                 const INPUT_TYPE input_type, const TILING_TYPE tiling_type, 
                 const COMPRESSED_FORMAT compression_type) : NinputInstanses(NinputInstanses_), Nneurons(Nneurons_), maxLayers(maxLayers_) {
    
        
    
    /*
    int i = 0;
    layers.resize(maxLayers);
    std::string layerFile = layerFile_prefix + "/neuron" + std::to_string(Nneurons) + "/n" + std::to_string(Nneurons) + "-l" + std::to_string(i+1);
    layerFile += (input_type == INPUT_TYPE::_TEXT_) ? ".tsv" : ".bin";
    //layers[i] = std::move(std::make_unique<Tiling<Weight>>(1, 1, 1, 1, layerFile, input_type, tiling_type, compression_type)); 
    layers[i] = std::move(std::make_unique<Tiling<Weight>>(Env::nranks, 1, Env::nranks, 1, (Nneurons+1), (Nneurons+1), layerFile, input_type, tiling_type, compression_type)); 
    //printf("EXITING\n");
    //Env::finalize();
    //exit(0);
    */
    
    
    
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Processing input feature file for %d neurons.\n", Nneurons);  
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
    
    inputFeatures = std::move(std::make_unique<Tiling<Weight>>((Env::nranks * Env::nranks), Env::nranks, Env::nranks, Env::nranks, (NinputInstanses+1), (Nneurons+1),
                        feature_file, input_type, tiling_type, compression_type));

    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Processing the category files for %d neurons and %d layers.\n", Nneurons, maxLayers); 
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
        IO::text_file_categories(categoryFile, trueCategories, inputFeatures->tile_height);
    }
    else {
        IO::binary_file_categories(categoryFile, trueCategories, inputFeatures->tile_height);
    }

    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Processing %d layer files (silent).\n", maxLayers); 
    //Logging::enabled = false; 
    maxLayers = 1 ;
    layers.resize(maxLayers);
    biasDenseVecs.resize(maxLayers);
    for(uint32_t i = 0; i < maxLayers; i++) {
        std::string layerFile = layerFile_prefix + "/neuron" + std::to_string(Nneurons) + "/n" + std::to_string(Nneurons) + "-l" + std::to_string(i+1);
        layerFile += (input_type == INPUT_TYPE::_TEXT_) ? ".tsv" : ".bin";
        //layers[i] = std::move(std::make_unique<Tiling<Weight>>((Env::nranks * Env::nranks), Env::nranks, Env::nranks, Env::nranks,
        //                 layerFile, input_type, tiling_type, compression_type)); 
        //biasDenseVecs[i] = std::vector<Weight>(inputFeatures->tile_height, biasValue);
        //layers[i] = std::move(std::make_unique<Tiling<Weight>>(Env::nranks, 1, Env::nranks, 1, (Nneurons+1), (Nneurons+1), layerFile, input_type, tiling_type, compression_type)); 
        
        if(tiling_type == TILING_TYPE::_1D_ROW_) {
            layers[i] = std::move(std::make_unique<Tiling<Weight>>(Env::nranks, 1, Env::nranks, 1, 
            inputFeatures->ncols, inputFeatures->ncols, layerFile, input_type, tiling_type, compression_type)); 
        }
        else if(tiling_type == TILING_TYPE::_1D_COL_) {
            layers[i] = std::move(std::make_unique<Tiling<Weight>>(Env::nranks, Env::nranks, 1, 1, 
            inputFeatures->ncols, inputFeatures->ncols, layerFile, input_type, tiling_type, compression_type)); 
        }
        else if (tiling_type == TILING_TYPE::_2D_) {
            layers[i] = std::move(std::make_unique<Tiling<Weight>>((Env::nranks * Env::nranks), Env::nranks, Env::nranks, 1, 
            inputFeatures->ncols, inputFeatures->ncols, layerFile, input_type, tiling_type, compression_type)); 
        }
        
        //break;
        biasDenseVecs[i] = std::vector<Weight>(inputFeatures->ncols, biasValue);
    }
    //spaDenseVec.resize(inputFeatures->tile_height);
    if(tiling_type == TILING_TYPE::_1D_ROW_) {
        spaDenseVec.resize(inputFeatures->rank_ncolgrps);
        for(uint32_t i = 0; i < inputFeatures->rank_ncolgrps; i++) {
            spaDenseVec[i].resize(inputFeatures->tile_height);
        }
    }
    else if(tiling_type == TILING_TYPE::_1D_COL_) {
        spaDenseVec.resize(inputFeatures->rank_nrowgrps);
        for(uint32_t i = 0; i < inputFeatures->rank_nrowgrps; i++) {
            spaDenseVec[i].resize(inputFeatures->tile_width);
        }
    }
    
    
    
    Logging::enabled = true;
    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Running the inferenceReLU method.\n"); 
    
    inferenceReLU(tiling_type);
    
    //std::tie(nrows, ncols, nnz) =  spmm_sym();
    
    /*    
    start = std::chrono::high_resolution_clock::now();
    inferenceReLU<WGT>(layersSpMat, biasesDenseVec, featuresSpMat, spa_VEC); // Train DNN 
    finish = std::chrono::high_resolution_clock::now();
    WGT challengeRunTime = (WGT)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish-start).count())/1e9;
    WGT challengeRunRate = NfeatureVectors * (DNNedges/challengeRunTime);
    printf("INFO: Run time (sec): %f, run rate (edges/sec): %f\n", challengeRunTime, challengeRunRate);
    
    validate_prediction<WGT>(featuresSpMat, trueCategories); // Test DNN
    */
    
    
    
}

template<typename Weight>
void Net<Weight>::inferenceReLU(TILING_TYPE tiling_type) {
    uint32_t nrows = 0;
    uint32_t ncols = 0;
    uint64_t nnz = 0;
    
    if(!Env::rank) {
        if(tiling_type == TILING_TYPE::_1D_ROW_) {
            int l = 0;
            for (uint32_t i = 0; i < inputFeatures->nrowgrps; i++) {
                for (uint32_t j = 0; j < inputFeatures->ncolgrps; j++) {
                    auto& tileA = inputFeatures->tiles[i][j];
                    if(Env::rank == tileA.rank) {
                        auto& spmatA = tileA.spmat;
                        auto& tileB = layers[l]->tiles[0][j];
                        auto& spmatB = tileB.spmat;
                        auto& spaS = spaDenseVec[j];
                        std::tie(nrows, ncols, nnz) =  spmm_sym(spmatA, spmatB, spaS);
                    }
                }
            }
        }
    }
    
    
    //std::tie(nrows, ncols, nnz) =  spmm_sym();
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: inferenceReLU? %d %d %lu\n", nrows, ncols, nnz); 
    
}

    


#endif 