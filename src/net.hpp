/*
 * net.hpp: Neural network base class 
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */
 
#ifndef NET_HPP
#define NET_HPP

#include "triple.hpp"
#include "tiling.hpp"

template<typename Weight>
class Net {
    public:
        Net() {};
        ~Net();
        
        Net(const uint32_t Nneurons_, const std::string inputFile_prefix, 
            const TILING_TYPE tiling_type_ = TILING_TYPE::_1D_ROW_,
            const INPUT_TYPE input_type = INPUT_TYPE::_BINARY_);
        
        std::vector<struct Triple<Weight>> triples;
        Tiling<Weight>* tiling;
        
        uint32_t Nneurons;
        Weight biasValue;
};

template<typename Weight>
Net<Weight>::Net(const uint32_t Nneurons_, const std::string inputFile_prefix,
                 const TILING_TYPE tiling_type, const INPUT_TYPE input_type) : Nneurons(Nneurons_) {
    std::vector<Weight> neuralNetBias = {-0.3,-0.35,-0.4,-0.45};
    std::vector<uint32_t> NneuronsVector = {1024, 4096, 16384, 65536};    
    uint32_t idxN = std::distance(NneuronsVector.begin(), std::find(NneuronsVector.begin(), NneuronsVector.end(), Nneurons));
    if(idxN >= NneuronsVector.size()) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Invalid number of neurons/layer %d", Nneurons);
        std::exit(Env::finalize());
    }    
    Weight biasValue = neuralNetBias[idxN];
    
    std::string feature_file = inputFile_prefix + "/sparse-images-" + std::to_string(Nneurons);
    feature_file += (input_type == INPUT_TYPE::_TEXT_) ? ".tsv" : ".bin";
    //printf("%s\n", feature_file.c_str());
    
/*    if(input_type == INPUT_TYPE::_TEXT_) {
        featureFile = inputFile_prefix + "/sparse-images-" + std::to_string(Nneurons) + ".tsv";
    }
    else {
        featureFile = inputFile_prefix + "/sparse-images-" + std::to_string(Nneurons) + ".bin";
  
  }
  */
    tiling = new Tiling<Weight>(tiling_type, (Env::nranks * Env::nranks), Env::nranks, Env::nranks, Env::nranks, feature_file, input_type);
    //printf("tiling ptr %p\n",  tiling.spmat1);
    //delete tiling.spmat1;
    //tiling.Del();
    
    
    //printf("tiling is done\n");
}

template<typename Weight>
Net<Weight>::~Net() {
    //printf("NN destructor\n");
    delete tiling;
}


#endif 