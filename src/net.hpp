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
        ~Net() {};
        
        Net(TILING_TYPE tiling_type_, uint32_t Nneurons_, std::string inputFile_prefix);
        
        std::vector<struct Triple<Weight>> triples;
        Tiling<Weight> tiling;
        
        uint32_t Nneurons;
        Weight biasValue;
};

template<typename Weight>
Net<Weight>::Net(TILING_TYPE tiling_type, uint32_t Nneurons_, std::string inputFile_prefix) : Nneurons(Nneurons_) {
    std::vector<Weight> neuralNetBias = {-0.3,-0.35,-0.4,-0.45};
    std::vector<uint32_t> NneuronsVector = {1024, 4096, 16384, 65536};    
    uint32_t idxN = std::distance(NneuronsVector.begin(), std::find(NneuronsVector.begin(), NneuronsVector.end(), Nneurons));
    if(idxN >= NneuronsVector.size()) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Invalid number of neurons/layer %d", Nneurons);
        std::exit(Env::finalize());
    }    
    Weight biasValue = neuralNetBias[idxN];
    
    std::string featureFile = inputFile_prefix + "/sparse-images-" + std::to_string(Nneurons) + ".tsv";
    tiling =  Tiling<Weight>(tiling_type, (Env::nranks * Env::nranks), Env::nranks, Env::nranks, Env::nranks, featureFile);

}


#endif 