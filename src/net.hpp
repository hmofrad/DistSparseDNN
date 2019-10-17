/*
 * net.hpp: Neural network base class 
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */
 
#ifndef NET_HPP
#define NET_HPP

#include "triple.hpp"
#include "tiling.hpp"
//#include "io.hpp"

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
    //Logging::print(Logging::LOG_LEVEL::INFO, "Start reading the feature file %s\n", featureFile.c_str());
    
    //std::tuple<uint64_t,uint64_t,uint64_t> io_tuple = IO_interface::get_text_info<Weight>(featureFile);
    //nrows = std::get<0>(io_tuple); 
    //ncols = std::get<1>(io_tuple);
    //nnz   = std::get<2>(io_tuple);
    
    //Logging::print(Logging::LOG_LEVEL::INFO, "Done  reading the feature file %s\n", featureFile.c_str());
    //Logging::print(Logging::LOG_LEVEL::INFO, "Feature file is [%d x %d] with nnz=%lu\n", nrows, ncols, nnz);
    
    tiling =  Tiling<Weight>(tiling_type, (Env::nranks * Env::nranks), Env::nranks, Env::nranks, Env::nranks, featureFile);
    
    
    //std::vector<struct Triple<WGT>> featuresTriples;
    //static_cast<void>(IO_interface::read_text<WGT>(featureFile, tiling.tiles));
    //printf("%lu\n", triples.size());
    //Logging::print(Logging::LOG_LEVEL::INFO, "Done  reading the feature file %s\n", featureFile.c_str());
    
    
    
    
}


#endif 