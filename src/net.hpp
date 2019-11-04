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
            //const TILING_TYPE tiling_type_ = TILING_TYPE::_1D_ROW_,
            const COMPRESSED_FORMAT compression_type = COMPRESSED_FORMAT::_CSC_);
        
        
        
        std::unique_ptr<struct Tiling<Weight>> inputFeatures = nullptr;
        std::vector<uint32_t> trueCategories;
        std::vector<std::unique_ptr<struct Tiling<Weight>>> layers;
        std::vector<std::vector<Weight>> biasDenseVecs;
        //std::vector<std::vector<Weight>> spaDenseVec;
        std::vector<std::vector<Weight>> spaDenseVec;
        
        std::unique_ptr<struct Tiling<Weight>> output = nullptr;

        uint32_t NinputInstanses;        
        uint32_t Nneurons;
        Weight biasValue;
        uint32_t maxLayers;
        
        void inferenceReLU(COMPRESSED_FORMAT compression_type);
        void printCounters(double time);
        void stats(std::vector<double>& vec, double& sum, double& mean, double& std_dev, double& min, double& max);
};

template<typename Weight>
Net<Weight>::Net(const uint32_t NinputInstanses_, const uint32_t Nneurons_, const std::string inputFile_prefix, 
                 const uint32_t maxLayers_, const std::string layerFile_prefix,
                 const INPUT_TYPE input_type, //const TILING_TYPE tiling_type, 
                 const COMPRESSED_FORMAT compression_type) : NinputInstanses(NinputInstanses_), Nneurons(Nneurons_), maxLayers(maxLayers_) {
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Processing input feature file for %d neurons.\n", Nneurons);  
    std::vector<Weight> neuralNetBias = {-0.3,-0.35,-0.4,-0.45};
    std::vector<uint32_t> NneuronsVector = {1024, 4096, 16384, 65536};    
    uint32_t idxN = std::distance(NneuronsVector.begin(), std::find(NneuronsVector.begin(), NneuronsVector.end(), Nneurons));
    if(idxN >= NneuronsVector.size()) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Invalid number of neurons %d", Nneurons);
        std::exit(Env::finalize());
    }    
    biasValue = neuralNetBias[idxN];
    
    
    std::string feature_file = inputFile_prefix + "/sparse-images-" + std::to_string(Nneurons);
    feature_file += (input_type == INPUT_TYPE::_TEXT_) ? ".tsv" : ".bin";
    
    uint64_t nnz = 0;
    uint32_t nrows = 0;
    uint32_t ncols = 0;
    
    std::tie(nnz, nrows, ncols) = (INPUT_TYPE::_TEXT_ == input_type) ? IO::text_file_stat<Weight>(feature_file)
                                                                     : IO::binary_file_stat<Weight>(feature_file);
                                                                     
    nrows = ((NinputInstanses + 1) > nrows) ? (NinputInstanses + 1) : nrows; 
    ncols = ((Nneurons+1) > ncols) ? (Nneurons+1) : ncols;                                                                      
    
    inputFeatures = std::move(std::make_unique<Tiling<Weight>>(Env::nranks, Env::nranks, 1, Env::nranks, nnz, nrows, ncols, feature_file, input_type, TILING_TYPE::_1D_ROW_, compression_type));
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Processing the category files for %d neurons and %d layers.\n", Nneurons, maxLayers); 
    std::vector<uint32_t> maxLayersVector = {120, 480, 1920};
    uint32_t idxL = std::distance(maxLayersVector.begin(), std::find(maxLayersVector.begin(), maxLayersVector.end(), maxLayers));
    if(idxL >= maxLayersVector.size()) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Invalid number of layers %d", maxLayers);
        std::exit(Env::finalize());
    }

    std::string categoryFile = layerFile_prefix + "/neuron" + std::to_string(Nneurons) + "-l" + std::to_string(maxLayers);
    categoryFile += (input_type == INPUT_TYPE::_TEXT_) ? "-categories.tsv" : "-categories.bin";
    
    if(INPUT_TYPE::_TEXT_ == input_type) {
        IO::text_file_categories(categoryFile, trueCategories, inputFeatures->tile_height);
    }
    else {
        IO::binary_file_categories(categoryFile, trueCategories, inputFeatures->tile_height);
    }

    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Processing %d layer files (silent).\n", maxLayers); 
    Logging::enabled = false; 
    //maxLayers = 3;
    layers.resize(maxLayers);
    biasDenseVecs.resize(maxLayers);
    for(uint32_t i = 0; i < maxLayers; i++) {
        std::string layerFile = layerFile_prefix + "/neuron" + std::to_string(Nneurons) + "/n" + std::to_string(Nneurons) + "-l" + std::to_string(i+1);
        layerFile += (input_type == INPUT_TYPE::_TEXT_) ? ".tsv" : ".bin";

        std::tie(nnz, nrows, ncols) = (INPUT_TYPE::_TEXT_ == input_type) ? IO::text_file_stat<Weight>(layerFile)
                                                                     : IO::binary_file_stat<Weight>(layerFile);                                                                     
        nrows = (inputFeatures->ncols > nrows) ? inputFeatures->ncols : nrows; 
        ncols = (inputFeatures->ncols > ncols) ? inputFeatures->ncols : ncols;            

        layers[i] = std::move(std::make_unique<Tiling<Weight>>(Env::nthreads, 1, Env::nthreads, 1, Env::nthreads, Env::nthreads, nnz, nrows, ncols, layerFile, input_type, TILING_TYPE::_1D_COL_, compression_type)); 
        //layers[i] = std::move(std::make_unique<Tiling<Weight>>(1, 1, 1, 1, nnz, nrows, ncols, layerFile, input_type, TILING_TYPE::_1D_COL_, compression_type)); 

        biasDenseVecs[i] = std::vector<Weight>(inputFeatures->ncols, biasValue);
    }
    /*
    spaDenseVec.resize(1);
    spaDenseVec[0].resize(inputFeatures->tile_height);
    */
    
    spaDenseVec.resize(Env::nthreads);
    for(int32_t i = 0; i < Env::nthreads; i++)
        spaDenseVec[i].resize(inputFeatures->tile_height);
    
    
    Logging::enabled = true;
    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Running the inferenceReLU method.\n"); 
    Env::barrier();
    auto start = std::chrono::high_resolution_clock::now();
    inferenceReLU(compression_type);
    auto finish = std::chrono::high_resolution_clock::now();
    double challengeRunTime = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish-start).count())/1e9;
    //Logging::print(Logging::LOG_LEVEL::INFO, "IO time %f\n", Env::io_time);
    //Logging::print(Logging::LOG_LEVEL::INFO, "Run time (sec): %f\n", challengeRunTime);
    
    auto& C_tile = inputFeatures->tiles[Env::rank][0];    
    auto& C_spmat = C_tile.spmat;
    bool passed = validate_prediction(C_spmat, trueCategories);
    //bool passed = false;
    if(passed) {
        Logging::print(Logging::LOG_LEVEL::INFO, "Challenge PASSED.\n");
    }
    else {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Challenge FAILED.\n");
    }
    Env::barrier();
    printCounters(challengeRunTime);
}

template<typename Weight>
void Net<Weight>::printCounters(double time) {
    std::vector<double> times(Env::nranks);
    MPI_Allgather(&time, 1, MPI_DOUBLE, times.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD); 
    double sum = 0.0, mean = 0.0, std_dev = 0.0, min = 0.0, max = 0.0;
    stats(times, sum, mean, std_dev, min, max);
    Logging::print(Logging::LOG_LEVEL::INFO, "Run time (sec): min | avg +/- std_dev | max: %f | %f +/- %f | %f\n", min, mean, std_dev, max);
    Logging::print(Logging::LOG_LEVEL::INFO, "I/O time %f\n", Env::io_time);
}

template<typename Weight>
void Net<Weight>::stats(std::vector<double>& vec, double& sum, double& mean, double& std_dev, double& min, double& max) {
    sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    mean = sum / vec.size();
    double sq_sum = std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0);
    std_dev = std::sqrt(sq_sum / vec.size() - mean * mean);
    std::pair bounds = std::minmax_element(vec.begin(), vec.end());
    min = *bounds.first;
    max = *bounds.second;
}

template<typename Weight>
void Net<Weight>::inferenceReLU(COMPRESSED_FORMAT compression_type) {
    
    uint64_t nnz = 0;
    //std::vector<uint64_t> offset_nnz(Env::nthreads);
    uint32_t nrows = inputFeatures->tile_height;
    uint32_t ncols = layers[0]->ncols;
    //printf("%d %d %d\n", nrows, ncols, Env::nthreads);
    
    //auto& B_tile = layers[0]->tiles[0][0];
    //auto& B0_spmat = B_tile.spmat;
    //auto& s_spa = spaDenseVec[0];
    //std::tie(nnz, nrows, ncols) =  spmm_sym(A0_spmat, B0_spmat, s_spa);
    //output = std::move(std::make_unique<Tiling<Weight>>(Env::nranks, Env::nranks, 1, Env::nranks, nnz, nrows, ncols, TILING_TYPE::_1D_ROW_, compression_type)); 
    //printf(">>> Rank=%d csc=%d\n", Env::rank, layers[0]->tiles[0][0].spmat == NULL);
    /*
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        Env::start_col[tid] = (ncols/Env::nthreads) * tid;
        Env::end_col[tid]   = (ncols/Env::nthreads) * (tid+1);
        
        auto& A_tile = inputFeatures->tiles[Env::rank][0];
        auto& A0_spmat = A_tile.spmat;
        auto& B_tile = layers[0]->tiles[0][tid];
        auto& B0_spmat = B_tile.spmat;
        auto& s_spa = spaDenseVec[tid];
        //printf("Rank=%dtid=%d %lu %lu\n", Env::rank, tid,layers[0]->tiles.size(), layers[0]->tiles[0].size() );
        //if(!Env::rank)
        //printf("Rank=%d tid=%d csc=%d\n", Env::rank, tid, B0_spmat == NULL);
        
        std::tie(Env::offset_nnz[tid], std::ignore, std::ignore) =  spmm_sym(A0_spmat, B0_spmat, s_spa, tid);
        
        //nnz += Env::offset_nnz[tid];
        
        #pragma omp barrier
        if(!tid) {
            nnz = std::accumulate(Env::offset_nnz.begin(), Env::offset_nnz.end(), 0);
            uint64_t sum = 0;
            for(int32_t i = Env::nthreads - 1; i > 0; i--) {
                sum += Env::offset_nnz[i];
                Env::offset_nnz[i] = nnz - sum;
                Env::index_nnz[i] = Env::offset_nnz[i];
            }
            Env::offset_nnz[0] = 0;                               
            Env::index_nnz[0] = 0;
            
            output = std::move(std::make_unique<Tiling<Weight>>(Env::nranks, Env::nranks, 1, Env::nranks, 
                nnz, nrows, ncols, TILING_TYPE::_1D_ROW_, compression_type)); 
        }
        #pragma omp barrier
        auto& C0_tile = output->tiles[Env::rank][0];    
        auto& C0_spmat = C0_tile.spmat;
        auto& b_bias = biasDenseVecs[0];
        //const uint32_t B_start_col = B_tile.start_col;
        //const uint32_t B_end_end = B_tile.end_col;
        //printf("%d %d\n", B_start_col, B_end_end);
        spmm(A0_spmat, B0_spmat, C0_spmat, s_spa, b_bias, tid);
    }
    */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        
        Env::start_col[tid] = (ncols/Env::nthreads) * tid;
        Env::end_col[tid]   = (ncols/Env::nthreads) * (tid+1);
        
        auto& A0_tile = inputFeatures->tiles[Env::rank][0];
        auto& A0_spmat = A0_tile.spmat;
        auto& B0_tile = layers[0]->tiles[0][tid];
        auto& B0_spmat = B0_tile.spmat;
        auto& s_spa = spaDenseVec[tid];
        
        std::tie(Env::offset_nnz[tid], std::ignore, std::ignore) =  spmm_sym(A0_spmat, B0_spmat, s_spa, tid);
        
        #pragma omp barrier
        if(!tid) {
            /*
            nnz = std::accumulate(Env::offset_nnz.begin(), Env::offset_nnz.end(), 0);
            uint64_t sum = 0;
            for(int32_t i = Env::nthreads - 1; i > 0; i--) {
                sum += Env::offset_nnz[i];
                Env::offset_nnz[i] = nnz - sum;
                Env::index_nnz[i] = Env::offset_nnz[i];
            }
            Env::offset_nnz[0] = 0;                               
            Env::index_nnz[0] = 0;
            */
            nnz = Env::assign_nnz();
            output = std::move(std::make_unique<Tiling<Weight>>(Env::nranks, Env::nranks, 1, Env::nranks, 
                nnz, nrows, ncols, TILING_TYPE::_1D_ROW_, compression_type)); 
        }
        #pragma omp barrier
        auto& C0_tile = output->tiles[Env::rank][0];    
        auto& C0_spmat = C0_tile.spmat;
        auto& b_bias = biasDenseVecs[0];
        spmm(A0_spmat, B0_spmat, C0_spmat, s_spa, b_bias, tid);
        
        
        struct Tile<Weight> A_tile;
        struct Tile<Weight> B_tile;
        struct Tile<Weight> C_tile;
        for (uint32_t l = 1; l < maxLayers; l++) {
            if(!tid) Logging::print(Logging::LOG_LEVEL::INFO, "Layer %d SpMM.\n", l); 
            //if(not(l%2)) {
                Env::checksum[tid] = 0;
                Env::checkcount[tid] = 0;
                
                if(not(l%2)) {
                    A_tile = inputFeatures->tiles[Env::rank][0];
                    C_tile = output->tiles[Env::rank][0];
                }
                else {
                    A_tile = output->tiles[Env::rank][0];
                    C_tile = inputFeatures->tiles[Env::rank][0];
                }
                
                //auto& A_tile = inputFeatures->tiles[Env::rank][0];
                //auto& A_spmat = A_tile.spmat;
                auto& A_spmat = A_tile.spmat;
                auto& C_spmat = C_tile.spmat;
                B_tile = layers[l]->tiles[0][tid];
                //auto& B_tile = layers[l]->tiles[0][tid];
                auto& B_spmat = B_tile.spmat;
                
                auto& s_spa = spaDenseVec[tid];
                auto& b_bias = biasDenseVecs[l];   
               // #pragma omp barrier
                std::tie(Env::offset_nnz[tid], std::ignore, std::ignore) =  spmm_sym(A_spmat, B_spmat, s_spa, tid);
                //auto& C_tile = output->tiles[Env::rank][0];    
                //auto& C_spmat = C_tile.spmat;
                
                
                
                //std::shared_ptr<struct Compressed_Format<Weight>> C_spmat = C_tile.spmat;
                //auto& C_spmat = C_tile1.spmat;
                
                
                #pragma omp barrier
                if(!tid) {
                    /*
                    nnz = std::accumulate(Env::offset_nnz.begin(), Env::offset_nnz.end(), 0);
                    uint64_t sum = 0;
                    for(int32_t i = Env::nthreads - 1; i > 0; i--) {
                        sum += Env::offset_nnz[i];
                        Env::offset_nnz[i] = nnz - sum;
                        Env::index_nnz[i] = Env::offset_nnz[i];
                    }
                    Env::offset_nnz[0] = 0;                               
                    Env::index_nnz[0] = 0;
                    */
                    nnz = Env::assign_nnz();
                    C_spmat->reallocate(nnz, nrows, ncols);
                }
                #pragma omp barrier
                spmm(A_spmat, B_spmat, C_spmat, s_spa, b_bias, tid);

               // #pragma omp barrier
            /*  
            }
            else {
                Env::checksum[tid] = 0;
                Env::checkcount[tid] = 0;
                auto& A_tile = output->tiles[Env::rank][0];
                auto& A_spmat = A_tile.spmat;
                auto& B_tile = layers[l]->tiles[0][tid];
                auto& B_spmat = B_tile.spmat;
                auto& s_spa = spaDenseVec[tid];
                                
                std::tie(Env::offset_nnz[tid], std::ignore, std::ignore) =  spmm_sym(A_spmat, B_spmat, s_spa, tid);
                auto& C_tile = inputFeatures->tiles[Env::rank][0];    
                auto& C_spmat = C_tile.spmat;
                //auto* C_tile = inputFeatures->tiles[Env::rank][0];    
                //auto* C_spmat = C_tile.spmat;
                auto& b_bias = biasDenseVecs[l];                   
                
                #pragma omp barrier
                if(!tid) {
                    nnz = std::accumulate(Env::offset_nnz.begin(), Env::offset_nnz.end(), 0);
                    C_spmat->reallocate(nnz, nrows, ncols);
                    uint64_t sum = 0;
                    for(int32_t i = Env::nthreads - 1; i > 0; i--) {
                        sum += Env::offset_nnz[i];
                        Env::offset_nnz[i] = nnz - sum;
                        Env::index_nnz[i] = Env::offset_nnz[i];
                    }
                    Env::offset_nnz[0] = 0;                               
                    Env::index_nnz[0] = 0;
                }
                #pragma omp barrier
                
                spmm(A_spmat, B_spmat, C_spmat, s_spa, b_bias, tid);
            }
            */
        }
    }
}


#endif 