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
    //Logging::enabled = false; 
    maxLayers = 1;
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
    //std::vector<uint64_t> nnz_t(Env::nthreads);
    uint32_t nrows = inputFeatures->nrows;
    uint32_t ncols = layers[0]->ncols;
    
    //auto& B_tile = layers[0]->tiles[0][0];
    //auto& B0_spmat = B_tile.spmat;
    //auto& s_spa = spaDenseVec[0];
    //std::tie(nnz, nrows, ncols) =  spmm_sym(A0_spmat, B0_spmat, s_spa);
    //output = std::move(std::make_unique<Tiling<Weight>>(Env::nranks, Env::nranks, 1, Env::nranks, nnz, nrows, ncols, TILING_TYPE::_1D_ROW_, compression_type)); 
    
    #pragma omp parallel reduction(+:nnz)
    {
        int tid = omp_get_thread_num();
        
        auto& A_tile = inputFeatures->tiles[Env::rank][0];
        auto& A0_spmat = A_tile.spmat;
        auto& B_tile = layers[0]->tiles[0][tid];
        auto& B0_spmat = B_tile.spmat;
        auto& s_spa = spaDenseVec[tid];
        std::tie(Env::nnz_t[tid], std::ignore, std::ignore) =  spmm_sym(A0_spmat, B0_spmat, s_spa);
        nnz += Env::nnz_t[tid];
    }
    output = std::move(std::make_unique<Tiling<Weight>>(Env::nranks, Env::nranks, 1, Env::nranks, 
                       nnz, nrows, ncols, TILING_TYPE::_1D_ROW_, compression_type)); 

    
  //          #pragma omp barrier
    //    if(!tid) {
                
    for(auto& n: Env::nnz_t)
        printf("%lu ", n);
    printf("\n");
            
            
//    nnz = std::accumulate(Env::nnz_t.begin(), Env::nnz_t.end(), 0);
    //nnz_t[0] = 0;
    //uint64_t sum = 0;
    //uint64_t temp1 = Env::nnz_t[0];
    
    std::vector<uint64_t> nnz_s(Env::nthreads);
    nnz_s[0] = 0;
    for(int32_t i = 1; i < Env::nthreads; i++) {
        //uint64_t temp = Env::nnz_t[i];
        //Env::nnz_t[i] = sum + Env::nnz_t[i-1];
        nnz_s[i] = nnz_s[i-1] + Env::nnz_t[i];
        //sum += temp + temp1;
        //temp1 = 0;
    }
    Env::nnz_t = nnz_s;
    /*
       offset_nnz[0] = 0;
    start_nnz[0] = 0;
    end_nnz[0] = length_nnz[0];
    uint64_t nnzmax = length_nnz[0];
    for(uint32_t i = 1; i < Env::nthreads; i++) {
        start_nnz[i] = end_nnz[i-1];
        end_nnz[i] = start_nnz[i] + length_nnz[i];
        offset_nnz[i] = start_nnz[i];
        nnzmax += length_nnz[i];
    }
    
*/
    
    for(auto& n: Env::nnz_t)
        printf("%lu ", n);
    printf("\n");

    
    
  //  printf(">>>>%lu %d %d\n", nnz, nrows, ncols);
//}
//#pragma omp barrier

    auto& C_tile = output->tiles[Env::rank][0];    
    auto& C_spmat = C_tile.spmat;
    auto& b_bias = biasDenseVecs[0];
    //spmm(A_spmat, B_spmat, C_spmat, s_spa, b_bias);
    
    
    

    
    /*
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint32_t share = ncols/Env::nthreads;
        uint32_t start = share * tid;
        uint32_t end = start + share;
        start_col[tid] = start;
        end_col[tid] = end;
        printf("%d %d %d\n", tid, start, end);

    }
    */
    
    
    
    //auto& C_tile = output->tiles[Env::rank][0];    
    //auto& C_spmat = C_tile.spmat;
    //auto& b_bias = biasDenseVecs[0];
    //spmm(A_spmat, B_spmat, C_spmat, s_spa, b_bias);
    //printf("%lu %d %d\n", nnz, nrows, ncols);

    maxLayers = 0;
    for (uint32_t l = 0; l < maxLayers; l++) {
        Logging::print(Logging::LOG_LEVEL::INFO, "Layer %d SpMM.\n", l); 
        if(not(l%2)) {
            auto& A_tile = inputFeatures->tiles[Env::rank][0];
            auto& A_spmat = A_tile.spmat;
            auto& B_tile = layers[l]->tiles[0][0];
            auto& B_spmat = B_tile.spmat;
            auto& s_spa = spaDenseVec[0];
            std::tie(nnz, nrows, ncols) =  spmm_sym(A_spmat, B_spmat, s_spa);
            auto& C_tile = output->tiles[Env::rank][0];    
            auto& C_spmat = C_tile.spmat;
            C_spmat->reallocate(nnz, nrows, ncols);
            auto& b_bias = biasDenseVecs[l];                   
            spmm(A_spmat, B_spmat, C_spmat, s_spa, b_bias);
        }
        else {
            auto& A_tile = output->tiles[Env::rank][0];
            auto& A_spmat = A_tile.spmat;
            auto& B_tile = layers[l]->tiles[0][0];
            auto& B_spmat = B_tile.spmat;
            auto& s_spa = spaDenseVec[0];           
            std::tie(nnz, nrows, ncols) =  spmm_sym(A_spmat, B_spmat, s_spa);
            auto& C_tile = inputFeatures->tiles[Env::rank][0];    
            auto& C_spmat = C_tile.spmat;
            auto& b_bias = biasDenseVecs[l];   
            C_spmat->reallocate(nnz, nrows, ncols);
            spmm(A_spmat, B_spmat, C_spmat, s_spa, b_bias);   
        }
    }
    
    
    
    
    
    
    //output = std::move(std::make_unique<Tiling<Weight>>(Env::nranks, Env::nranks, 1, Env::nranks, 1, nrows, ncols, tiling_type, compression_type)); 
    //if(!Env::rank) {
    
    //if(tiling_type == TILING_TYPE::_1D_ROW_) {
       // for (uint32_t l = 0; l < maxLayers; l++) {
         //   Logging::print(Logging::LOG_LEVEL::INFO, "Layer %d SpMM.\n", l); 
                            //const std::shared_ptr<struct CSC<Weight>> A_CSC = std::static_pointer_cast<struct CSC<Weight>>(A);
                            /*
                            auto& A_tile = inputFeatures->tiles[Env::rank][0];
                            auto& A_spmat = A_tile.spmat;
                            auto& B_tile = layers[l]->tiles[0][0];
                            auto& B_spmat = B_tile.spmat;
                            auto& s_spa = spaDenseVec[0];
                            std::tie(nnz, nrows, ncols) =  spmm_sym(A_spmat, B_spmat, s_spa);
                            auto& C_tile = output->tiles[Env::rank][0];    
                            auto& C_spmat = C_tile.spmat;
                            C_spmat->reallocate(nnz, nrows, ncols);
                            auto& b_bias = biasDenseVecs[l];     
                            spmm(A_spmat, B_spmat, C_spmat, s_spa, b_bias);
                            */
                            //A_spmat->repopulate(C_spmat);
                            /*
                            
                            
                                          
                            
                            */
                            //printf("ifif.Rank = %d, Layer = %d [nrows= %d ncols = %d nnz = %lu] C[nrows= %d ncols = %d nnz = %lu] A[nrows= %d ncols = %d nnz = %lu]\n", Env::rank, l, nrows, ncols, nnz, C_spmat->nrows, C_spmat->ncols, C_spmat->nnz, A_spmat->nrows, A_spmat->ncols, A_spmat->nnz);
            
            
            
            
            //for (uint32_t i = 0; i < inputFeatures->nrowgrps; i++) {
              //  for (uint32_t j = 0; j < inputFeatures->ncolgrps; j++) {
                    //if(Env::rank == A_tile.rank) {
                        /*
                        if(not(l%2)) {
                            auto& A_tile = inputFeatures->tiles[Env::rank][0];
                            auto& A_spmat = A_tile.spmat;
                            auto& B_tile = layers[l]->tiles[0][0];
                            auto& B_spmat = B_tile.spmat;
                            auto& s_spa = spaDenseVec[0];
                            std::tie(nnz, nrows, ncols) =  spmm_sym(A_spmat, B_spmat, s_spa);
                            
                            if(l == 0) {
                                output = std::move(std::make_unique<Tiling<Weight>>(Env::nranks, Env::nranks, 1, Env::nranks, nnz, nrows, ncols, TILING_TYPE::_1D_ROW_, compression_type)); 
                                auto& C_tile = output->tiles[Env::rank][0];    
                                auto& C_spmat = C_tile.spmat;
                                auto& b_bias = biasDenseVecs[l];                   
                                spmm(A_spmat, B_spmat, C_spmat, s_spa, b_bias);                                
                             //   printf("ifif.Rank = %d, Layer = %d nrows= %d ncols = %d nnz = %lu [%lu %lu %lu]\n", Env::rank, l, C_spmat->nrows, C_spmat->ncols, C_spmat->nnz, nnz, C_spmat->A_blk->nitems, C_spmat->A_blk->nbytes);
                            }
                            else {
                                
                                auto& C_tile = output->tiles[Env::rank][0];    
                                auto& C_spmat = C_tile.spmat;
                                C_spmat->reallocate(nnz, nrows, ncols);
                                auto& b_bias = biasDenseVecs[l];                   
                                spmm(A_spmat, B_spmat, C_spmat, s_spa, b_bias);
                          ///      printf("ifel.Rank = %d, Layer = %d nrows= %d ncols = %d nnz = %lu [%lu %lu %lu]\n", Env::rank, l, C_spmat->nrows, C_spmat->ncols, C_spmat->nnz, nnz, C_spmat->A_blk->nitems, C_spmat->A_blk->nbytes);
                            }
                        }
                        else {
                            auto& A_tile = output->tiles[Env::rank][0];
                            auto& A_spmat = A_tile.spmat;
                            auto& B_tile = layers[l]->tiles[0][0];
                            auto& B_spmat = B_tile.spmat;
                            auto& s_spa = spaDenseVec[0];
                            //for(auto& s: s_spa)
                            //    s = 0;
                            std::tie(nnz, nrows, ncols) =  spmm_sym(A_spmat, B_spmat, s_spa);
                            //printf("New sizes: l=%d nrows=%d ncols=%d nnz=%lu %f\n", l, nrows, ncols, nnz, biasValue);
                            
                            //output = std::move(std::make_unique<Tiling<Weight>>(Env::nranks, Env::nranks, 1, Env::nranks, nnz, nrows, ncols, tiling_type, compression_type)); 
                            auto& C_tile = inputFeatures->tiles[Env::rank][0];    
                            auto& C_spmat = C_tile.spmat;
                            auto& b_bias = biasDenseVecs[l];   
                            C_spmat->reallocate(nnz, nrows, ncols);
                            spmm(A_spmat, B_spmat, C_spmat, s_spa, b_bias);                            
                            //printf("Second iter %d %d %d %lu | %d %d %lu\n", l, nrows, ncols, nnz, C_spmat->nrows, C_spmat->ncols, C_spmat->nnz);
                        //    printf("elseRank = %d, Layer = %d nrows= %d ncols = %d nnz = %lu [%lu %lu %lu]\n", Env::rank, l, C_spmat->nrows, C_spmat->ncols, C_spmat->nnz, nnz, C_spmat->A_blk->nitems, C_spmat->A_blk->nbytes);
                            
                            //exit(0);
                        }
                        */
                        
                    //}
                //}
            //}
      //  }
    //}
}


#endif 