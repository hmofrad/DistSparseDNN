/*
 * net.hpp: Neural network base class 
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */
 
#ifndef NET_HPP
#define NET_HPP

#include "triple.hpp"
#include "tiling.hpp"
#include "spops.hpp"

/* Input x layers */
enum PARALLELISM_TYPE {_DATA_X_DATA_, _DATA_X_MODEL_, _HYBRID_X_HYBRID_};
const char* PARALLELISM_TYPES[] = {"_DATA_X_DATA_", "_DATA_X_MODEL_", "_HYBRID_X_HYBRID_"};

template<typename Weight>
class Net {
    public:
        Net() {};
        ~Net() {};
        
        Net(const uint32_t NinputInstanses_, const uint32_t Nneurons_, 
            const std::string inputFile_prefix, const uint32_t maxLayers_, const std::string layerFile_prefix,
            const PARALLELISM_TYPE parallelism_type_  = PARALLELISM_TYPE::_DATA_X_DATA_,
            const INPUT_TYPE input_type = INPUT_TYPE::_BINARY_);

        std::unique_ptr<struct Tiling<Weight>> inputFeatures = nullptr;
        std::vector<uint32_t> trueCategories;
        std::vector<std::unique_ptr<struct Tiling<Weight>>> layers;
        
        std::vector<std::shared_ptr<struct Data_Block<Weight>>> biasWeightVecs;
        std::vector<std::shared_ptr<struct Data_Block<bool>>> spaBoolVec;
        std::vector<std::shared_ptr<struct Data_Block<Weight>>> spaWeightVec;
        
        std::unique_ptr<struct Tiling<Weight>> output = nullptr;

        uint32_t NinputInstanses;        
        uint32_t Nneurons;
        Weight biasValue;
        uint32_t maxLayers;
        
        PARALLELISM_TYPE parallelism_type;
        bool repartition = false;
        
        void printTimes();
        void printTimesExcel();
        void execute();
        void inferenceReLU(const int32_t tid);
        
        void data_x_model(const int32_t tid);
        void data_x_data(const int32_t tid);
        void hybrid_x_hybrid(const int32_t tid);
};

template<typename Weight>
Net<Weight>::Net(const uint32_t NinputInstanses_, const uint32_t Nneurons_, 
                 const std::string inputFile_prefix, const uint32_t maxLayers_, const std::string layerFile_prefix,
                 const PARALLELISM_TYPE parallelism_type_, const INPUT_TYPE input_type) 
                     : NinputInstanses(NinputInstanses_), Nneurons(Nneurons_), 
                       maxLayers(maxLayers_), parallelism_type(parallelism_type_) {
    
    auto start = std::chrono::high_resolution_clock::now();
    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Processing input feature file for %d neurons and %s\n", Nneurons, PARALLELISM_TYPES[parallelism_type]);  
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

    nrows = ((NinputInstanses + 2) > nrows) ? (NinputInstanses + 2) : nrows; 
    ncols = ((Nneurons + 2) > ncols) ? (Nneurons + 2) : ncols;
    ncols += (ncols % Env::nthreads) ? (Env::nthreads - (ncols % Env::nthreads)) : 0;  
    
    if(parallelism_type == PARALLELISM_TYPE::_DATA_X_MODEL_) {
        inputFeatures = std::move(std::make_unique<Tiling<Weight>>(Env::nranks, Env::nranks, 1, Env::nranks, 
                                                                   nnz, nrows, ncols, 
                                                                   feature_file, input_type, 
                                                                   TILING_TYPE::_1D_ROW_, repartition));
    }
    else if (parallelism_type == PARALLELISM_TYPE::_DATA_X_DATA_) {                                                               
        inputFeatures = std::move(std::make_unique<Tiling<Weight>>(Env::nranks * Env::nthreads, Env::nranks * Env::nthreads, 1, Env::nranks,
                                                                   Env::nthreads, Env::nranks * Env::nthreads, 
                                                                   nnz, nrows, ncols, 
                                                                   feature_file, input_type, 
                                                                   TILING_TYPE::_1D_ROW_, repartition));
        inputFeatures->set_threads_indices();                                                           
    }
    
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
        IO::text_file_categories(categoryFile, trueCategories, inputFeatures->nrows);
    }
    else {
        IO::binary_file_categories(categoryFile, trueCategories, inputFeatures->nrows);
    }

    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Processing %d layer files (silent).\n", maxLayers); 
    //maxLayers = 5;
    layers.resize(maxLayers);
    biasWeightVecs.resize(maxLayers);
    for(uint32_t i = 0; i < maxLayers; i++) {
        std::string layerFile = layerFile_prefix + "/neuron" + std::to_string(Nneurons) + "/n" + std::to_string(Nneurons) + "-l" + std::to_string(i+1);
        layerFile += (input_type == INPUT_TYPE::_TEXT_) ? ".tsv" : ".bin";

        std::tie(nnz, nrows, ncols) = (INPUT_TYPE::_TEXT_ == input_type) ? IO::text_file_stat<Weight>(layerFile)
                                                                     : IO::binary_file_stat<Weight>(layerFile);                                                                     
        nrows = (inputFeatures->ncols > nrows) ? inputFeatures->ncols : nrows; 
        ncols = (inputFeatures->ncols > ncols) ? inputFeatures->ncols : ncols; 
        
        if(parallelism_type == PARALLELISM_TYPE::_DATA_X_MODEL_) {
            layers[i] = std::move(std::make_unique<Tiling<Weight>>(Env::nthreads, 1, Env::nthreads, 1, 
                                                                   Env::nthreads, Env::nthreads, 
                                                                   nnz, nrows, ncols, 
                                                                   layerFile, input_type, 
                                                                   TILING_TYPE::_1D_COL_, repartition));
        }
        else if(parallelism_type == PARALLELISM_TYPE::_DATA_X_DATA_) {     
            layers[i] = std::move(std::make_unique<Tiling<Weight>>(1, 1, 1, 1, 
                                                                   nnz, nrows, ncols, 
                                                                   layerFile, input_type, 
                                                                   TILING_TYPE::_1D_COL_, false));
        }     
        biasWeightVecs[i] = std::move(std::make_shared<struct Data_Block<Weight>>(inputFeatures->ncols, Env::rank_socket_id));
        Logging::enabled = false; 
    }
    Logging::enabled = true;
    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Done reading %d layer files.\n", maxLayers); 

    for(uint32_t i = 0; i < maxLayers; i++) {
        Weight* b_A = biasWeightVecs[i]->ptr;
        for(uint32_t i = 0; i < inputFeatures->ncols; i++) {
            b_A[i] = biasValue;
        }
    }

    spaWeightVec.resize(Env::nthreads);
    for(int32_t i = 0; i < Env::nthreads; i++) {
        if(parallelism_type == PARALLELISM_TYPE::_DATA_X_MODEL_) {
            spaWeightVec[i] = std::move(std::make_shared<struct Data_Block<Weight>>(inputFeatures->get_tile_info("height", 0), Env::threads_socket_id[i]));
            
        }
        else if(parallelism_type == PARALLELISM_TYPE::_DATA_X_DATA_) { 
            spaWeightVec[i] = std::move(std::make_shared<struct Data_Block<Weight>>(inputFeatures->get_tile_info("height", i), Env::threads_socket_id[i]));
        }
    }

    if(parallelism_type == PARALLELISM_TYPE::_DATA_X_MODEL_) {
        output = std::move(std::make_unique<Tiling<Weight>>(Env::nranks, Env::nranks, 1, Env::nranks, 
                                                            0, inputFeatures->nrows, inputFeatures->ncols, 
                                                            TILING_TYPE::_1D_ROW_, repartition));
    }
    else if (parallelism_type == PARALLELISM_TYPE::_DATA_X_DATA_) { 
        output = std::move(std::make_unique<Tiling<Weight>>(Env::nranks * Env::nthreads, Env::nranks * Env::nthreads, 1, Env::nranks, 
                                                            Env::nthreads, Env::nranks * Env::nthreads, 
                                                            0, inputFeatures->nrows, inputFeatures->ncols, 
                                                            TILING_TYPE::_1D_ROW_, repartition));
    }
    output->set_tile_info(inputFeatures->tiles);
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Running the inferenceReLU method.\n"); 
    auto finish = std::chrono::high_resolution_clock::now();
    Env::io_time = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish-start).count())/1e9;
    Env::barrier();
    
    execute();

    finish = std::chrono::high_resolution_clock::now();
    Env::end_to_end_time = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish-start).count())/1e9;
    Env::barrier();
    printTimesExcel();
    //printTimes();
}

template<typename Weight>
void Net<Weight>::printTimes() {
    Env::barrier();
    double sum = 0.0, mean = 0.0, std_dev = 0.0, min = 0.0, max = 0.0;
    const char* TIME_MSGS[] = {"I/O          ", "SpMM Symbolic", "SpMM Real    ", "Mem realloc  ", "Execution    ", "Total Run    "};
    const double TIME_VALUES[] = {Env::io_time, Env::spmm_sym_time, Env::spmm_time, Env::memory_time, Env::exec_time, Env::end_to_end_time};
    const int32_t n = 6;
    
    for(int32_t i = 0; i < n; i++) {
        std::tie(sum, mean, std_dev, min, max) =  Env::statistics<double>(TIME_VALUES[i]);
        Logging::print(Logging::LOG_LEVEL::INFO, "%s time (sec): avg +/- std_dev: %.3f +/- %.3f | min: %.3f | max: %.3f\n", TIME_MSGS[i], mean, std_dev, min, max);
    }
    std::vector<double> sum_time; 
    std::vector<double> mean_time;
    std::vector<double> std_dev_time;
    std::vector<double> min_time;
    std::vector<double> max_time;
    Logging::print(Logging::LOG_LEVEL::VOID, "time | nnz | nnz_i\n");
    Logging::print(Logging::LOG_LEVEL::VOID, "l mean std_dev min max mean std_dev min max mean std_dev min max\n");
    for (uint32_t l = 0; l < maxLayers; l++) {
        std::tie(sum, mean, std_dev, min, max) =  Env::statistics<double>(Env::time_ranks[l]);
        Logging::print(Logging::LOG_LEVEL::VOID, "%3d %.3f %.3f %.3f %.3f ", l, mean, std_dev, min, max);
        uint64_t sum1 = 0.0, mean1 = 0.0, std_dev1 = 0.0, min1 = 0.0, max1 = 0.0;
        std::tie(sum1, mean1, std_dev1, min1, max1) =  Env::statistics<uint64_t>(Env::nnz_ranks[l]);
        Logging::print(Logging::LOG_LEVEL::VOID, "%12lu %12lu %12lu %12lu ", mean1, std_dev1, min1, max1);
        std::tie(sum1, mean1, std_dev1, min1, max1) =  Env::statistics<uint64_t>(Env::nnz_i_ranks[l]);
        Logging::print(Logging::LOG_LEVEL::VOID, "%12lu %12lu %12lu %12lu ", mean1, std_dev1, min1, max1);
        std::tie(sum1, mean1, std_dev1, min1, max1) =  Env::statistics<uint64_t>(Env::nnz_mean_thread_ranks[l]);
        Logging::print(Logging::LOG_LEVEL::VOID, "%12lu %12lu %12lu %12lu ", mean1, std_dev1, min1, max1);
        std::tie(sum1, mean1, std_dev1, min1, max1) =  Env::statistics<uint64_t>(Env::nnz_std_dev_thread_ranks[l]);
        Logging::print(Logging::LOG_LEVEL::VOID, "%12lu %12lu %12lu %12lu ", mean1, std_dev1, min1, max1);
        std::tie(sum1, mean1, std_dev1, min1, max1) =  Env::statistics<uint64_t>(Env::nnz_i_mean_thread_ranks[l]);
        Logging::print(Logging::LOG_LEVEL::VOID, "%12lu %12lu %12lu %12lu ", mean1, std_dev1, min1, max1);        
        std::tie(sum1, mean1, std_dev1, min1, max1) =  Env::statistics<uint64_t>(Env::nnz_i_std_dev_thread_ranks[l]);
        Logging::print(Logging::LOG_LEVEL::VOID, "%12lu %12lu %12lu %12lu\n", mean1, std_dev1, min1, max1);        
        
        Env::barrier();
    }
}

template<typename Weight>
void Net<Weight>::printTimesExcel() {
    Env::barrier();
    Logging::print(Logging::LOG_LEVEL::VOID, "exec: mean, std_dev, min, max, spmm_sym_mean, spmm_mean, mem_mean\n");
    double sum = 0.0, mean = 0.0, std_dev = 0.0, min = 0.0, max = 0.0;
    
    //if(parallelism_type == PARALLELISM_TYPE::_DATA_X_DATA_) {
        int index = std::distance(Env::execution_time.begin(), std::max_element(Env::execution_time.begin(), Env::execution_time.end()));
        Env::exec_time = Env::execution_time[index];
        Env::spmm_sym_time = Env::spmm_symb_time[index];
        Env::spmm_time = Env::spmm_real_time[index];
        Env::memory_time = Env::memory_allocation_time[index];
    //}
    //printf("%d %f %d %f %f %f\n", Env::rank, Env::spmm_sym_time, index, Env::spmm_symb_time[0], Env::spmm_symb_time[1], Env::spmm_symb_time[2]);
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics<double>(Env::exec_time);
    Logging::print(Logging::LOG_LEVEL::VOID, "time: %.3f %.3f %.3f %.3f ", mean, std_dev, min, max);
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics<double>(Env::spmm_sym_time);
    Logging::print(Logging::LOG_LEVEL::VOID, "%.3f ", mean);
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics<double>(Env::spmm_time);
    Logging::print(Logging::LOG_LEVEL::VOID, "%.3f ", mean);
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics<double>(Env::memory_time);
    Logging::print(Logging::LOG_LEVEL::VOID, "%.3f\n", mean);
    /*
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics_t<double>(Env::execution_time);
    Logging::print(Logging::LOG_LEVEL::VOID, "time: %.3f %.3f %.3f %.3f ", mean, std_dev, min, max);
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics_t<double>(Env::spmm_symb_time);
    Logging::print(Logging::LOG_LEVEL::VOID, "%.3f ", mean);
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics_t<double>(Env::spmm_real_time);
    Logging::print(Logging::LOG_LEVEL::VOID, "%.3f ", mean);
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics_t<double>(Env::memory_allocation_time);
    Logging::print(Logging::LOG_LEVEL::VOID, "%.3f\n", mean);
    */
}

template<typename Weight>
void Net<Weight>::execute() {
    std::vector<std::thread> threads;
    
    for(int i = 0; i < Env::nthreads; i++) {
        threads.push_back(std::thread(&Net<Weight>::inferenceReLU, this, i));
    }
    
    for(std::thread& th: threads) {
        th.join();
    }
}

template<typename Weight>
void Net<Weight>::inferenceReLU(const int32_t tid) {
    
    if(parallelism_type == PARALLELISM_TYPE::_DATA_X_MODEL_) {
        data_x_model(tid);
    }
    if(parallelism_type == PARALLELISM_TYPE::_DATA_X_DATA_) {
        data_x_data(tid);
    }else {
    
    if(Env::NUMA_ALLOC)
        (void)Env::set_thread_affinity(tid);   
    double start_time = 0;
    uint64_t nnz = 0;
    uint32_t nrows = inputFeatures->tile_height;
    uint32_t ncols = layers[0]->ncols;
    
    struct Tile<Weight> A_tile;
    struct Tile<Weight> B_tile;
    struct Tile<Weight> C_tile;
    auto& s_spa = spaWeightVec[tid];
    /*
    if(parallelism_type == PARALLELISM_TYPE::_DATA_X_MODEL_) {
        Env::assign_col(ncols, tid);
        auto start = std::chrono::high_resolution_clock::now();  
            for (uint32_t l = 0; l < maxLayers; l++) {
                A_tile = inputFeatures->tiles[Env::rank][0];
                std::shared_ptr<struct CSC<Weight>> A_spmat = A_tile.spmat;
                B_tile = layers[l]->tiles[0][tid];
                std::shared_ptr<struct CSC<Weight>> B_spmat = B_tile.spmat;

                Env::start_col[tid] = B_tile.start_col;
                Env::end_col[tid] = B_tile.end_col;
                uint32_t start_col = 0;
                uint32_t end_col   = B_spmat->ncols;
                std::tie(Env::offset_nnz[tid], nrows, std::ignore) =  spmm_sym(A_spmat, B_spmat, s_spa, start_col, end_col, tid);
                
                pthread_barrier_wait(&Env::thread_barrier);
                
                start_time = Env::tic();
                if(!tid) {
                    nnz = Env::assign_nnz();
                    Env::memory_time += Env::toc(start_time);
                }
                Env::memory_allocation_time[tid] += Env::toc(start_time);
                
                C_tile = output->tiles[Env::rank][0];
                std::shared_ptr<struct CSC<Weight>> C_spmat = C_tile.spmat;
                auto& b_bias = biasWeightVecs[l];
                
                start_time = Env::tic();
                
                if(!tid) {
                    C_spmat->reallocate(nnz, nrows, ncols);
                    Env::memory_time += Env::toc(start_time);
                }
                Env::memory_allocation_time[tid] += Env::toc(start_time);
                uint32_t offset = Env::start_col[tid];
                pthread_barrier_wait(&Env::thread_barrier);
                spmm(A_spmat, B_spmat, C_spmat, s_spa, b_bias, start_col, end_col, offset, tid);
                pthread_barrier_wait(&Env::thread_barrier);
                
                adjust(C_spmat, tid);
                //repopulate(A_spmat, C_spmat, tid);
                
                if(!tid) walk_by_rank(A_spmat);
                
                if(!tid) Env::iteration++;
            }    

        auto finish = std::chrono::high_resolution_clock::now();
        if(!tid) Env::exec_time = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish-start).count())/1e9;
        Env::execution_time[tid] = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish - start).count())/1e9    ;

        C_tile = inputFeatures->tiles[Env::rank][0];
        auto& C_spmat = C_tile.spmat;
        if(!tid) {
            bool passed = validate_prediction(C_spmat, trueCategories, C_tile.start_row);
            if(passed) {
                Logging::print(Logging::LOG_LEVEL::INFO, "Challenge PASSED.\n");
            }
            else {
                Logging::print(Logging::LOG_LEVEL::ERROR, "Challenge FAILED.\n");
            }
        }
        pthread_barrier_wait(&Env::thread_barrier);
    }
    */
    /*
    else if (parallelism_type  == PARALLELISM_TYPE::_DATA_X_DATA_) {
        auto start = std::chrono::high_resolution_clock::now();  
        for (uint32_t l = 0; l < maxLayers; l++) {
            if(not(l%2)) {
                A_tile = inputFeatures->tiles[Env::tile_index[tid]][0];
                C_tile = output->tiles[Env::tile_index[tid]][0];
            }
            else {
                A_tile = output->tiles[Env::tile_index[tid]][0];
                C_tile = inputFeatures->tiles[Env::tile_index[tid]][0];
            }

            auto& A_spmat = A_tile.spmat;
            auto& C_spmat = C_tile.spmat;
            B_tile = layers[l]->tiles[0][0];
            auto& B_spmat = B_tile.spmat;
            auto& b_bias = biasWeightVecs[l];  

            const uint32_t start_col = 0;
            const uint32_t end_col   = std::static_pointer_cast<struct CSC<Weight>>(B_spmat)->ncols;
            std::tie(Env::offset_nnz[tid], nrows, ncols) =  spmm_sym(A_spmat, B_spmat, s_spa, start_col, end_col, tid);
            Env::index_nnz[tid] = 0;
            
            start_time = Env::tic();
            C_spmat->reallocate(Env::offset_nnz[tid], nrows, ncols, tid);
            
            if(!tid) Env::memory_time += Env::toc(start_time);
            Env::memory_allocation_time[tid] += Env::toc(start_time);

            spmm(A_spmat, B_spmat, C_spmat, s_spa, b_bias, refine, tid);
        
            //{	
            //    adjust(C_spmat, tid);	
            //    walk_by_tid(C_spmat, tid);	
            //}

            if(!tid) Env::iteration++;
        }
    }
    */
    //else 
        if (parallelism_type  == PARALLELISM_TYPE::_DATA_X_DATA_) {
        std::vector<int32_t> my_follower_threads;
        my_follower_threads.push_back(tid);
        //if(tid == 0) sleep(6);
        //if(tid == 4) sleep(10);
        auto start = std::chrono::high_resolution_clock::now();  
        for (uint32_t l = 0; l < maxLayers; l++) {
            if(not(l%2)) {
                A_tile = inputFeatures->tiles[Env::tile_index[tid]][0];
                C_tile = output->tiles[Env::tile_index[tid]][0];
            }
            else {
                A_tile = output->tiles[Env::tile_index[tid]][0];
                C_tile = inputFeatures->tiles[Env::tile_index[tid]][0];
            }

            //auto& 
            std::shared_ptr<struct CSC<Weight>> A_spmat = A_tile.spmat;
            //auto& 
            std::shared_ptr<struct CSC<Weight>> C_spmat = C_tile.spmat;
            B_tile = layers[l]->tiles[0][0];
            //auto& 
            std::shared_ptr<struct CSC<Weight>> B_spmat = B_tile.spmat;
            auto& b_bias = biasWeightVecs[l];  
            //if(0){
            //if(!Env::follower_threads.empty()) {
                        //pthread_mutex_lock(&Env::thread_mutexes[tid]);    
            if(!Env::follower_threads.empty() and (my_follower_threads.size() == 1)) {
                pthread_mutex_lock(&Env::thread_mutex);
                if(!Env::follower_threads.empty()) {
                    my_follower_threads.insert(my_follower_threads.end(), Env::follower_threads.begin(), Env::follower_threads.end());
                    Env::follower_threads.erase(Env::follower_threads.begin(), Env::follower_threads.end());
                    
                    //num_threads += Env::n_follower_threads;
                    //Env::n_follower_threads = 0;
                    
                    //my_follower_threads.resize(Env::follower_threads.size());
                    //std::move(Env::follower_threads.begin(), Env::follower_threads.end(), my_follower_threads.begin()); 
                    //my_follower_threads.push_back(tid);
                    
                    
                    for(uint32_t t = 0; t < my_follower_threads.size(); t++) {
                        int32_t thread = my_follower_threads[t];
                        uint32_t rowgroup = Env::tile_index[tid];
                        uint32_t layer = l;
                        uint32_t start_col = ((ncols/my_follower_threads.size()) * t);
                        uint32_t end_col   = (t == (my_follower_threads.size()-1)) ? ncols : ((ncols/my_follower_threads.size()) * (t+1));
                        Env::follower_threads_info[tid][thread] = std::move(Env::helper_thread_info(thread, rowgroup, layer, start_col, end_col));
                        Env::follower_to_leader[thread] = tid;
                    }
                    Env::num_follower_threads[tid] = my_follower_threads.size()-1;
                    //printf("%d: There are [%lu %lu] %d\n", tid, Env::follower_threads.size(), my_follower_threads.size(), Env::follower_to_leader[0]);
                }
                    //for(int32_t i = 0; i < num_threads; i++) {
                      //  auto info = Env::follower_threads_info[tid][i];
                        //printf("%d: %d %d %d %d\n", i, info.layer, info.rowgroup, info.start_col, info.end_col);
                    //}
                    

                    
                    
                

            //    const uint32_t start_col = Env::follower_threads_info[tid][tid].start_col;
              //  const uint32_t end_col = Env::follower_threads_info[tid][tid].end_col;
                //std::tie(Env::offset_nnz[tid], nrows, ncols) =  spmm_sym(A_spmat, B_spmat, s_spa, start_col, end_col, tid);
                //Env::index_nnz[tid] = 0;
                
                //printf("leader tid=%d starts with %lu followers: ", tid, my_follower_threads.size()-1); 
                //for(auto f: my_follower_threads) {printf("%d ", f);} printf("\n");
                
                
                pthread_barrier_init(&Env::thread_barriers[tid], NULL, my_follower_threads.size());
                
                
                
                pthread_cond_broadcast(&Env::thread_cond); 
                //pthread_cond_signal(&Env::thread_cond); 
                pthread_mutex_unlock(&Env::thread_mutex);
                
            }
            

            //pthread_cond_broadcast(&Env::thread_conds[tid]);  
            //pthread_mutex_unlock(&Env::thread_mutexes[tid]);
            
            //else {
            if(my_follower_threads.size() == 1) {
                //printf("-leader tid=%d, layer=%d\n", tid, l);
                const uint32_t start_col = 0;
                const uint32_t end_col   = std::static_pointer_cast<struct CSC<Weight>>(B_spmat)->ncols;
                std::tie(Env::offset_nnz[tid], nrows, ncols) =  spmm_sym(A_spmat, B_spmat, s_spa, start_col, end_col, tid);
                Env::index_nnz[tid] = 0;
                //printf("1. tid=%d nnz=%lu ht=%lu\n", tid, Env::offset_nnz[tid], my_follower_threads.size());
                
                start_time = Env::tic();
                //C_spmat->reallocate(Env::offset_nnz[tid], nrows, ncols);//, tid);
                
                
                
                if(!tid) Env::memory_time += Env::toc(start_time);
                Env::memory_allocation_time[tid] += Env::toc(start_time);
                uint32_t offset = 0;    
                spmm(A_spmat, B_spmat, C_spmat, s_spa, b_bias, start_col, end_col, offset, Env::index_nnz[tid], tid);
            }
            else { // my_follower_threads.size() > 1
                //printf("+leader %d layer=%d nfollowers=%lu\n", tid, l, my_follower_threads.size()-1);
                
                
                //pthread_mutex_lock(&Env::thread_mutexes[tid]);
                //pthread_cond_broadcast(&Env::thread_conds[tid]);  
                //pthread_mutex_unlock(&Env::thread_mutexes[tid]);
                
                pthread_barrier_wait(&Env::thread_barriers[tid]);
                //pthread_mutex_lock(&Env::thread_mutexes[tid]);
                //for(uint32_t i = 0; i < my_follower_threads.size()-1; i++)
                  //  pthread_cond_signal(&Env::thread_conds[tid]); 
                //pthread_cond_broadcast(&Env::thread_conds[tid]);             
                //pthread_mutex_unlock(&Env::thread_mutexes[tid]);
                
                //pthread_mutex_lock(&Env::thread_mutexes[tid]);
                //pthread_cond_wait(&Env::thread_conds[tid], &Env::thread_mutexes[tid]);  
                //pthread_mutex_unlock(&Env::thread_mutexes[tid]);
                
                const uint32_t start_col = Env::follower_threads_info[tid][tid].start_col;
                const uint32_t end_col   = Env::follower_threads_info[tid][tid].end_col;
                uint64_t& my_nnz = Env::follower_threads_info[tid][tid].nnz;
                std::tie(Env::offset_nnz[tid], nrows, ncols) =  spmm_sym(A_spmat, B_spmat, s_spa, start_col, end_col, tid);
                my_nnz = Env::offset_nnz[tid];
                //Env::index_nnz[tid] = 0;
                
                //printf("2. tid=%d nnz=%lu ht=%lu\n", tid, Env::offset_nnz[tid], my_follower_threads.size());
                
                start_time = Env::tic();
                //while((Env::thread_counters[tid] + 1) != my_follower_threads.size()) {
                  //  std::this_thread::sleep_for(std::chrono::nanoseconds(1));
                    //printf("%d %lu\n", Env::thread_counters[tid], (my_follower_threads.size() - 1));
                //}
                
                //pthread_mutex_lock(&Env::thread_mutexes1[tid]);
                //pthread_cond_wait(&Env::thread_conds1[tid], &Env::thread_mutexes1[tid]);  
                //pthread_mutex_unlock(&Env::thread_mutexes1[tid]);
                Env::thread_counters[tid] = 0;
                //printf("+leader tid=%d, layer=%d, before nnz\n", tid, l);
                
                pthread_barrier_wait(&Env::thread_barriers[tid]);
                //pthread_mutex_lock(&Env::thread_mutexes[tid]);
                
                nnz = 0;
                for(auto t: my_follower_threads) {
                    //auto& thread_info = Env::follower_threads_info[tid][t];
                    //printf("t=%d  nnz = %lu\n", t, thread_info.nnz);
                    nnz += Env::offset_nnz[t];
                    //thread_info.nnz = 0;
                }
                //printf("+leader tid=%d, layer=%d, total nnz = %lu\n", tid, l, nnz);
                
                    
                    //uint64_t nnz = std::accumulate(Env::offset_nnz.begin(), Env::offset_nnz.end(), 0);
    
                uint64_t sum = 0;
                for(int32_t i = my_follower_threads.size() - 1; i > 0; i--) {
                    int32_t t = my_follower_threads[i];
                    //auto& thread_info = Env::follower_threads_info[tid][t];
                    sum += Env::offset_nnz[t];
                    Env::offset_nnz[t] = nnz - sum;
                    Env::index_nnz[t] = Env::offset_nnz[t];
                }
                Env::offset_nnz[tid] = 0;                               
                Env::index_nnz[tid] = 0;
                    
                //C_spmat->reallocate(nnz, nrows, ncols);          
                    
                //pthread_cond_broadcast(&Env::thread_conds[tid]);  
                //pthread_mutex_unlock(&Env::thread_mutexes[tid]);
                
                
                
                Env::memory_time += Env::toc(start_time);
                Env::memory_allocation_time[tid] += Env::toc(start_time);
                uint32_t offset = 0;    
                pthread_barrier_wait(&Env::thread_barriers[tid]);
                spmm(A_spmat, B_spmat, C_spmat, s_spa, b_bias, start_col, end_col, offset, Env::index_nnz[tid], tid);
                pthread_barrier_wait(&Env::thread_barriers[tid]);
                
                //pthread_mutex_lock(&Env::thread_mutexes1[tid]);
                //pthread_cond_broadcast(&Env::thread_conds1[tid]);  
                //pthread_mutex_unlock(&Env::thread_mutexes1[tid]);
                Env::displacement_nnz[tid] = 0;
                for(uint32_t i = 1; i < my_follower_threads.size(); i++) {    
                    int32_t t_minus_1 = my_follower_threads[i-1];
                    int32_t t = my_follower_threads[i];
                    Env::displacement_nnz[t] = Env::offset_nnz[t] - Env::index_nnz[t_minus_1];
                }
                
                const std::shared_ptr<struct CSC<Weight>> C_CSC = std::static_pointer_cast<struct CSC<Weight>>(C_spmat);
                C_CSC->nnz_i = 0;
                for(uint32_t i = 0; i < my_follower_threads.size(); i++) {    
                    int32_t t = my_follower_threads[i];
                    C_CSC->nnz_i += (Env::index_nnz[t] - Env::offset_nnz[t]);
                }
                //printf("+leader tid=%d, layer=%d, nnzi=%lu\n", tid, l, C_CSC->nnz_i);
                
                //pthread_mutex_lock(&Env::thread_mutexes2[tid]);
                //pthread_cond_broadcast(&Env::thread_conds2[tid]);  
                //pthread_mutex_unlock(&Env::thread_mutexes2[tid]);
                pthread_barrier_wait(&Env::thread_barriers[tid]);
                repopulate(A_spmat, C_spmat, tid, tid, my_follower_threads);
                
                
                //if(!Env::follower_threads.empty()) {
                   // pthread_mutex_lock(&Env::thread_mutexes[my_leader]);
                   // pthread_cond_wait(&Env::thread_conds[my_leader], &Env::thread_mutexes[my_leader]);  
                    //pthread_mutex_unlock(&Env::thread_mutexes[my_leader]);
                //}
                
                //printf("+leader tid=%d, layer=%d done\n", tid, l);
                //pthread_barrier_wait(&Env::thread_barriers[my_leader]);
                
                //pthread_mutex_lock(&Env::thread_mutexes[tid]);
                //pthread_cond_broadcast(&Env::thread_conds[tid]);  
                //pthread_mutex_unlock(&Env::thread_mutexes[tid]);
                
               
               
               /*
                pthread_barrier_wait(&Env::thread_barrier);
                spmm(A_spmat, B_spmat, C_spmat, s_spa, b_bias, refine, tid);
                pthread_barrier_wait(&Env::thread_barrier);
                
                adjust(C_spmat, tid);
                repopulate(A_spmat, C_spmat, tid);
                */
                

                
            }
            //}


            //{	
            //    adjust(C_spmat, tid);	
            //    walk_by_tid(C_spmat, tid);	
            //}

            if(!tid) Env::iteration++;
        }

        //printf("tid=%d added to the queue %lu %lu: ", tid, Env::follower_threads.size(), my_follower_threads.size());   
        //for(auto f: my_follower_threads) {printf("%d ", f);} printf("\n");        
        //if(Env::follower_threads.size() == (uint32_t) Env::nthreads) Env::done = true;
        
        //printf("%d %lu %lu\n", tid, my_follower_threads.size(), Env::follower_threads.size());    
        while(Env::follower_threads.size() != (uint32_t) Env::nthreads) {
            //pthread_barrier_wait(&Env::thread_barriers[Env::follower_to_leader[tid]]);
            pthread_mutex_lock(&Env::thread_mutex);
                //Env::n_follower_threads += num_threads;
                //Env::n_follower_threads++;
                //Env::follower_threads.push_back(tid);
                if(!my_follower_threads.empty()) {
                    Env::follower_threads.insert(Env::follower_threads.end(), my_follower_threads.begin(), my_follower_threads.end());
                    my_follower_threads.erase(my_follower_threads.begin(), my_follower_threads.end());
                }
                Env::follower_to_leader[tid] = -1;
                if(Env::follower_threads.size() != (uint32_t) Env::nthreads) {
                    //printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> tid %d is waiting %lu\n", tid, Env::follower_threads.size());
                    pthread_cond_wait(&Env::thread_cond, &Env::thread_mutex);    
                }
                else {
                    //printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Bcast all %d %lu\n", tid, Env::follower_threads.size());
                    pthread_cond_broadcast(&Env::thread_cond);    
                }
            pthread_mutex_unlock(&Env::thread_mutex);
            
            
            const int32_t my_leader = Env::follower_to_leader[tid];
            //printf("tid %d with leader %d\n", tid, my_leader);
            if(my_leader != -1) {
                //printf("%d: layer = %d\n", tid, layer);
                uint32_t           layer = Env::follower_threads_info[my_leader][tid].layer;
                const uint32_t rowgroup  = Env::follower_threads_info[my_leader][tid].rowgroup;
                uint64_t& my_nnz = Env::follower_threads_info[my_leader][tid].nnz;
                for (uint32_t l = layer; l < maxLayers; l++) {
                    //printf(" leader %d layer=%d, follower=%d\n", my_leader, l, tid);
                    
                    pthread_barrier_wait(&Env::thread_barriers[my_leader]);
                    //pthread_mutex_lock(&Env::thread_mutexes[my_leader]);
                    //pthread_cond_wait(&Env::thread_conds[my_leader], &Env::thread_mutexes[my_leader]);  
                    //pthread_mutex_unlock(&Env::thread_mutexes[my_leader]);
                    
                    
                    
                    /*
                    pthread_mutex_lock(&Env::thread_mutexes[my_leader]);
                    Env::thread_counters[my_leader]++;
                    if(Env::thread_counters[my_leader] == Env::num_follower_threads[my_leader]) {
                        pthread_mutex_lock(&Env::thread_mutexes1[my_leader]);
                        pthread_cond_signal(&Env::thread_conds1[my_leader]); 
                        pthread_mutex_unlock(&Env::thread_mutexes1[my_leader]);
                    }
                    pthread_cond_wait(&Env::thread_conds[my_leader], &Env::thread_mutexes[my_leader]);  
                    pthread_mutex_unlock(&Env::thread_mutexes[my_leader]);
                    */
                    const uint32_t start_col = Env::follower_threads_info[my_leader][tid].start_col;
                    const uint32_t end_col   = Env::follower_threads_info[my_leader][tid].end_col;
                    
                    
                    
                    if(not(l%2)) {
                        A_tile = inputFeatures->tiles[rowgroup][0];
                        C_tile = output->tiles[rowgroup][0];
                    }
                    else {
                        A_tile = output->tiles[rowgroup][0];
                        C_tile = inputFeatures->tiles[rowgroup][0];
                    }

                    //auto& 
                    std::shared_ptr<struct CSC<Weight>> A_spmat = A_tile.spmat;
                    //auto& 
                    std::shared_ptr<struct CSC<Weight>> C_spmat = C_tile.spmat;
                    B_tile = layers[l]->tiles[0][0];
                    //auto& 
                    std::shared_ptr<struct CSC<Weight>> B_spmat = B_tile.spmat;
                    auto& b_bias = biasWeightVecs[l];  
                
                    std::tie(Env::offset_nnz[tid], nrows, ncols) =  spmm_sym(A_spmat, B_spmat, s_spa, start_col, end_col, tid);
                    my_nnz = Env::offset_nnz[tid];
                    //Env::index_nnz[tid] = 0;
                    //printf("follower=%d/%d layer=%d my nnz=%lu\n", tid, my_leader,l,  my_nnz);
                    //pthread_mutex_lock(&Env::thread_mutexes[my_leader]);
                    //Env::thread_counters[my_leader]++;
                    //printf("<%d %d>\n", tid, Env::thread_counters[my_leader]);
                    //if(Env::thread_counters[my_leader] == Env::num_follower_threads[my_leader]) {
                    ///  pthread_mutex_lock(&Env::thread_mutexes1[my_leader]);
                    //    pthread_cond_signal(&Env::thread_conds1[my_leader]); 
                    //    pthread_mutex_unlock(&Env::thread_mutexes1[my_leader]);
                    //}
                    //pthread_cond_wait(&Env::thread_conds[my_leader], &Env::thread_mutexes[my_leader]);  
                    //pthread_mutex_unlock(&Env::thread_mutexes[my_leader]);
                    
                    pthread_barrier_wait(&Env::thread_barriers[my_leader]);
                    
                    uint32_t offset = 0;    
                    pthread_barrier_wait(&Env::thread_barriers[my_leader]);
                    //printf("follower=%d/%d layer=%d going for spmm\n", tid, my_leader, l);
                    
                    spmm(A_spmat, B_spmat, C_spmat, s_spa, b_bias, start_col, end_col, offset, Env::index_nnz[tid], tid);
                    pthread_barrier_wait(&Env::thread_barriers[my_leader]);
                    
                    //pthread_mutex_lock(&Env::thread_mutexes1[my_leader]);
                    //pthread_cond_wait(&Env::thread_conds1[my_leader], &Env::thread_mutexes1[my_leader]);  
                    //pthread_mutex_unlock(&Env::thread_mutexes1[my_leader]);
                    
                    //printf("follower=%d/%d layer=%d spmm is done\n", tid, my_leader, l);
                    
                    
                    //pthread_mutex_lock(&Env::thread_mutexes2[my_leader]);
                    //pthread_cond_wait(&Env::thread_conds2[my_leader], &Env::thread_mutexes2[my_leader]);  
                    //pthread_mutex_unlock(&Env::thread_mutexes2[my_leader]);
                    
                    pthread_barrier_wait(&Env::thread_barriers[my_leader]);
                    //printf("follower=%d/%d layer=%d synch is done\n", tid, my_leader, l);
                    repopulate(A_spmat, C_spmat, tid, my_leader, my_follower_threads);
                    //printf("follower=%d/%d layer=%d repo is done\n", tid, my_leader, l);
                    //pthread_barrier_wait(&Env::thread_barriers[my_leader]);
                    
                    //printf("2. tid=%d/%d [idx=%lu nnz=%lu] ht=%lu %lu\n", my_leader, tid, Env::index_nnz[tid], Env::offset_nnz[tid], my_follower_threads.size(), C_spmat->nnz);
                }
                //my_leader = -1;
            }
        }
        
        auto finish = std::chrono::high_resolution_clock::now();
        if(!tid) Env::exec_time = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish - start).count())/1e9;
        Env::execution_time[tid] = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish - start).count())/1e9;
        
        
        
        //while(Env::follower_threads.size() != (uint32_t) Env::nthreads) {
            //std::this_thread::sleep_for(std::chrono::nanoseconds(1));
            
        //}
        
        //printf("tid=%d left %d\n", tid, Env::follower_to_leader[tid]);    
        

        
        C_tile = inputFeatures->tiles[Env::tile_index[tid]][0];
        auto& C_spmat = C_tile.spmat;
        bool passed = validate_prediction(C_spmat, trueCategories, C_tile.start_row, tid);
        //bool passed = false;
        //printf("tid=%d passed %d\n", tid, passed);   
        if(passed) {
            if(!tid) Logging::print(Logging::LOG_LEVEL::INFO, "Challenge PASSED.\n");
        }
        else {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Challenge FAILED.\n");
        }
        
        pthread_barrier_wait(&Env::thread_barrier);        
    }   
        }
}

template<typename Weight>
void Net<Weight>::data_x_model(const int32_t tid) {
    if(Env::NUMA_ALLOC)
        (void)Env::set_thread_affinity(tid);   
    
    double start_time = 0;
    uint64_t nnz   = 0;
    uint32_t nrows = inputFeatures->tile_height;
    uint32_t ncols = layers[0]->ncols;
    
    struct Tile<Weight> A_tile;
    struct Tile<Weight> B_tile;
    struct Tile<Weight> C_tile;
    
    auto& s_spa = spaWeightVec[tid];
    auto& thread_st = Env::threads[tid];
    
    auto start = std::chrono::high_resolution_clock::now();  
    for (uint32_t l = 0; l < maxLayers; l++) {
        start_time = Env::tic(); 
            A_tile = inputFeatures->tiles[Env::rank][0];
            std::shared_ptr<struct CSC<Weight>> A_CSC = A_tile.spmat;
            B_tile = layers[l]->tiles[0][tid];
            std::shared_ptr<struct CSC<Weight>> B_CSC = B_tile.spmat;
            C_tile = output->tiles[Env::rank][0];
            std::shared_ptr<struct CSC<Weight>> C_CSC = C_tile.spmat;
            auto& b_bias = biasWeightVecs[l];

            uint32_t start_col = 0;
            uint32_t end_col   = B_CSC->ncols;
            std::tie(thread_st.off_nnz, nrows, std::ignore) =  spmm_sym(A_CSC, B_CSC, s_spa, start_col, end_col, tid);
            pthread_barrier_wait(&Env::thread_barrier);
        Env::spmm_symb_time[tid] += Env::toc(start_time);   
        
        start_time = Env::tic();
            const int32_t leader_tid = 0;
            Env::adjust_nnz(nnz, leader_tid, tid);
            C_CSC->reallocate(nnz, nrows, ncols, leader_tid, tid);
        Env::memory_allocation_time[tid] += Env::toc(start_time);
        
        start_time = Env::tic();
            uint32_t off_col   = B_tile.start_col;
            pthread_barrier_wait(&Env::thread_barrier);
            spmm(A_CSC, B_CSC, C_CSC, s_spa, b_bias, start_col, end_col, off_col, thread_st.idx_nnz, tid);
            pthread_barrier_wait(&Env::thread_barrier);
        Env::spmm_real_time[tid] += Env::toc(start_time);
        
        start_time = Env::tic();
            Env::adjust_displacement(tid);
            C_CSC->adjust(leader_tid, tid);	
            start_col = B_tile.start_col;
            end_col   = B_tile.end_col;
            uint32_t dis_nnz = thread_st.dis_nnz;
            repopulate(A_CSC, C_CSC, start_col, end_col, dis_nnz, tid);
        Env::memory_allocation_time[tid] += Env::toc(start_time);
        
        //A_CSC->walk_dxm(false, leader_tid, tid);
        
        if(!tid) Env::iteration++;
    }    
    auto finish = std::chrono::high_resolution_clock::now();
    Env::execution_time[tid] = (double)(std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count())/1e9;

    A_tile = inputFeatures->tiles[Env::rank][0];
    const std::shared_ptr<struct CSC<Weight>> A_CSC = A_tile.spmat;
    if(!tid) {
        bool passed = validate_prediction(A_CSC, trueCategories, A_tile.start_row);
        if(passed) {
            Logging::print(Logging::LOG_LEVEL::INFO, "Challenge PASSED.\n");
        }
        else {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Challenge FAILED.\n");
        }
    }
    pthread_barrier_wait(&Env::thread_barrier);
}

template<typename Weight>
void Net<Weight>::data_x_data(const int32_t tid) {
    if(Env::NUMA_ALLOC)
        (void)Env::set_thread_affinity(tid);   
    
    double start_time = 0;
    uint64_t nnz = 0;
    uint32_t nrows = inputFeatures->tile_height;
    uint32_t ncols = layers[0]->ncols;
    
    struct Tile<Weight> A_tile;
    struct Tile<Weight> B_tile;
    struct Tile<Weight> C_tile;
    auto& s_spa = spaWeightVec[tid];
    auto& thread_st = Env::threads[tid];
    
    auto start = std::chrono::high_resolution_clock::now();  
    for (uint32_t l = 0; l < maxLayers; l++) {
        start_time = Env::tic(); 
            if(not(l%2)) {
                A_tile = inputFeatures->tiles[Env::tile_index[tid]][0];
                C_tile = output->tiles[Env::tile_index[tid]][0];
            }
            else {
                A_tile = output->tiles[Env::tile_index[tid]][0];
                C_tile = inputFeatures->tiles[Env::tile_index[tid]][0];
            }

            std::shared_ptr<struct CSC<Weight>> A_CSC = A_tile.spmat;
            std::shared_ptr<struct CSC<Weight>> C_CSC = C_tile.spmat;
            B_tile = layers[l]->tiles[0][0];
            std::shared_ptr<struct CSC<Weight>> B_CSC = B_tile.spmat;
            auto& b_bias = biasWeightVecs[l];  

            const uint32_t start_col = 0;
            const uint32_t end_col   = B_CSC->ncols;
            std::tie(thread_st.off_nnz, nrows, ncols) =  spmm_sym(A_CSC, B_CSC, s_spa, start_col, end_col, tid);
        Env::spmm_symb_time[tid] += Env::toc(start_time);      
        
        start_time = Env::tic();
            int32_t leader_tid = -1;
            C_CSC->reallocate(thread_st.off_nnz, nrows, ncols, leader_tid, tid);
        Env::memory_allocation_time[tid] += Env::toc(start_time);

        start_time = Env::tic();
            uint32_t off_col   = 0;
            thread_st.idx_nnz = 0;
            spmm(A_CSC, B_CSC, C_CSC, s_spa, b_bias, start_col, end_col, off_col, thread_st.idx_nnz, tid);
            Env::adjust_displacement(tid);
            C_CSC->adjust(tid);
        Env::spmm_real_time[tid] += Env::toc(start_time);  
        
        //leader_tid = 0;
        //C_CSC->walk_dxd(false, leader_tid, tid);

        if(!tid) Env::iteration++;
    }
    
    auto finish = std::chrono::high_resolution_clock::now();
    if(!tid) Env::exec_time = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish - start).count())/1e9;
    Env::execution_time[tid] = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish - start).count())/1e9;
    
    C_tile = inputFeatures->tiles[Env::tile_index[tid]][0];
    std::shared_ptr<struct CSC<Weight>> C_CSC = C_tile.spmat;
    bool passed = validate_prediction(C_CSC, trueCategories, C_tile.start_row, tid);
    if(passed) {
        if(!tid) Logging::print(Logging::LOG_LEVEL::INFO, "Challenge PASSED.\n");
    }
    else {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Challenge FAILED.\n");
    }
    
    pthread_barrier_wait(&Env::thread_barrier);   
    
}

#endif 
