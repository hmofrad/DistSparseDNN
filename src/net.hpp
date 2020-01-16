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
#include <deque>

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
        std::vector<std::shared_ptr<struct Data_Block<Weight>>> spaWeightVec;
        
        std::unique_ptr<struct Tiling<Weight>> output = nullptr;

        uint32_t NinputInstanses;        
        uint32_t Nneurons;
        Weight biasValue;
        uint32_t maxLayers;
        int32_t nCategories;
        
        PARALLELISM_TYPE parallelism_type;
        bool repartition = false;
        bool greedy = false;
        
        void printTimes();
        void printTimesExcel();
        void execute();
        void inferenceReLU(const int32_t tid);
        
        void data_x_model(const int32_t tid);
        void data_x_data(const int32_t tid);
        void hybrid_x_hybrid(const int32_t tid);
        uint32_t hybrid_x_data(std::deque<int32_t>& my_threads, const int32_t tid);
        void hybrid_x_model(std::deque<int32_t>& my_threads, const uint32_t leader_rowgroup, const uint32_t leader_start_layer, const int32_t leader_tid, const int32_t tid);
        //void hybrid(std::vector<int32_t>& my_threads, const uint32_t leader_rowgroup, const uint32_t start_layer, const int32_t leader, const int32_t tid);
        
        bool add_to_idle_threads(std::deque<int32_t>& my_threads, const int32_t tid);
        int32_t add_to_my_follower_threads(std::deque<int32_t>& my_threads, const uint32_t start_layer, const uint32_t ncols, const int32_t tid);
        void    add_to_my_follower_threads(std::deque<int32_t>& my_threads, const uint32_t start_layer, const uint32_t ncols, const int32_t leader, const int32_t tid);
        
        int32_t add_to_idle_ranks(uint32_t& leader_rowgroup, uint32_t& start_layer, const int32_t tid);
        bool add_to_my_follower_ranks(const uint32_t leader_rowgroup, const uint32_t start_layer, const std::deque<int32_t> my_threads, const int32_t tid);
        
        
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
    else {
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
        nCategories = IO::text_file_categories(categoryFile, trueCategories, inputFeatures->nrows);
    }
    else {
        nCategories = IO::binary_file_categories(categoryFile, trueCategories, inputFeatures->nrows);
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
        else {
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
        else {
            spaWeightVec[i] = std::move(std::make_shared<struct Data_Block<Weight>>(inputFeatures->get_tile_info("height", i), Env::threads_socket_id[i]));
        }
    }

    if(parallelism_type == PARALLELISM_TYPE::_DATA_X_MODEL_) {
        output = std::move(std::make_unique<Tiling<Weight>>(Env::nranks, Env::nranks, 1, Env::nranks, 
                                                            0, inputFeatures->nrows, inputFeatures->ncols, 
                                                            TILING_TYPE::_1D_ROW_, repartition));
    }
    else {
        output = std::move(std::make_unique<Tiling<Weight>>(Env::nranks * Env::nthreads, Env::nranks * Env::nthreads, 1, Env::nranks, 
                                                            Env::nthreads, Env::nranks * Env::nthreads, 
                                                            0, inputFeatures->nrows, inputFeatures->ncols, 
                                                            TILING_TYPE::_1D_ROW_, repartition));
    }
    output->set_tile_info(inputFeatures->tiles);
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Running the inferenceReLU method [%s].\n", PARALLELISM_TYPES[parallelism_type]); 
    auto finish = std::chrono::high_resolution_clock::now();
    Env::io_time = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish-start).count())/1e9;
    Env::barrier();
    
    execute();

    finish = std::chrono::high_resolution_clock::now();
    Env::end_to_end_time = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish-start).count())/1e9;
    Env::barrier();
    printTimesExcel();
}

template<typename Weight>
void Net<Weight>::printTimesExcel() {
    Env::barrier();
    
    double sum = 0.0, mean = 0.0, std_dev = 0.0, min = 0.0, max = 0.0;
    /*
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics<double>(Env::io_time);
    Logging::print(Logging::LOG_LEVEL::VOID, "i/o time: mean, std_dev, min, max\n");
    Logging::print(Logging::LOG_LEVEL::VOID, "          %.3f %.3f %.3f %.3f\n", mean, std_dev, min, max);
    */
    int index = std::distance(Env::execution_time.begin(), std::max_element(Env::execution_time.begin(), Env::execution_time.end()));
    double exec_time = Env::execution_time[index];
    double spmm_sym_time = Env::spmm_symb_time[index];
    double spmm_time = Env::spmm_real_time[index];
    double memory_time = Env::memory_allocation_time[index];
    double hybrid_time = Env::hybrid_probe_time[index];
    
    /*
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics<double>(exec_time);
    Logging::print(Logging::LOG_LEVEL::VOID, "exe time: mean, std_dev, min, max\n");
    Logging::print(Logging::LOG_LEVEL::VOID, "          %.3f %.3f %.3f %.3f\n", mean, std_dev, min, max);
    Logging::print(Logging::LOG_LEVEL::VOID, "exe mean: spmm_symb spmm_real memory hybrid\n");
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics<double>(spmm_sym_time);
    Logging::print(Logging::LOG_LEVEL::VOID, "          %.3f ", mean);
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics<double>(spmm_time);
    Logging::print(Logging::LOG_LEVEL::VOID, "%.3f ", mean);
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics<double>(memory_time);
    Logging::print(Logging::LOG_LEVEL::VOID, "%.3f ", mean);
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics<double>(hybrid_time);
    Logging::print(Logging::LOG_LEVEL::VOID, "%.3f\n", mean);
    
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics<double>(Env::end_to_end_time);
    Logging::print(Logging::LOG_LEVEL::VOID, "run time: mean, std_dev, min, max\n");
    Logging::print(Logging::LOG_LEVEL::VOID, "          %.3f %.3f %.3f %.3f\n", mean, std_dev, min, max);
    */
    
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics<double>(exec_time);
    Logging::print(Logging::LOG_LEVEL::VOID, "exec time: %.3f %.3f %.3f %.3f ", mean, std_dev, min, max);
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics<double>(spmm_sym_time);
    Logging::print(Logging::LOG_LEVEL::VOID, "%.3f ", mean);
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics<double>(spmm_time);
    Logging::print(Logging::LOG_LEVEL::VOID, "%.3f ", mean);
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics<double>(memory_time);
    Logging::print(Logging::LOG_LEVEL::VOID, "%.3f ", mean);
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics<double>(hybrid_time);
    Logging::print(Logging::LOG_LEVEL::VOID, "%.3f\n", mean);
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
    if(Env::NUMA_ALLOC) {
        (void)Env::set_thread_affinity(tid);   
    }
    
    if(parallelism_type == PARALLELISM_TYPE::_DATA_X_MODEL_) {
        data_x_model(tid);
    }
    else if(parallelism_type == PARALLELISM_TYPE::_DATA_X_DATA_) {
        data_x_data(tid);
    }
    else if(parallelism_type == PARALLELISM_TYPE::_HYBRID_X_HYBRID_) {
        hybrid_x_hybrid(tid);
    }
}

template<typename Weight>
void Net<Weight>::data_x_model(const int32_t tid) {
    uint32_t leader_rowgroup = Env::rank;
    const int32_t leader_tid = 0;
    struct Env::thread_struct& thread_st = Env::threads[tid];
    
    struct Tile<Weight>& A_tile = inputFeatures->tiles[leader_rowgroup][0];
    struct Tile<Weight>& C_tile = output->tiles[leader_rowgroup][0];
    std::shared_ptr<struct Data_Block<Weight>> s_spa = spaWeightVec[tid];
    std::shared_ptr<struct Data_Block<Weight>> b_bias;
    
    const uint32_t nrows = inputFeatures->tile_height;
    const uint32_t ncols = layers[0]->ncols;
    uint32_t B_start_col, B_end_col;
    uint32_t B_sub_start_col, B_sub_end_col;

    auto start = std::chrono::high_resolution_clock::now();  
    for (uint32_t l = 0; l < maxLayers; l++) {
        std::shared_ptr<struct CSC<Weight>> A_CSC = A_tile.spmat;
        struct Tile<Weight>& B_tile = layers[l]->tiles[0][tid];
        std::shared_ptr<struct CSC<Weight>> B_CSC = B_tile.spmat;
        std::shared_ptr<struct CSC<Weight>> C_CSC = C_tile.spmat;
        b_bias = biasWeightVecs[l];
        B_start_col = 0;
        B_end_col = B_CSC->ncols;
        B_sub_start_col = B_tile.start_col;
        B_sub_end_col   = B_tile.end_col;
        
        data_x_model_1_iter(A_CSC, B_CSC, C_CSC, s_spa, b_bias, 
                            nrows, ncols, B_start_col, B_end_col, 
                            B_sub_start_col, B_sub_end_col, 
                            thread_st, leader_tid, tid);
    }
    auto finish = std::chrono::high_resolution_clock::now();
    
    Env::execution_time[tid] = (double)(std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count())/1e9;

    const std::shared_ptr<struct CSC<Weight>> A_CSC = A_tile.spmat;
    data_x_model_validate_prediction(A_CSC, A_tile.start_row, trueCategories, nCategories, leader_tid, tid);
}

template<typename Weight>
void Net<Weight>::data_x_data(const int32_t tid) {
    uint32_t leader_rowgroup = Env::thread_rowgroup[tid];
    int32_t leader_tid = 0;
    struct Env::thread_struct& thread_st = Env::threads[tid];
    
    std::shared_ptr<struct Data_Block<Weight>> s_spa = spaWeightVec[tid];
    std::shared_ptr<struct Data_Block<Weight>> b_bias;
    
    const uint32_t nrows = inputFeatures->tile_height;
    const uint32_t ncols = layers[0]->ncols;
    uint32_t B_start_col, B_end_col;
    const uint32_t B_off_col = 0;
    
    auto start = std::chrono::high_resolution_clock::now();  
    for (uint32_t l = 0; l < maxLayers; l++) {
        struct Tile<Weight>& A_tile = (not(l%2)) ? inputFeatures->tiles[leader_rowgroup][0]
                                                 : output->tiles[leader_rowgroup][0];
        std::shared_ptr<struct CSC<Weight>> A_CSC = A_tile.spmat;
        struct Tile<Weight>& B_tile = layers[l]->tiles[0][0];    
        std::shared_ptr<struct CSC<Weight>> B_CSC = B_tile.spmat;
        struct Tile<Weight>& C_tile = (not(l%2)) ? output->tiles[leader_rowgroup][0]
                                                 : inputFeatures->tiles[leader_rowgroup][0];
        std::shared_ptr<struct CSC<Weight>> C_CSC = C_tile.spmat;
        b_bias = biasWeightVecs[l];      
        B_start_col = 0;
        B_end_col = B_CSC->ncols;
        
        data_x_data_1_iter(A_CSC, B_CSC, C_CSC, s_spa, b_bias, 
                           nrows, ncols, B_start_col, B_end_col, B_off_col, 
                           thread_st, leader_tid, tid);       
    }
    auto finish = std::chrono::high_resolution_clock::now();
    
    Env::execution_time[tid] = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish - start).count())/1e9;
    
    struct Tile<Weight>& A_tile = inputFeatures->tiles[leader_rowgroup][0];
    const std::shared_ptr<struct CSC<Weight>> A_CSC = A_tile.spmat;
    data_x_data_validate_prediction(A_CSC, A_tile.start_row, trueCategories, nCategories, leader_tid, tid);
}

template<typename Weight>
void Net<Weight>::hybrid_x_hybrid(const int32_t tid) {
    
    //if(tid == 1)(5);
    uint32_t my_rowgroup = Env::thread_rowgroup[tid];
    int32_t my_leader_tid = 0;
    std::deque<int32_t> my_threads;
    
    auto start = std::chrono::high_resolution_clock::now();  
    if(!tid) {
        Env::global_time = Env::clock();
    }
    pthread_barrier_wait(&Env::thread_barrier);
    double elapsed_time = Env::clock() - Env::global_time;
    printf("Rank=%d tid=%2d time=%2.2f Started\n", Env::rank, tid, elapsed_time);
    uint32_t my_start_layer = hybrid_x_data(my_threads, tid);
    if(my_start_layer < maxLayers) {
        hybrid_x_model(my_threads, my_rowgroup, my_start_layer, tid, tid);
    }
    
    //printf("Rank=%d tid=%d added to queue\n", Env::rank, tid);
    while(add_to_idle_threads(my_threads, tid)) {
        const int32_t leader = Env::threads[tid].leader;
        if(leader != -1) {
            //printf("Rank=%d tid=%d Waked leader=%d\n", Env::rank, tid, leader);
            uint32_t leader_rowgroup = Env::threads[tid].rowgroup;
            uint32_t leader_start_layer = Env::threads[tid].start_layer;
            hybrid_x_model(my_threads, leader_rowgroup, leader_start_layer, leader, tid);
        }
    }
    auto finish = std::chrono::high_resolution_clock::now();
    elapsed_time = Env::clock() - Env::global_time;
    printf("Rank=%d tid=%2d time=%2.2f Finished\n", Env::rank, tid, elapsed_time);
    Env::execution_time[tid] = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish - start).count())/1e9;
    //printf("Rank=%d tid=%d done\n", Env::rank, tid);
    struct Tile<Weight>& A_tile = inputFeatures->tiles[my_rowgroup][0];
    const std::shared_ptr<struct CSC<Weight>> A_CSC = A_tile.spmat;
    data_x_data_validate_prediction(A_CSC, A_tile.start_row, trueCategories, nCategories, my_leader_tid, tid);
}

template<typename Weight>
uint32_t Net<Weight>::hybrid_x_data(std::deque<int32_t>& my_threads, const int32_t tid) {
    uint32_t leader_rowgroup = Env::thread_rowgroup[tid];
    int32_t leader_tid = 0;
    struct Env::thread_struct& thread_st = Env::threads[tid];
    //std::vector<int32_t> my_threads;
    my_threads.push_back(tid);
    //std::vector<int32_t> my_rowgrps;
    //bool has_data_to_share = true;
    
    std::shared_ptr<struct Data_Block<Weight>> s_spa = spaWeightVec[tid];
    std::shared_ptr<struct Data_Block<Weight>> b_bias;
    
    uint32_t nrows;
    const uint32_t ncols = layers[0]->ncols;
    uint32_t B_start_col, B_end_col;
    const uint32_t B_off_col = 0;
    uint32_t l = 0; 
    for (l = 0; l < maxLayers; l++) {
        if(add_to_my_follower_threads(my_threads, l, ncols, tid) == 1) {
            struct Tile<Weight>& A_tile = (not(l%2)) ? inputFeatures->tiles[leader_rowgroup][0]
                                                 : output->tiles[leader_rowgroup][0];
            std::shared_ptr<struct CSC<Weight>> A_CSC = A_tile.spmat;
            struct Tile<Weight>& B_tile = layers[l]->tiles[0][0];    
            std::shared_ptr<struct CSC<Weight>> B_CSC = B_tile.spmat;
            struct Tile<Weight>& C_tile = (not(l%2)) ? output->tiles[leader_rowgroup][0]
                                                     : inputFeatures->tiles[leader_rowgroup][0];
            std::shared_ptr<struct CSC<Weight>> C_CSC = C_tile.spmat;
            b_bias = biasWeightVecs[l];      
            B_start_col = 0;
            B_end_col = B_CSC->ncols;
            nrows = A_CSC->nrows;
            
            data_x_data_1_iter(A_CSC, B_CSC, C_CSC, s_spa, b_bias, 
                               nrows, ncols, B_start_col, B_end_col, B_off_col, 
                               thread_st, leader_tid, tid); 
            //Env::counters[tid].layer_index++;   
            Env::scores[tid]++;            
        }
        else {
            break;
        }
    }
    return(l);
}

template<typename Weight>
void Net<Weight>::hybrid_x_model(std::deque<int32_t>& my_threads, const uint32_t leader_rowgroup, const uint32_t leader_start_layer, const int32_t leader_tid, const int32_t tid) {
    struct Env::thread_struct& thread_st = Env::threads[tid];
    
    struct Tile<Weight>& A_tile = inputFeatures->tiles[leader_rowgroup][0];
    struct Tile<Weight>& C_tile = output->tiles[leader_rowgroup][0];
    std::shared_ptr<struct Data_Block<Weight>> s_spa = spaWeightVec[tid];
    std::shared_ptr<struct Data_Block<Weight>> b_bias;
    
    uint32_t nrows;
    const uint32_t ncols = layers[0]->ncols;
    uint32_t B_start_col, B_end_col;
    const uint32_t B_off_col = 0;   
    double start_time = 0;
    //printf("rank=%d tid=%d layer=%d leader=%d\n", Env::rank, tid, leader_start_layer, leader_tid);
    for (uint32_t l = leader_start_layer; l < maxLayers; l++) {
        //printf("rank=%d tid=%d layer=%d\n", Env::rank, tid, l);    
        
        //if(greedy) {
            //if(l > leader_start_layer) {
            start_time = Env::tic();   
                add_to_my_follower_threads(my_threads, l, ncols, leader_tid, tid);
                Env::decrease_num_threads(1, leader_tid, tid);
                Env::init_num_threads(my_threads.size(), leader_tid, tid);
            Env::hybrid_probe_time[tid] += Env::toc(start_time);     
          //  }
        //}
        
        std::shared_ptr<struct CSC<Weight>> A_CSC = A_tile.spmat;
        struct Tile<Weight>& B_tile = layers[l]->tiles[0][0];
        std::shared_ptr<struct CSC<Weight>> B_CSC = B_tile.spmat;
        std::shared_ptr<struct CSC<Weight>> C_CSC = C_tile.spmat;
        b_bias = biasWeightVecs[l];
        B_start_col = Env::threads[tid].start_col;
        B_end_col = Env::threads[tid].end_col;
        nrows = A_CSC->nrows;
        data_x_model_hybrid_1_iter(A_CSC, B_CSC, C_CSC, s_spa, b_bias, 
                                   nrows, ncols, B_start_col, B_end_col, B_off_col,
                                   my_threads, thread_st, leader_tid, tid);
       //Env::counters[tid].layer_index++;
       Env::scores[tid]++;
    }
}

template<typename Weight>
int32_t Net<Weight>::add_to_my_follower_threads(std::deque<int32_t>& my_threads, const uint32_t start_layer, const uint32_t ncols, const int32_t tid) {
    uint32_t num_threads = my_threads.size();
    //uint32_t old_num_threads = 0;
    //uint32_t new_num_threads = 0;
    if(greedy) {
        if(!Env::follower_threads.empty()) {
            pthread_mutex_lock(&Env::thread_mutex);  
            if(!Env::follower_threads.empty()) {
                num_threads += Env::follower_threads.size();
                my_threads.insert(my_threads.end(), Env::follower_threads.begin(), Env::follower_threads.end());
                Env::follower_threads.erase(Env::follower_threads.begin(), Env::follower_threads.end());
                for(uint32_t i = 0; i < num_threads; i++) {
                    int32_t t = my_threads[i];
                    Env::threads[t].thread_id = t;
                    Env::threads[t].leader = tid;
                    Env::threads[t].rowgroup = Env::thread_rowgroup[tid];
                    Env::threads[t].start_layer = start_layer;
                    Env::threads[t].start_col = ((ncols/num_threads) * i);
                    Env::threads[t].end_col   = (i == (num_threads-1)) ? ncols : ((ncols/num_threads) * (i+1));
                }
                Env::init_num_threads(num_threads, tid, tid);
                
                
                double elapsed_time = Env::clock() - Env::global_time;
                printf("Rank=%d tid=%2d time=%02.2f Added to my   queue my threads[", Env::rank, tid, elapsed_time);
                for(auto t: my_threads) {
                    printf("%d ", t);
                }
                printf("] layer = %d\n", start_layer);
                
            }
            
            pthread_barrier_init(&Env::thread_barriers[tid], NULL, num_threads);
            
            pthread_cond_broadcast(&Env::thread_cond); 
            pthread_mutex_unlock(&Env::thread_mutex);
        }
    }
    else {
        //printf("1.Rank=%d tid=%d %d\n", Env::rank, tid, start_layer);
        //double threshold = .9;
        //double progress = (double) start_layer/maxLayers;
        //if((progress < threshold)){// and (my_threads.size() == 1)) {
        if(start_layer > (.5 * maxLayers)){// and (my_threads.size() == 1)){            
            if(!Env::follower_threads.empty()) {
                pthread_mutex_lock(&Env::thread_mutex);  
                //old_num_threads = my_threads.size();
                if(!Env::follower_threads.empty()) {
                    uint32_t min_score_value = *std::min_element(Env::scores.begin(), Env::scores.end());
                    //uint32_t min_score_index = std::min_element(Env::scores.begin(), Env::scores.end()) - Env::scores.begin();
                    //printf("%d %d idx=%d v=%f %d\n", Env::rank, tid, best_score_index, best_score_value, start_layer);
                    //printf("%d %d [%lu %lu]\n", Env::rank, tid, Env::follower_threads.size(), my_threads.size());
                    if((Env::scores[tid] - min_score_value) < 5) {// and (Env::follower_threads.size() > 1)) {

                        int32_t ith = -1;
                        /*
                        std::deque<int>::iterator it;
                        for(it = Env::follower_threads.begin(); it != Env::follower_threads.end(); it++) {
                            if(Env::threads_socket_id[*it] == Env::threads_socket_id[tid]) {
                                ith = it - Env::follower_threads.begin();  
                                my_threads.push_back(*it);                                
                                Env::follower_threads.erase(it);
                                num_threads++;
                                break;
                            }
                        }
                        */
                        
                        //if(ith == -1) {
                            my_threads.push_back(Env::follower_threads.front());
                            Env::follower_threads.pop_front(); 
                            num_threads++;
                        //}
                        


                        for(uint32_t i = 0; i < num_threads; i++) {
                            int32_t t = my_threads[i];
                            Env::threads[t].thread_id = t;
                            Env::threads[t].leader = tid;
                            Env::threads[t].rowgroup = Env::thread_rowgroup[tid];
                            Env::threads[t].start_layer = start_layer;
                            Env::threads[t].start_col = ((ncols/num_threads) * i);
                            Env::threads[t].end_col   = (i == (num_threads-1)) ? ncols : ((ncols/num_threads) * (i+1));
                        }
                        
                        Env::init_num_threads(num_threads, tid, tid);
                        
                        double elapsed_time = Env::clock() - Env::global_time;
                        
                        printf("Rank=%d tid=%2d time=%02.2f Added to my   queue my threads[", Env::rank, tid, elapsed_time);
                        for(auto t: my_threads) {
                            printf("(%d %d)", t, Env::threads_socket_id[t]);
                        }
                        printf("] idle_threads=[");
                        for(auto t: Env::follower_threads) {
                            printf("%d ", t);
                        }
                        printf("] layer = %d\n", start_layer);
                        
                        
                    }
                    //printf("%d %d [%lu %lu]\n", Env::rank, tid, Env::follower_threads.size(), my_threads.size());

                }

                
            }
            pthread_barrier_init(&Env::thread_barriers[tid], NULL, num_threads);
            if(my_threads.size() == 2) {
                pthread_cond_signal(&Env::thread_cond); 
            }
            //pthread_cond_broadcast(&Env::thread_cond); 
            pthread_mutex_unlock(&Env::thread_mutex);
        }
    }
    //printf("2.Rank=%d tid=%d\n", Env::rank, tid);
    return(num_threads);
}


template<typename Weight>
void Net<Weight>::add_to_my_follower_threads(std::deque<int32_t>& my_threads, const uint32_t start_layer, const uint32_t ncols, const int32_t leader, const int32_t tid) {  
    uint32_t num_threads = my_threads.size();
    uint32_t old_num_threads = 0;
    uint32_t num_new_threads = 0;

    if(tid == leader) {
        if(greedy) {
            old_num_threads = my_threads.size();
            if(!Env::follower_threads.empty()) {
                pthread_mutex_lock(&Env::thread_mutex);  
                if(!Env::follower_threads.empty()) {
                    num_threads += Env::follower_threads.size();
                    my_threads.insert(my_threads.end(), Env::follower_threads.begin(), Env::follower_threads.end());
                    Env::follower_threads.erase(Env::follower_threads.begin(), Env::follower_threads.end());
                    for(uint32_t i = 0; i < num_threads; i++) {
                        int32_t t = my_threads[i];
                        Env::threads[t].leader = tid;
                        Env::threads[t].rowgroup = Env::thread_rowgroup[tid];
                        Env::threads[t].start_layer = start_layer;
                        Env::threads[t].start_col = ((ncols/num_threads) * i);
                        Env::threads[t].end_col   = (i == (num_threads-1)) ? ncols : ((ncols/num_threads) * (i+1));
                    }
                    pthread_barrier_destroy(&Env::thread_barriers[tid]);
                    pthread_barrier_init(&Env::thread_barriers[tid], NULL, num_threads);
                    
                    num_new_threads = num_threads - old_num_threads;
                    Env::increase_num_threads(num_new_threads, leader, tid);
                    
                    double elapsed_time = Env::clock() - Env::global_time;
                    printf("Rank=%d tid=%2d time=%02.2f Added to my   queue my threads[", Env::rank, tid, elapsed_time);
                    for(auto t: my_threads) {
                        printf("%d ", t);
                    }
                    printf("] layer = %d\n", start_layer);
                    
                    
                }
                pthread_cond_broadcast(&Env::thread_cond); 
                pthread_mutex_unlock(&Env::thread_mutex);
            }
        }
        else {
            old_num_threads = my_threads.size();
            if(!Env::follower_threads.empty()) {
                pthread_mutex_lock(&Env::thread_mutex);  
                

                
                
                if(!Env::follower_threads.empty()) {                                    
                    uint32_t nfinished = Env::num_finished_threads;
                    uint32_t nworking = Env::nthreads - Env::num_finished_threads;
                    uint32_t nidles = Env::follower_threads.size();
                    uint32_t nhelping = Env::num_finished_threads - Env::follower_threads.size();
                    
                    double elapsed_time = Env::clock() - Env::global_time;
                    printf("Rank=%d tid=%2d time=%02.2f nworking=%d nfinished=%d nhelping=%d nidles=%d layer=%d\n", Env::rank, tid, elapsed_time, nworking, nfinished, nhelping, nidles, start_layer);
                    
                    uint32_t min_score_value = *std::min_element(Env::scores.begin(), Env::scores.end());
                    if(((Env::scores[tid] - min_score_value) < 5) or (nidles > nworking)) {
                        
                        my_threads.push_back(Env::follower_threads.front());
                        Env::follower_threads.pop_front(); 
                        num_threads++;
                        
                        for(uint32_t i = 0; i < num_threads; i++) {
                            int32_t t = my_threads[i];
                            Env::threads[t].thread_id = t;
                            Env::threads[t].leader = tid;
                            Env::threads[t].rowgroup = Env::thread_rowgroup[tid];
                            Env::threads[t].start_layer = start_layer;
                            Env::threads[t].start_col = ((ncols/num_threads) * i);
                            Env::threads[t].end_col   = (i == (num_threads-1)) ? ncols : ((ncols/num_threads) * (i+1));
                        }
  
                        pthread_barrier_destroy(&Env::thread_barriers[tid]);
                        pthread_barrier_init(&Env::thread_barriers[tid], NULL, num_threads);
                        
                        num_new_threads = num_threads - old_num_threads;
                        Env::increase_num_threads(num_new_threads, leader, tid);
                        
                        double elapsed_time = Env::clock() - Env::global_time;
                        printf("Rank=%d tid=%2d time=%02.2f Added to my   queue my threads[", Env::rank, tid, elapsed_time);
                        for(auto t: my_threads) {
                            printf("%d ", t);
                        }
                        printf("] layer = %d\n", start_layer);
                    }    
                    
                }
                for(uint32_t i = 0; i < num_new_threads; i++) {
                    pthread_cond_signal(&Env::thread_cond); 
                }
                pthread_mutex_unlock(&Env::thread_mutex);
            }
            
        }
    }
}



template<typename Weight>
bool Net<Weight>::add_to_idle_threads(std::deque<int32_t>& my_threads, const int32_t tid) {
    bool status = true;
    if(Env::follower_threads.size() != (uint32_t) Env::nthreads) {
        pthread_mutex_lock(&Env::thread_mutex);
            Env::follower_threads.push_back(tid);
            double elapsed_time = Env::clock() - Env::global_time;
            printf("Rank=%d tid=%2d time=%2.2f Added to idle queue idle_threads=[", Env::rank, tid, elapsed_time);
            for(auto t: Env::follower_threads) {
                printf("%d ", t);
            }
            printf("]\n");
            my_threads.erase(my_threads.begin(), my_threads.end());
            Env::threads[tid].leader = -1;
            if(Env::finished_threads[tid]) {
                Env::num_finished_threads++;
                Env::finished_threads[tid] = false;
            }
            if(Env::follower_threads.size() != (uint32_t) Env::nthreads) {
                pthread_cond_wait(&Env::thread_cond, &Env::thread_mutex); 
            }
            else {
                pthread_cond_broadcast(&Env::thread_cond);    
                status = false;
            }
        pthread_mutex_unlock(&Env::thread_mutex);
        //printf("Rank=%d tid=%d just waked up\n", Env::rank, tid);
    }
    else {
        status = false;
    }
    return(status);
}



/*
template<typename Weight>
void Net<Weight>::hybrid_x_hybrid(const int32_t tid) {
    if(Env::NUMA_ALLOC)
        (void)Env::set_thread_affinity(tid); 
    
    double start_time = 0;
    uint64_t nnz   = 0;
    uint32_t nrows = inputFeatures->tile_height;
    uint32_t ncols = layers[0]->ncols;
    
    std::shared_ptr<struct Data_Block<Weight>> s_spa = spaWeightVec[tid];
    struct Env::thread_struct& thread_st = Env::threads[tid];
    std::vector<int32_t> my_threads;
    my_threads.push_back(tid);
    std::vector<int32_t> my_rowgrps;
    bool has_data_to_share = true;
    auto start = std::chrono::high_resolution_clock::now();  

    for (uint32_t l = 0; l < maxLayers; l++) {
        if(add_to_my_follower_threads(my_threads, l, ncols, tid) == 1) {
            start_time = Env::tic(); 
                struct Tile<Weight>& A_tile = (not(l%2)) ? inputFeatures->tiles[Env::thread_rowgroup[tid]][0]
                                                         : output->tiles[Env::thread_rowgroup[tid]][0];
                                                         
                struct Tile<Weight>& C_tile = (not(l%2)) ? output->tiles[Env::thread_rowgroup[tid]][0]
                                                         : inputFeatures->tiles[Env::thread_rowgroup[tid]][0];
                std::shared_ptr<struct CSC<Weight>> A_CSC = A_tile.spmat;
                std::shared_ptr<struct CSC<Weight>> C_CSC = C_tile.spmat;
                struct Tile<Weight>& B_tile = layers[l]->tiles[0][0];
                std::shared_ptr<struct CSC<Weight>> B_CSC = B_tile.spmat;
                std::shared_ptr<struct Data_Block<Weight>> b_bias = biasWeightVecs[l];  
            Env::spmm_symb_time[tid] += Env::toc(start_time);      
            
            start_time = Env::tic();
            if(has_data_to_share) {
                has_data_to_share = add_to_my_follower_ranks(Env::thread_rowgroup[tid], l, my_threads, tid);
            }
        
            start_time = Env::tic(); 
                const uint32_t start_col = 0;
                const uint32_t end_col   = B_CSC->ncols;
                std::tie(thread_st.off_nnz, nrows, ncols) =  spmm_symb(A_CSC, B_CSC, s_spa, start_col, end_col, tid);
            Env::spmm_symb_time[tid] += Env::toc(start_time);      
            
            start_time = Env::tic();
                int32_t leader_tid = -1;
                C_CSC->reallocate(thread_st.off_nnz, nrows, ncols, leader_tid, tid);
            Env::memory_allocation_time[tid] += Env::toc(start_time);
            
            start_time = Env::tic();
                uint32_t off_col   = 0;
                thread_st.idx_nnz = 0;
                spmm_real(A_CSC, B_CSC, C_CSC, s_spa, b_bias, start_col, end_col, off_col, thread_st.idx_nnz, tid);
                Env::adjust_displacement(tid);
                C_CSC->adjust(tid);
                A_tile.nedges = A_CSC->nnz_i;
                C_tile.nedges = C_CSC->nnz_i;
            Env::spmm_real_time[tid] += Env::toc(start_time);  
            if(!tid) Env::iteration++;
        }
        else {
            hybrid(my_threads, Env::thread_rowgroup[tid], l, tid, tid);
            break;
        }
        
    }

    while(add_to_idle_threads(my_threads, tid)) {
        const int32_t leader = Env::threads[tid].leader;
        if(leader != -1) {
            uint32_t leader_rowgroup = Env::threads[tid].rowgroup;
            uint32_t start_layer = Env::threads[tid].start_layer;
            hybrid(my_threads, leader_rowgroup, start_layer, leader, tid);
        }
    }
    
    uint32_t leader_rowgroup = 0;
    uint32_t start_layer = 0;     
    int32_t ret = 0;
    while((ret = add_to_idle_ranks(leader_rowgroup, start_layer, tid))) {
        if(ret == 1) {
            if(std::find(my_rowgrps.begin(), my_rowgrps.end(), leader_rowgroup) == my_rowgrps.end()) {
                my_rowgrps.push_back(leader_rowgroup);
            }

            for (uint32_t l = start_layer; l < maxLayers; l++) {
                start_time = Env::tic();
                    struct Tile<Weight>& A_tile = (not(l%2)) ? inputFeatures->tiles[leader_rowgroup][0].in_subtiles.back()
                                                             : output->tiles[leader_rowgroup][0].in_subtiles.back();
                                                             
                    struct Tile<Weight>& C_tile = (not(l%2)) ? output->tiles[leader_rowgroup][0].in_subtiles.back()
                                                             : inputFeatures->tiles[leader_rowgroup][0].in_subtiles.back();                

                    std::shared_ptr<struct CSC<Weight>> A_CSC = A_tile.spmat;
                    std::shared_ptr<struct CSC<Weight>> C_CSC = C_tile.spmat;
                    struct Tile<Weight>& B_tile = layers[l]->tiles[0][0];
                    std::shared_ptr<struct CSC<Weight>> B_CSC = B_tile.spmat;
                    std::shared_ptr<struct Data_Block<Weight>> b_bias = biasWeightVecs[l];
                    const uint32_t start_col = 0;
                    const uint32_t end_col   = B_CSC->ncols;
                    std::tie(thread_st.off_nnz, nrows, ncols) =  spmm_symb(A_CSC, B_CSC, s_spa, start_col, end_col, tid);
                Env::spmm_symb_time[tid] += Env::toc(start_time);
                
                start_time = Env::tic();
                    int32_t leader_tid = -1;
                    C_CSC->reallocate(thread_st.off_nnz, nrows, ncols, leader_tid, tid);
                Env::memory_allocation_time[tid] += Env::toc(start_time);
                
                start_time = Env::tic();
                    uint32_t off_col   = 0;
                    thread_st.idx_nnz = 0;
                    spmm_real(A_CSC, B_CSC, C_CSC, s_spa, b_bias, start_col, end_col, off_col, thread_st.idx_nnz, tid);
                    Env::adjust_displacement(tid);
                    C_CSC->adjust(tid);
                Env::spmm_real_time[tid] += Env::toc(start_time);
                
            }
        }
    }
    
    pthread_barrier_wait(&Env::thread_barrier);
    Env::barrier();
    
    if (!Env::rank and !tid) {
        printf("Rank=%d:\n", Env::rank);
        for(int j = 0; j < Env::nthreads; j++) {
            printf("%d: ", j);
            for(int i =0;i < Env::nranks+1; i++) {
                int& k = *(Env::idle_threads[j] + i);
                printf("%d ", k);
            }
            printf("\n");
        }
        printf("\n");
    }
    pthread_barrier_wait(&Env::thread_barrier);
    Env::barrier();
    
    auto finish = std::chrono::high_resolution_clock::now();
    Env::execution_time[tid] = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish - start).count())/1e9;

    struct Tile<Weight>& C_tile = inputFeatures->tiles[Env::thread_rowgroup[tid]][0];
    std::shared_ptr<struct CSC<Weight>>& C_spmat = C_tile.spmat;
    int32_t count = validate_prediction1(C_spmat, trueCategories, C_tile.start_row);
    
    for(uint32_t rg: my_rowgrps) {
        for(auto tile: inputFeatures->tiles[rg][0].in_subtiles) {
            count += validate_prediction1(tile.spmat, trueCategories, tile.start_row);
        }
    }
    
    Env::counters[tid].checkcount = count;
    pthread_barrier_wait(&Env::thread_barrier);
    Env::barrier();
    if(!tid) {
        int counts = 0;
        for(auto counter: Env::counters) {
            counts += (int) counter.checkcount;
        }
        int countss = 0;
        MPI_Allreduce(&counts, &countss, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        bool passed = (countss == nCategories);
        if(passed) {
            if(!tid) Logging::print(Logging::LOG_LEVEL::INFO, "Challenge PASSED.\n");
        }
        else {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Challenge FAILED.\n");
        }
    }
    pthread_barrier_wait(&Env::thread_barrier);
    Env::barrier();
    
}

*/

template<typename Weight>
int32_t Net<Weight>::add_to_idle_ranks(uint32_t& leader_rowgroup, uint32_t& start_layer, const int32_t tid){
    MPI_Request request;
    std::vector<MPI_Request> requests;
    MPI_Datatype WEIGHT_TYPE = MPI_Types::get_mpi_data_type<Weight>();   
    
    int32_t done = 1;
    int window_host_rank = 0;
    int num_follower_ranks = 0;
    int some_val = 1;
    int some_res = 0;
    int some_cmp = 0;
    
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, window_host_rank, 0, Env::thread_windows[tid]);    
    
        MPI_Fetch_and_op(&some_val, &some_res, MPI_INT, window_host_rank, Env::nranks, MPI_SUM, Env::thread_windows[tid]);
        MPI_Win_flush(window_host_rank, Env::thread_windows[tid]);
        num_follower_ranks = some_res+1;

        some_val = 1;
        MPI_Fetch_and_op(&some_val, &some_res, MPI_INT, window_host_rank, Env::rank, MPI_REPLACE, Env::thread_windows[tid]);
        MPI_Win_flush(window_host_rank, Env::thread_windows[tid]);

    MPI_Win_unlock(window_host_rank, Env::thread_windows[tid]);
    
    if(num_follower_ranks == Env::nranks) {
        for(int i = 0; i < Env::nranks; i++) {
            if(i != Env::rank) {
                int follower_rank = i;
                std::vector<uint32_t> csc_metadata = {0, 0, 0, 0, 0, 0, 0};
                MPI_Isend(csc_metadata.data(), csc_metadata.size(), MPI_UNSIGNED, follower_rank, follower_rank, Env::thread_communicators[tid], &request);
                requests.push_back(request);
            }
        }
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        requests.clear();
        requests.shrink_to_fit();
        done = 0;
    }
    else {
        MPI_Status status;
        int source_rank;
        int source_tag;

        do{    
            int flag = 0;
            while(!flag) {
                MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, Env::thread_communicators[tid], &flag, &status);
            }
            source_rank = status.MPI_SOURCE;
            source_tag = status.MPI_TAG;
        } while(source_tag != Env::rank);
        
        
        std::vector<uint32_t> csc_metadata(7);
        MPI_Recv(csc_metadata.data(), csc_metadata.size(), MPI_UNSIGNED, source_rank, source_tag, Env::thread_communicators[tid], &status);
        
        uint32_t msg_status = csc_metadata[0];
        
        if(msg_status) {
            leader_rowgroup = csc_metadata[1];
            start_layer = csc_metadata[2];
            uint64_t csc_nedges = (uint64_t) csc_metadata[3];
            uint32_t csc_start_row = csc_metadata[4];
            uint32_t csc_height = csc_metadata[5];
            uint32_t csc_width = csc_metadata[6];
            if(not(start_layer%2)) {
                inputFeatures->update_in_subtiles(leader_rowgroup, start_layer, output->tiles, csc_nedges, csc_start_row, csc_height, csc_width, tid);
            }
            else {
                output->update_in_subtiles(leader_rowgroup, start_layer, inputFeatures->tiles, csc_nedges, csc_start_row, csc_height, csc_width, tid);
            }
            
            struct Tile<Weight>& A_tile = (not(start_layer%2)) ? inputFeatures->tiles[leader_rowgroup][0].in_subtiles.back()
                                                               : output->tiles[leader_rowgroup][0].in_subtiles.back();
            
            std::shared_ptr<struct CSC<Weight>>& A_CSC = A_tile.spmat;
            A_CSC->Irecv(requests,source_rank, tid);
            
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
            requests.clear();
            requests.shrink_to_fit();
        }
        done = msg_status;
    }
    return(done);
}

template<typename Weight>
bool Net<Weight>::add_to_my_follower_ranks(const uint32_t leader_rowgroup, const uint32_t start_layer, const std::deque<int32_t> my_threads, const int32_t tid) {
    bool done = true;
    int window_host_rank = 0;
    int num_follower_ranks = 0;
    int some_val = 0;
    int some_res = 0;
    int some_cmp = 0;
    std::vector<int32_t> follower_ranks;

    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, window_host_rank, 0, Env::thread_windows[tid]);
        MPI_Fetch_and_op(&some_val, &some_res, MPI_INT, window_host_rank, Env::nranks, MPI_NO_OP, Env::thread_windows[tid]);
        MPI_Win_flush(window_host_rank, Env::thread_windows[tid]);
        num_follower_ranks = some_res;
        
        if(num_follower_ranks) {
        for(int i = 0; i < Env::nranks; i++) {
            if(i != Env::rank) {
                some_val = 0;
                some_res = 0;
                some_cmp = 1;
                
                MPI_Compare_and_swap(&some_val, &some_cmp, &some_res, MPI_INT, window_host_rank, i, Env::thread_windows[tid]);
                MPI_Win_flush_all(Env::thread_windows[tid]);
                if(some_res){
                    some_val = -1;
                    MPI_Fetch_and_op(&some_val, &some_res, MPI_INT, window_host_rank, Env::nranks, MPI_SUM, Env::thread_windows[tid]);
                    MPI_Win_flush(window_host_rank, Env::thread_windows[tid]);
                    
                    some_val = 0;
                    MPI_Fetch_and_op(&some_val, &some_res, MPI_INT, window_host_rank, i, MPI_REPLACE, Env::thread_windows[tid]);
                    MPI_Win_flush(window_host_rank, Env::thread_windows[tid]);
                    
                    int follower_rank = i;
                    follower_ranks.push_back(follower_rank);
                }
            }   
        }
    }
    MPI_Win_unlock(window_host_rank, Env::thread_windows[tid]);
    
    if(not follower_ranks.empty()) {
        MPI_Request request;
        std::vector<MPI_Request> requests;
        
        std::vector<struct Tile<Weight>> subtiles;
        std::vector<std::shared_ptr<struct CSC<Weight>>> subcscs;
        uint32_t nthreads_local = my_threads.size();
        if(not(start_layer%2)) {
            inputFeatures->update_out_subtiles(leader_rowgroup, start_layer, output->tiles, subtiles, subcscs, follower_ranks, nthreads_local, tid);
        }
        else {
            output->update_out_subtiles(leader_rowgroup, start_layer, inputFeatures->tiles, subtiles, subcscs, follower_ranks, nthreads_local, tid);
        }
        for(uint32_t k = 1; k < subtiles.size(); k++) {
            struct Tile<Weight> subtile = subtiles[k];
            uint32_t msg_status = (subtile.nedges) ? 1 : 2;

            std::vector<uint32_t> csc_metadata = {msg_status, leader_rowgroup, start_layer, (uint32_t)subtile.nedges, subtile.start_row, subtile.height, subtile.width};
            MPI_Isend(csc_metadata.data(), csc_metadata.size(), MPI_UNSIGNED, subtile.rank, subtile.rank, Env::thread_communicators[tid], &request);        
            requests.push_back(request);
            
            if(msg_status == 2) {
                done = false;
            }
        }
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        requests.clear();
        requests.shrink_to_fit();
        
        for(uint32_t k = 1; k < subtiles.size(); k++) {
            struct Tile<Weight> subtile = subtiles[k];
            std::shared_ptr<struct CSC<Weight>>& subcsc = subcscs[k];
            subcsc->Isend(requests, subtile.rank, tid);
        }
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        requests.clear();
        requests.shrink_to_fit();
        
        subtiles.clear();
        subtiles.shrink_to_fit();
        subcscs.clear();
        subcscs.shrink_to_fit();
        
        follower_ranks.clear();
        follower_ranks.shrink_to_fit();
        
        struct Tile<Weight>& A_tile = (not(start_layer%2)) ? inputFeatures->tiles[leader_rowgroup][0]
                                                           : output->tiles[leader_rowgroup][0];
    }
    return(done);
}
#endif 