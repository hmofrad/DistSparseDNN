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
        std::vector<std::shared_ptr<struct Data_Block<Weight>>> spaWeightVec;
        
        std::unique_ptr<struct Tiling<Weight>> output = nullptr;

        uint32_t NinputInstanses;        
        uint32_t Nneurons;
        Weight biasValue;
        uint32_t maxLayers;
        int32_t nCategories;
        
        PARALLELISM_TYPE parallelism_type;
        bool repartition = false;
        
        void printTimes();
        void printTimesExcel();
        void execute();
        void inferenceReLU(const int32_t tid);
        
        void data_x_model(const int32_t tid);
        void data_x_data(const int32_t tid);
        void hybrid_x_hybrid(const int32_t tid);
        void hybrid(std::vector<int32_t>& my_threads, const uint32_t leader_rowgroup, const uint32_t start_layer, const int32_t leader, const int32_t tid);
        
        bool add_to_idle_threads(std::vector<int32_t>& my_threads, const int32_t tid);
        int32_t add_to_my_follower_threads(std::vector<int32_t>& my_threads, const uint32_t start_layer, const uint32_t ncols, const int32_t tid);
        void    add_to_my_follower_threads(std::vector<int32_t>& my_threads, const uint32_t start_layer, const uint32_t ncols, const int32_t leader, const int32_t tid);
        
        int32_t add_to_idle_ranks(uint32_t& leader_rowgroup, uint32_t& start_layer, const int32_t tid);
        bool add_to_my_follower_ranks(const uint32_t leader_rowgroup, const uint32_t start_layer, const std::vector<int32_t> my_threads, const int32_t tid);
        
        
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
    //maxLayers = 3;
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
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Running the inferenceReLU method.\n"); 
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
    if(Env::NUMA_ALLOC)
        (void)Env::set_thread_affinity(tid);   
    
    double start_time = 0;
    uint64_t nnz   = 0;
    uint32_t nrows = inputFeatures->tile_height;
    uint32_t ncols = layers[0]->ncols;
    
    struct Tile<Weight> A_tile;
    struct Tile<Weight> B_tile;
    struct Tile<Weight> C_tile;
    
    std::shared_ptr<struct Data_Block<Weight>> s_spa = spaWeightVec[tid];
    struct Env::thread_struct& thread_st = Env::threads[tid];
    
    auto start = std::chrono::high_resolution_clock::now();  
    for (uint32_t l = 0; l < maxLayers; l++) {
        start_time = Env::tic(); 
            A_tile = inputFeatures->tiles[Env::rank][0];
            std::shared_ptr<struct CSC<Weight>> A_CSC = A_tile.spmat;
            B_tile = layers[l]->tiles[0][tid];
            std::shared_ptr<struct CSC<Weight>> B_CSC = B_tile.spmat;
            C_tile = output->tiles[Env::rank][0];
            std::shared_ptr<struct CSC<Weight>> C_CSC = C_tile.spmat;
            std::shared_ptr<struct Data_Block<Weight>> b_bias = biasWeightVecs[l];

            uint32_t start_col = 0;
            uint32_t end_col   = B_CSC->ncols;
            std::tie(thread_st.off_nnz, nrows, std::ignore) =  spmm_symb(A_CSC, B_CSC, s_spa, start_col, end_col, tid);
            pthread_barrier_wait(&Env::thread_barrier);
        Env::spmm_symb_time[tid] += Env::toc(start_time);   
        
        start_time = Env::tic();
            const int32_t leader_tid = 0;
            nnz = Env::adjust_nnz(leader_tid, tid);
            C_CSC->reallocate(nnz, nrows, ncols, leader_tid, tid);
        Env::memory_allocation_time[tid] += Env::toc(start_time);
        
        start_time = Env::tic();
            uint32_t off_col   = B_tile.start_col;
            pthread_barrier_wait(&Env::thread_barrier);
            spmm_real(A_CSC, B_CSC, C_CSC, s_spa, b_bias, start_col, end_col, off_col, thread_st.idx_nnz, tid);
            pthread_barrier_wait(&Env::thread_barrier);
            Env::adjust_displacement(tid);
            C_CSC->adjust(leader_tid, tid);	
        Env::spmm_real_time[tid] += Env::toc(start_time);
        
        start_time = Env::tic();
            start_col = B_tile.start_col;
            end_col   = B_tile.end_col;
            uint32_t dis_nnz = thread_st.dis_nnz;
            A_CSC->repopulate(C_CSC, start_col, end_col, dis_nnz, leader_tid, tid);
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
    std::shared_ptr<struct Data_Block<Weight>> s_spa = spaWeightVec[tid];
    struct Env::thread_struct& thread_st = Env::threads[tid];
    
    auto start = std::chrono::high_resolution_clock::now();  
    for (uint32_t l = 0; l < maxLayers; l++) {
        start_time = Env::tic(); 
            if(not(l%2)) {
                A_tile = inputFeatures->tiles[Env::thread_rowgroup[tid]][0];
                C_tile = output->tiles[Env::thread_rowgroup[tid]][0];
            }
            else {
                A_tile = output->tiles[Env::thread_rowgroup[tid]][0];
                C_tile = inputFeatures->tiles[Env::thread_rowgroup[tid]][0];
            }

            std::shared_ptr<struct CSC<Weight>> A_CSC = A_tile.spmat;
            std::shared_ptr<struct CSC<Weight>> C_CSC = C_tile.spmat;
            B_tile = layers[l]->tiles[0][0];
            std::shared_ptr<struct CSC<Weight>> B_CSC = B_tile.spmat;
            auto& b_bias = biasWeightVecs[l];  

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
        
        //leader_tid = 0;
        //C_CSC->walk_dxd(false, leader_tid, tid);

        if(!tid) Env::iteration++;
    }
    
    auto finish = std::chrono::high_resolution_clock::now();
    //if(!tid) Env::exec_time = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish - start).count())/1e9;
    Env::execution_time[tid] = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish - start).count())/1e9;
    
    C_tile = inputFeatures->tiles[Env::thread_rowgroup[tid]][0];
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
    if(Env::rank == 1) {
        sleep(5);
    }
    for (uint32_t l = 0; l < maxLayers; l++) {
        //printf("hybrid_x_hybrid: rank=%d tid=%d layer=%d nthreads=%lu\n", Env::rank, tid, l, my_threads.size());
        if(add_to_my_follower_threads(my_threads, l, ncols, tid) == 1) {
            //printf("rank=%d tid=%d layer=%d Adding\n", Env::rank, tid, l);    
            //printf("rank=%d tid=%d layer=%d Done\n", Env::rank, tid, l);
            start_time = Env::tic(); 
                struct Tile<Weight>& A_tile = (not(l%2)) ? inputFeatures->tiles[Env::thread_rowgroup[tid]][0]
                                                         : output->tiles[Env::thread_rowgroup[tid]][0];
                                                         
                struct Tile<Weight>& C_tile = (not(l%2)) ? output->tiles[Env::thread_rowgroup[tid]][0]
                                                         : inputFeatures->tiles[Env::thread_rowgroup[tid]][0];                
                
                /*
                if(not(l%2)) {
                    A_tile = inputFeatures->tiles[Env::thread_rowgroup[tid]][0];
                    C_tile = output->tiles[Env::thread_rowgroup[tid]][0];
                }
                else {
                    A_tile = output->tiles[Env::thread_rowgroup[tid]][0];
                    C_tile = inputFeatures->tiles[Env::thread_rowgroup[tid]][0];
                }
                */
                //printf("%d %d: %lu %d %d\n", Env::rank, tid, A_tile.nedges, A_tile.width, A_tile.height);
                //printf("%d %d: %lu %d %d\n", Env::rank, tid, C_tile.nedges, C_tile.width, C_tile.height);

                std::shared_ptr<struct CSC<Weight>> A_CSC = A_tile.spmat;
                std::shared_ptr<struct CSC<Weight>> C_CSC = C_tile.spmat;
                struct Tile<Weight>& B_tile = layers[l]->tiles[0][0];
                std::shared_ptr<struct CSC<Weight>> B_CSC = B_tile.spmat;
                std::shared_ptr<struct Data_Block<Weight>> b_bias = biasWeightVecs[l];  

            Env::spmm_symb_time[tid] += Env::toc(start_time);      
        
            //if(A_tile.nedges) {
                //printf("rank=%d tid=%d layer=%d nnz=%lu =%lu =%lu [%d %d]\n", Env::rank, tid, l, A_tile.nedges, C_tile.nedges, thread_st.off_nnz, A_tile.height, C_tile.height);
                
                start_time = Env::tic();
                if(has_data_to_share) {
                    has_data_to_share = add_to_my_follower_ranks(Env::thread_rowgroup[tid], l, my_threads, tid);
                }
                Env::hybrid_probe_time[tid] += Env::toc(start_time);  
                
            //}
            //else {
              //  printf("######## rank=%d tid=%d layer=%d\n", Env::rank, tid, l);
            //}
        
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
            //printf("rank=%d tid=%d layer=%d hybrid Done\n", Env::rank, tid, l);
            break;
        }
        
    }
    //printf("rank=%d tid=%d add_to_idle_threads\n", Env::rank, tid);
    while(add_to_idle_threads(my_threads, tid)) {
        const int32_t leader = Env::threads[tid].leader;
        if(leader != -1) {
            uint32_t leader_rowgroup = Env::threads[tid].rowgroup;
            uint32_t start_layer = Env::threads[tid].start_layer;
            hybrid(my_threads, leader_rowgroup, start_layer, leader, tid);
        }
    }
    //printf("Rank=%d, tid=%d add_to_idle_ranks\n", Env::rank, tid);
    
    uint32_t leader_rowgroup = 0;
    uint32_t start_layer = 0;     
    int32_t ret = 0;
    while((ret = add_to_idle_ranks(leader_rowgroup, start_layer, tid))) {
        if(ret == 1) {
        if(std::find(my_rowgrps.begin(), my_rowgrps.end(), leader_rowgroup) == my_rowgrps.end()) {
            my_rowgrps.push_back(leader_rowgroup);
        }
        //auto& s_spa = Net::spaWeightVec[tid];
        //printf("rank=%d tid=%d rowgroup=%d start_layer=%d\n", Env::rank, tid, leader_rowgroup, start_layer);
        for (uint32_t l = start_layer; l < maxLayers; l++) {
          //  printf("rank=%d tid=%d layer=%d\n", Env::rank, tid, l);
            start_time = Env::tic();
                struct Tile<Weight>& A_tile = (not(l%2)) ? inputFeatures->tiles[leader_rowgroup][0].in_subtiles.back()
                                                         : output->tiles[leader_rowgroup][0].in_subtiles.back();
                                                         
                struct Tile<Weight>& C_tile = (not(l%2)) ? output->tiles[leader_rowgroup][0].in_subtiles.back()
                                                         : inputFeatures->tiles[leader_rowgroup][0].in_subtiles.back();                
            
            /*
                if(not(l%2)) {
                    A_tile = inputFeatures->tiles[leader_rowgroup][0].in_subtiles.back();
                    C_tile = output->tiles[leader_rowgroup][0].in_subtiles.back();
                }
                else {
                    A_tile = output->tiles[leader_rowgroup][0].in_subtiles.back();
                    C_tile = inputFeatures->tiles[leader_rowgroup][0].in_subtiles.back();
                }
            */
                std::shared_ptr<struct CSC<Weight>> A_CSC = A_tile.spmat;
                std::shared_ptr<struct CSC<Weight>> C_CSC = C_tile.spmat;
                struct Tile<Weight>& B_tile = layers[l]->tiles[0][0];
                std::shared_ptr<struct CSC<Weight>> B_CSC = B_tile.spmat;
                std::shared_ptr<struct Data_Block<Weight>> b_bias = biasWeightVecs[l];
                
                //printf("rank=%d tid=%d: layer=%d rg=%d A[%d %d] C[%d %d] SPA{%lu}\n", Env::rank, tid, l, leader_rowgroup, A_tile.width, A_tile.height, C_tile.width, C_tile.height, s_spa->nitems);
                //printf("%d %d: %lu %d %d\n", Env::rank, tid, C_tile.nedges, C_tile.width, C_tile.height);
                
                
              
                
                
                
                
                
                
            Env::spmm_symb_time[tid] += Env::toc(start_time);
            //printf("%d %d Symb start\n", Env::rank, tid);
            start_time = Env::tic();
                const uint32_t start_col = 0;
                const uint32_t end_col   = B_CSC->ncols;
                std::tie(thread_st.off_nnz, nrows, ncols) =  spmm_symb(A_CSC, B_CSC, s_spa, start_col, end_col, tid);
            Env::spmm_symb_time[tid] += Env::toc(start_time);
            //printf("%d %d Symb end %lu %d %d\n",Env::rank, tid, thread_st.off_nnz, nrows, ncols);
            start_time = Env::tic();
                int32_t leader_tid = -1;
                C_CSC->reallocate(thread_st.off_nnz, nrows, ncols, leader_tid, tid);
            Env::memory_allocation_time[tid] += Env::toc(start_time);
            //printf("%d %d spmm start\n",Env::rank, tid);
            start_time = Env::tic();
                uint32_t off_col   = 0;
                thread_st.idx_nnz = 0;
                spmm_real(A_CSC, B_CSC, C_CSC, s_spa, b_bias, start_col, end_col, off_col, thread_st.idx_nnz, tid);
                Env::adjust_displacement(tid);
                C_CSC->adjust(tid);
            Env::spmm_real_time[tid] += Env::toc(start_time);
            //printf("%d %d spmm end\n",Env::rank, tid);
        }
        }
        //printf("rank=%d tid=%d layer=%d nnz=%lu done done done\n", Env::rank, tid, start_layer, output->tiles[leader_rowgroup][0].in_subtiles.back().spmat->nnz_i);
    }
    
    
    
    
    
    //printf("Rank=%d, tid=%d Finish\n", Env::rank, tid);
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
    //printf("Rank=%d, tid=%d Run checksum\n", Env::rank, tid);
    
    auto finish = std::chrono::high_resolution_clock::now();
    //if(!tid) Env::exec_time = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish - start).count())/1e9;
    Env::execution_time[tid] = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish - start).count())/1e9;

    struct Tile<Weight>& C_tile = inputFeatures->tiles[Env::thread_rowgroup[tid]][0];
    std::shared_ptr<struct CSC<Weight>>& C_spmat = C_tile.spmat;
    int32_t count = validate_prediction1(C_spmat, trueCategories, C_tile.start_row);
    
    for(uint32_t rg: my_rowgrps) {
        for(auto tile: inputFeatures->tiles[rg][0].in_subtiles) {
            count += validate_prediction1(tile.spmat, trueCategories, tile.start_row);
            
            /*
            //printf("%d %d - rg=%d c0=%d c=%d\n", Env::rank, tid, rg, c0, c);
            std::shared_ptr<struct CSC<Weight>>& A_CSC = tile.spmat;
            const uint32_t start_row = tile.start_row;
            const uint64_t A_nnz   = A_CSC->nnz;
            const uint32_t A_nrows = A_CSC->nrows;
            const uint32_t A_ncols = A_CSC->ncols;
            const uint32_t* A_IA   = A_CSC->IA_blk->ptr;
            const uint32_t* A_JA   = A_CSC->JA_blk->ptr;
            const Weight*    A_A   = A_CSC->A_blk->ptr;
            
            std::vector<uint32_t> allCategories(A_nrows);
            uint32_t k = 0;
            for(uint32_t j = 0; j < A_ncols; j++) {
                for(uint32_t i = A_JA[j]; i < A_JA[j+1]; i++) {
                    allCategories[A_IA[i]] = 1;
                }
            }
            
            bool passed = true;
            uint32_t j = 0;
            for(uint32_t i = 0; i < A_nrows; i++) {
                if(trueCategories[start_row + i] != allCategories[i]) {
                    passed = false;
                    break;
                }
                if(trueCategories[start_row + i]) k++;
            }
            //printf("check: %d %d %d\n", Env::rank, tid, k);
            */
            /*
            if(passed) {
                Logging::print(Logging::LOG_LEVEL::INFO, "Challenge PASSED.\n");
            }
            else {
                Logging::print(Logging::LOG_LEVEL::ERROR, "Challenge FAILED.\n");
            }
            */
            //c0++;
            
        }
        //c++;
    }
    
    Env::counters[tid].checkcount = count;
    //printf("Rank=%d tid=%d\n", Env::rank, tid);
    pthread_barrier_wait(&Env::thread_barrier);
    Env::barrier();
    if(!tid) {
        int counts = 0;
        for(auto counter: Env::counters) {
            counts += (int) counter.checkcount;
        }
        printf("%d %d\n", tid, counts);
        int countss = 0;
        //Env::barrier();
        MPI_Allreduce(&counts, &countss, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        //Env::barrier();
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
        //printf("Rank=%d tid=%d sending exit signal \n", Env::rank, tid);
        for(int i = 0; i < Env::nranks; i++) {
            if(i != Env::rank) {
                int follower_rank = i;
                //int th_csc_handshake = -1;
                std::vector<uint32_t> csc_metadata = {0, 0, 0, 0, 0, 0, 0};
                MPI_Isend(csc_metadata.data(), csc_metadata.size(), MPI_UNSIGNED, follower_rank, follower_rank, Env::thread_communicators[tid], &request);
                //MPI_Isend(&th_csc_handshake, 1, MPI_INT, follower_rank, follower_rank, Env::thread_communicators[tid], &request);   
                requests.push_back(request);
                //printf("rank=%d tid=%d --> rank=%d add_to_idle_ranks MPI_Send zeros\n", Env::rank, tid, follower_rank);
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
        //printf("Rank=%d tid=%d waiting \n", Env::rank, tid);
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
        //printf("Rank=%d tid=%d <-- k=%d v=%d rowgroup=%d layer=%d nedges=%d start_row=%d height=%d width=%d\n", Env::rank, tid, 0, csc_metadata[0], csc_metadata[1], csc_metadata[2], csc_metadata[3], csc_metadata[4], csc_metadata[5],  csc_metadata[6]);
        
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
            //printf("0: Recv\n");
            std::shared_ptr<struct CSC<Weight>>& A_CSC = A_tile.spmat;
            A_CSC->Irecv(requests,source_rank, tid);
            //printf("1: Recv\n");
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
            requests.clear();
            requests.shrink_to_fit();
            //uint64_t sz = spaWeightVec[tid]->nitems;
            spaWeightVec[tid]->reallocate(A_tile.height);
            //printf("%d %d %d [%d %d %d] old=%lu\n", Env::rank, tid, A_CSC->nrows, A_tile.height, inputFeatures->tiles[leader_rowgroup][0].in_subtiles.back().height, output->tiles[leader_rowgroup][0].in_subtiles.back().height, sz);
        }
        done = msg_status;
    }
    return(done);
}

template<typename Weight>
bool Net<Weight>::add_to_my_follower_ranks(const uint32_t leader_rowgroup, const uint32_t start_layer, const std::vector<int32_t> my_threads, const int32_t tid) {
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
        //bool p = false;
        for(uint32_t k = 1; k < subtiles.size(); k++) {
            struct Tile<Weight> subtile = subtiles[k];
            //if(!subtile.nedges) p = true;
            //printf("Rank=%d tid=%d --> k=%d v=%d rowgroup=%d layer=%d nedges=%lu start_row=%d height=%d width=%d nthreads=%d\n", Env::rank, tid, subtile.rank, 1, leader_rowgroup, start_layer, subtile.nedges, subtile.start_row, subtile.height, subtile.width, nthreads_local);
            uint32_t msg_status = (subtile.nedges) ? 1 : 2;
            //if(msg_status == 1) {
                std::vector<uint32_t> csc_metadata = {msg_status, leader_rowgroup, start_layer, (uint32_t)subtile.nedges, subtile.start_row, subtile.height, subtile.width};
                MPI_Isend(csc_metadata.data(), csc_metadata.size(), MPI_UNSIGNED, subtile.rank, subtile.rank, Env::thread_communicators[tid], &request);        
                requests.push_back(request);
            ///}
            if(msg_status == 2) {
                /*
                for(uint32_t k = 0; k < subtiles.size(); k++) {
                    struct Tile<Weight> subtile = subtiles[k];
                    printf("Rank=%d tid=%d --> kk=%d v=%d rowgroup=%d layer=%d nedges=%lu start_row=%d height=%d width=%d\n", Env::rank, tid, subtile.rank, 1, leader_rowgroup, start_layer, subtile.nedges, subtile.start_row, subtile.height, subtile.width);                
                }
                */
                
                
                /*
                MPI_Win_lock(MPI_LOCK_EXCLUSIVE, window_host_rank, 0, Env::thread_windows[tid]);
                    some_val = 1;
                    MPI_Fetch_and_op(&some_val, &some_res, MPI_INT, window_host_rank, Env::nranks, MPI_SUM, Env::thread_windows[tid]);
                    MPI_Win_flush(window_host_rank, Env::thread_windows[tid]);
                    MPI_Fetch_and_op(&some_val, &some_res, MPI_INT, window_host_rank, subtile.rank, MPI_REPLACE, Env::thread_windows[tid]);
                    MPI_Win_flush(window_host_rank, Env::thread_windows[tid]);
                MPI_Win_unlock(window_host_rank, Env::thread_windows[tid]);
                */
                done = false;
            }
        }
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        requests.clear();
        requests.shrink_to_fit();
        
        /*
        if(p) {
            for(uint32_t k = 0; k < subtiles.size(); k++) {
                struct Tile<Weight> subtile = subtiles[k];
                printf("Rank=%d tid=%d --> kk=%d v=%d rowgroup=%d layer=%d nedges=%lu start_row=%d height=%d width=%d\n", Env::rank, tid, k, 1, leader_rowgroup, start_layer, subtile.nedges, subtile.start_row, subtile.height, subtile.width);                
            }
            std::exit(0);
        }
        */
        
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
        //uint64_t sz = Net::spaWeightVec[tid]->nitems;
        
        for(auto t: my_threads) {
            spaWeightVec[t]->reallocate(A_tile.height);   
        }
        //printf("rank=%d tid=%d old=%lu new=%lu h=[%d %d]%d\n", Env::rank, tid, sz, Net::spaWeightVec[tid]->nitems, inputFeatures->tiles[leader_rowgroup][0].height, output->tiles[leader_rowgroup][0].height, A_tile.height);
    }
    return(done);
}

template<typename Weight>
int32_t Net<Weight>::add_to_my_follower_threads(std::vector<int32_t>& my_threads, const uint32_t start_layer, const uint32_t ncols, const int32_t tid) {
    //printf("Rank=%d tid=%d 1. add_to_my_follower_threads>>>>>\n", Env::rank, tid);
    uint32_t num_threads = my_threads.size();
    if(!Env::follower_threads.empty()) {
        uint32_t new_height = (not(start_layer%2)) ? output->tiles[Env::thread_rowgroup[tid]][0].height
                                                   : inputFeatures->tiles[Env::thread_rowgroup[tid]][0].height;  
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
                spaWeightVec[t]->reallocate(new_height);
            }
            Env::init_num_threads(num_threads, tid, tid);
        }
        
        pthread_barrier_init(&Env::thread_barriers[tid], NULL, num_threads);
        
        pthread_cond_broadcast(&Env::thread_cond); 
        pthread_mutex_unlock(&Env::thread_mutex);
    }
    //printf("Rank=%d tid=%d 1. add_to_my_follower_threads<<<<<\n", Env::rank, tid);
    return(num_threads);
}


template<typename Weight>
void Net<Weight>::add_to_my_follower_threads(std::vector<int32_t>& my_threads, const uint32_t start_layer, const uint32_t ncols, const int32_t leader, const int32_t tid) {  
    //printf("Rank=%d tid=%d 2. add_to_my_follower_threads>>>>>\n", Env::rank, tid);
    uint32_t old_num_threads = 0;
    uint32_t new_num_threads = 0;
    uint32_t num_threads = 0;

    if(tid == leader) {
        uint32_t new_height = (not(start_layer%2)) ? output->tiles[Env::thread_rowgroup[tid]][0].height
                                                     : inputFeatures->tiles[Env::thread_rowgroup[tid]][0].height;  
        old_num_threads = my_threads.size();
        if(!Env::follower_threads.empty()) {
            pthread_mutex_lock(&Env::thread_mutex);  
            if(!Env::follower_threads.empty()) {
                num_threads = (my_threads.size() + Env::follower_threads.size());
                my_threads.insert(my_threads.end(), Env::follower_threads.begin(), Env::follower_threads.end());
                Env::follower_threads.erase(Env::follower_threads.begin(), Env::follower_threads.end());
                for(uint32_t i = 0; i < num_threads; i++) {
                    int32_t t = my_threads[i];
                    Env::threads[t].leader = tid;
                    Env::threads[t].rowgroup = Env::thread_rowgroup[tid];
                    Env::threads[t].start_layer = start_layer;
                    Env::threads[t].start_col = ((ncols/num_threads) * i);
                    Env::threads[t].end_col   = (i == (num_threads-1)) ? ncols : ((ncols/num_threads) * (i+1));
                    spaWeightVec[t]->reallocate(new_height);
                }
                pthread_barrier_destroy(&Env::thread_barriers[tid]);
                pthread_barrier_init(&Env::thread_barriers[tid], NULL, num_threads);
                
                new_num_threads = num_threads - old_num_threads;
                Env::increase_num_threads(new_num_threads, leader, tid);
            }
            pthread_cond_broadcast(&Env::thread_cond); 
            pthread_mutex_unlock(&Env::thread_mutex);
        }
    }
    //printf("Rank=%d tid=%d 2. add_to_my_follower_threads<<<<<\n", Env::rank, tid);
}

template<typename Weight>
bool Net<Weight>::add_to_idle_threads(std::vector<int32_t>& my_threads, const int32_t tid) {
    //printf("Rank=%d tid=%d  Env::follower_threads.size()=%lu add_to_idle_threads>>>>>\n", Env::rank, tid, Env::follower_threads.size());
    bool status = true;
    if(Env::follower_threads.size() != (uint32_t) Env::nthreads) {
        pthread_mutex_lock(&Env::thread_mutex);
            Env::follower_threads.push_back(tid);
            my_threads.erase(my_threads.begin(), my_threads.end());
            Env::threads[tid].leader = -1;
            if(Env::follower_threads.size() != (uint32_t) Env::nthreads) {
                pthread_cond_wait(&Env::thread_cond, &Env::thread_mutex); 
            }
            else {
                pthread_cond_broadcast(&Env::thread_cond);    
                status = false;
            }
        pthread_mutex_unlock(&Env::thread_mutex);
    }
    else {
        status = false;
    }
    //printf("Rank=%d tid=%d  add_to_idle_threads<<<<<<\n", Env::rank, tid);
    return(status);
}

template<typename Weight>
void Net<Weight>::hybrid(std::vector<int32_t>& my_threads, const uint32_t leader_rowgroup, const uint32_t start_layer, const int32_t leader, const int32_t tid) {
    double start_time = 0;
    uint64_t nnz   = 0;
    uint32_t nrows = inputFeatures->tile_height;
    uint32_t ncols = layers[0]->ncols;
    
    //struct Tile<Weight> A_tile;
    //struct Tile<Weight> B_tile;
    //struct Tile<Weight> C_tile;
    
    //if(tid != leader) {
    //    spaWeightVec[tid]->reallocate(inputFeatures->tiles[leader_rowgroup][0].height);   
        //printf("Rank=%d tid=%d nit=%lu\n", Env::rank, tid, spaWeightVec[tid]->nitems);
    //}
    
    std::shared_ptr<struct Data_Block<Weight>> s_spa = spaWeightVec[tid];  
    
    struct Env::thread_struct& thread_st = Env::threads[tid];
    bool old_thread = false;
    bool has_data_to_share = true;
    for(uint32_t l = start_layer; l < maxLayers; l++) {
        //if(tid == leader) 
        //printf("1.hybrid: rank=%d tid=%d layer=%d nthreads=%lu\n", Env::rank, tid, l, my_threads.size());
        start_time = Env::tic();
            add_to_my_follower_threads(my_threads, l, ncols, leader, tid);
            Env::decrease_num_threads(1, leader, tid);
            Env::init_num_threads(my_threads.size(), leader, tid);
        Env::hybrid_probe_time[tid] += Env::toc(start_time);     
            
        start_time = Env::tic();
        
            struct Tile<Weight>& A_tile = (not(l%2)) ? inputFeatures->tiles[leader_rowgroup][0]
                                                     : output->tiles[leader_rowgroup][0];
                                                         
            struct Tile<Weight>& C_tile = (not(l%2)) ? output->tiles[leader_rowgroup][0]
                                                     : inputFeatures->tiles[leader_rowgroup][0];                
        
        
            /*
            if(not(l%2)) {
                A_tile = inputFeatures->tiles[leader_rowgroup][0];
                C_tile = output->tiles[leader_rowgroup][0];
            }
            else {
                A_tile = output->tiles[leader_rowgroup][0];
                C_tile = inputFeatures->tiles[leader_rowgroup][0];
            }
            */

            std::shared_ptr<struct CSC<Weight>> A_CSC = A_tile.spmat;
            std::shared_ptr<struct CSC<Weight>> C_CSC = C_tile.spmat;
            struct Tile<Weight>& B_tile = layers[l]->tiles[0][0];
            std::shared_ptr<struct CSC<Weight>> B_CSC = B_tile.spmat;
            std::shared_ptr<struct Data_Block<Weight>> b_bias = biasWeightVecs[l];  
            
            start_time = Env::tic();
            if((tid == leader) and has_data_to_share) {
                has_data_to_share = add_to_my_follower_ranks(leader_rowgroup, l, my_threads, tid);
                /*
                if(tid == leader) {
                    
                    if(has_data_to_share) {
                        for(auto t: my_threads) {
                            //if(t != tid) {
                                spaWeightVec[t]->reallocate(A_tile.height);   
                            //}
                            //printf("h=%d tidh=%lu tid=t=%lu\n", A_tile.height, spaWeightVec[tid]->nitems, spaWeightVec[t]->nitems);
                        }
                        
                    }
                }
                */
            }
            pthread_barrier_wait(&Env::thread_barriers[leader]);
            Env::hybrid_probe_time[tid] += Env::toc(start_time);     
            
            
            //printf("Rank=%d tid=%d [%lu]SYMB\n", Env::rank, tid, my_threads.size());
            const uint32_t start_col = Env::threads[tid].start_col;// Env::follower_threads_info[leader][tid].start_col;
            const uint32_t end_col   = Env::threads[tid].end_col; //Env::follower_threads_info[leader][tid].end_col;
            /*
            {
                printf("Rank=%d tid=%d  nnz=%lu height=%d width=%d nit=%lu [%d %d]\n", Env::rank, tid, A_tile.nedges, A_tile.height, A_tile.width, s_spa->nitems, inputFeatures->tiles[leader_rowgroup][0].height, output->tiles[leader_rowgroup][0].height);
                const uint64_t A_nnz   = A_CSC->nnz;
                const uint32_t A_nrows = A_CSC->nrows;
                const uint32_t A_ncols = A_CSC->ncols;
                const uint32_t* A_IA   = A_CSC->IA_blk->ptr;
                const uint32_t* A_JA   = A_CSC->JA_blk->ptr;
                const Weight*    A_A   = A_CSC->A_blk->ptr;
                    
                const uint64_t B_nnz   = B_CSC->nnz;
                const uint32_t B_nrows = B_CSC->nrows;
                const uint32_t B_ncols = B_CSC->ncols;
                const uint32_t* B_IA   = B_CSC->IA_blk->ptr;
                const uint32_t* B_JA   = B_CSC->JA_blk->ptr;
                const Weight*    B_A   = B_CSC->A_blk->ptr;
                
                Weight*          s_A   = s_spa->ptr;
                
                printf("[%d %d %d]: [%lu %d %d] [%lu %d %d]\n", Env::rank, tid, tid==leader, A_nnz, A_nrows, A_ncols, B_nnz, B_nrows, B_ncols);
                
                uint32_t* IA = A_CSC->IA_blk->ptr;
                uint32_t* JA = A_CSC->JA_blk->ptr;
                Weight*    A = A_CSC->A_blk->ptr;
                
                
                uint64_t checksum = 0;
                double checkcount = 0;
                if(tid == leader) {
                    for(uint32_t j = 0; j < A_ncols; j++) { 
                            //std::cout << "j=" << j << "," << j << ": " << JA[j] << "--" << JA[j + 1] << ": " <<  JA[j + 1] - JA[j] << std::endl;    
                        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                            (void) IA[i];
                            (void) A[i];
                            checksum += A[i];
                            checkcount++;
                            s_A[IA[i]] = 1;
                            //std::cout << "    i=" << i << ",i=" << IA[i] <<  ",value=" << A[i] << std::endl;
                        }
                    }  
                }
                printf("rank=%d tid=%d checksum=%lu checkcount=%f\n", Env::rank, tid, checksum, checkcount);
                
                
            }
            */
            
            
            
            std::tie(thread_st.off_nnz, nrows, ncols) =  spmm_symb(A_CSC, B_CSC, s_spa, start_col, end_col, tid);
            //printf("Rank=%d tid=%d nnz=%lu nrows=%d ncols=%d\n", Env::rank, tid, thread_st.off_nnz, nrows, ncols);
            pthread_barrier_wait(&Env::thread_barriers[leader]);
        Env::spmm_symb_time[tid] += Env::toc(start_time);   

        start_time = Env::tic();
            nnz = Env::adjust_nnz(my_threads, leader, tid);
            C_CSC->reallocate(nnz, nrows, ncols, leader, tid);
        Env::memory_allocation_time[tid] += Env::toc(start_time);

        start_time = Env::tic();
            uint32_t off_col = 0;   
            pthread_barrier_wait(&Env::thread_barriers[leader]);
            spmm_real(A_CSC, B_CSC, C_CSC, s_spa, b_bias, start_col, end_col, off_col, thread_st.idx_nnz, tid);
            pthread_barrier_wait(&Env::thread_barriers[leader]);
            Env::adjust_displacement(my_threads, leader, tid);
            C_CSC->adjust(my_threads, leader, tid);	
        Env::spmm_real_time[tid] += Env::toc(start_time);

        start_time = Env::tic();
            pthread_barrier_wait(&Env::thread_barriers[leader]);
            A_CSC->repopulate(C_CSC, my_threads, leader, tid);
        Env::memory_allocation_time[tid] += Env::toc(start_time);
        if(!tid) Env::iteration++;
        //printf("2.hybrid: rank=%d tid=%d layer=%d nthreads=%lu\n", Env::rank, tid, l, my_threads.size());
    }
}
#endif 
