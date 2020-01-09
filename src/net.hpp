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
        
        PARALLELISM_TYPE parallelism_type;
        bool repartition = false;
        
        void printTimes();
        void printTimesExcel();
        void execute();
        void inferenceReLU(const int32_t tid);
        
        void data_x_model(const int32_t tid);
        void data_x_data(const int32_t tid);
        void hybrid_x_hybrid(const int32_t tid);
        void hybrid(std::vector<int32_t>& my_threads, const uint32_t rowgroup, const uint32_t layer, const int32_t leader, const int32_t tid);
        int32_t add_to_my_followers(std::vector<int32_t>& my_threads, const uint32_t start_layer, const uint32_t ncols, const int32_t tid);
        void add_to_my_followers(std::vector<int32_t>& my_threads, const uint32_t start_layer, const uint32_t ncols, const int32_t leader, const int32_t tid);
        bool check_for_idle_ranks(int32_t tid);
        bool add_to_idle_threads(std::vector<int32_t>& my_threads, const int32_t tid);
        
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
        IO::text_file_categories(categoryFile, trueCategories, inputFeatures->nrows);
    }
    else {
        IO::binary_file_categories(categoryFile, trueCategories, inputFeatures->nrows);
    }

    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Processing %d layer files (silent).\n", maxLayers); 
    maxLayers = 5;
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
    
    struct Tile<Weight> A_tile;
    struct Tile<Weight> B_tile;
    struct Tile<Weight> C_tile;
    std::shared_ptr<struct Data_Block<Weight>> s_spa = spaWeightVec[tid];
    struct Env::thread_struct& thread_st = Env::threads[tid];
    std::vector<int32_t> my_threads;
    my_threads.push_back(tid);
    auto start = std::chrono::high_resolution_clock::now();  
    if(Env::rank == 1) {
        sleep(15);
    }
    for (uint32_t l = 0; l < maxLayers; l++) {
        printf("rank=%d tid=%d layer=%d\n", Env::rank, tid, l);
        (void)check_for_idle_ranks(tid);
        if(add_to_my_followers(my_threads, l, ncols, tid) == 1) {
            
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
                std::shared_ptr<struct Data_Block<Weight>> b_bias = biasWeightVecs[l];  

            Env::spmm_symb_time[tid] += Env::toc(start_time);      
        
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
        //const int32_t leader = Env::follower_to_leader[tid];
        if(leader != -1) {
            uint32_t rowgroup = Env::threads[tid].rowgroup; //Env::follower_threads_info[leader][tid].rowgroup;
            uint32_t layer = Env::threads[tid].start_layer; //Env::follower_threads_info[leader][tid].layer;
            hybrid(my_threads, rowgroup, layer, leader, tid);
        }
    }
    
    
    
    while(true) {
        int window_host_rank = 0;
        int num_follower_ranks = 0;
        int some_val = 1;
        int some_res = 0;
        int some_cmp = 0;
        
        //MPI_Win_lock(MPI_LOCK_EXCLUSIVE, window_host_rank, 0, Env::thread_windows[tid]);    
        MPI_Win_lock_all(0, Env::thread_windows[tid]);
        //MPI_Win_sync(Env::thread_windows[tid]);
        //MPI_Win_flush(window_host_rank, Env::thread_windows[tid]);
        MPI_Fetch_and_op(&some_val, &some_res, MPI_INT, window_host_rank, Env::nranks, MPI_SUM, Env::thread_windows[tid]);
        //MPI_Win_flush(window_host_rank, Env::thread_windows[tid]);
        MPI_Win_flush_all(Env::thread_windows[tid]);
        num_follower_ranks = some_res+1;
        //printf("rank=%d tid=%d: outsi: num_follower_ranks=%d\n", Env::rank, tid, some_res);
        //some_val = num_follower_ranks;
        //MPI_Fetch_and_op(&some_val, &some_res, MPI_INT, window_host_rank, Env::nranks, MPI_REPLACE, Env::thread_windows[tid]);
        some_val = 1;
        MPI_Fetch_and_op(&some_val, &some_res, MPI_INT, window_host_rank, Env::rank, MPI_REPLACE, Env::thread_windows[tid]);
        MPI_Win_flush(window_host_rank, Env::thread_windows[tid]);
        //MPI_Win_flush_all(Env::thread_windows[tid]);
        //MPI_Win_sync(Env::thread_windows[tid]);
        //MPI_Win_flush(window_host_rank, Env::thread_windows[tid]);
        //MPI_Win_flush_all(Env::thread_windows[tid]);
        //MPI_Win_sync(Env::thread_windows[tid]);
        //MPI_Win_flush(Env::rank, Env::thread_windows[tid]);
        //MPI_Win_unlock(window_host_rank, Env::thread_windows[tid]);
        MPI_Win_unlock_all(Env::thread_windows[tid]);
        printf("rank=%d tid=%d While: num=%d\n", Env::rank, tid, num_follower_ranks);
        
        if(num_follower_ranks == Env::nranks) {
            MPI_Request request;
            std::vector<MPI_Request> requests;
            printf("rank=%d tid=%d Exiting\n", Env::rank, tid);
            for(int i = 0; i < Env::nranks; i++) {
                if(i != Env::rank) {
                    int follower_rank = i;
                    int th_csc_handshake = -1;
                    
                    MPI_Isend(&th_csc_handshake, 1, MPI_INT, follower_rank, follower_rank, Env::thread_communicators[tid], &request);   
                    requests.push_back(request);
                    printf("rank=%d tid=%d While exit: --> rank=%d MP_Send handshake (%d)\n", Env::rank, tid, follower_rank, th_csc_handshake);
                }
            }
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
            requests.clear();
            requests.shrink_to_fit();
            break;
        }
        else {
            MPI_Status status;

            int th_csc_handshake;
            int source_rank;
            int source_tag;
            //while(true) {
            do{    
                int flag = 0;
                while(!flag) {
                    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, Env::thread_communicators[tid], &flag, &status);
                }
                //printf("source=%d tag=%d\n", status.MPI_SOURCE, status.MPI_TAG);
                source_rank = status.MPI_SOURCE;
                source_tag = status.MPI_TAG;
            } while(source_tag != Env::rank);
            
            
            MPI_Recv(&th_csc_handshake, 1, MPI_INT, source_rank , source_tag, Env::thread_communicators[tid], &status);
            printf("rank=%d tid=%d While wait: <--- rank=%d MPI_Recv handshake (%d)\n", Env::rank, tid, source_rank, th_csc_handshake);
            //std::this_thread::sleep_for(std::chrono::nanoseconds(100));

            if(th_csc_handshake == -1) {
                break;
            }
        }
    
    }
    
    
    
    /*
    
    //printf("Rank=%d, tid=%d I'm done, ask for something? \n", Env::rank, tid);
    
    //if(!tid) {
        printf("-1.Rank=%d, tid=%d waiting\n", Env::rank, tid);
        int origin_value = -1;
        int result_value = 0;
        int compare_value = 0;
        MPI_Status status;
        MPI_Request request;
        std::vector<MPI_Request> requests;
        int target_rank = 0;
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, Env::rank, 0, Env::thread_windows[tid]);    
        MPI_Compare_and_swap(&origin_value, &compare_value, &result_value, MPI_INT, Env::rank, Env::nranks, Env::thread_windows[tid]);
        printf("rank=%d tid=%d result_value=%d %d\n", Env::rank, tid, result_value, *(Env::idle_threads[tid] + Env::nranks));
        MPI_Win_sync(Env::thread_windows[tid]);
        MPI_Win_flush(Env::rank, Env::thread_windows[tid]);
        MPI_Win_unlock(Env::rank, Env::thread_windows[tid]);
            //MPI_Fetch_and_op(&idle_status, &counter, MPI_INT, Env::rank, Env::nranks, MPI_REPLACE, Env::thread_windows[tid]);
            //MPI_Put();
            //int value 
            //MPI_Fetch_and_op(&idle_status, &counter, MPI_INT, Env::rank, Env::nranks, MPI_REPLACE, Env::thread_windows[tid]);
            //int& k = *(Env::idle_threads[tid] + Env::nranks);
    
        if(result_value != 0) {
            for(int i = 1; i < Env::nranks; i++) {
                int target_rank_index = (Env::rank - i + Env::nranks) % Env::nranks;
                int target_rank_public = *(Env::idle_threads[tid] + target_rank_index) - 1;
                int target_rank_private = (Env::rank - i + Env::nranks) % Env::nranks;
                if(target_rank_public == target_rank_private) {
                    printf("Should communicate rank=%d, tid=%d, %d %d %d\n", Env::rank, tid, result_value, target_rank_public, target_rank_private);
                    int thread_no_op = -1;
                    target_rank = target_rank_private;
                    MPI_Isend(&thread_no_op, 1, MPI_INT, target_rank, tid, Env::thread_communicators[tid], &request);
                    requests.push_back(request);
                }
            }
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
            requests.clear();
            requests.shrink_to_fit();
        }

        
       
        
      //  printf("Should not communicate rank=%d, tid=%d, %d %d\n", Env::rank, tid, result_value, *(Env::idle_threads[tid] + Env::nranks));

        

    //}
    
    //printf("0.Rank=%d, tid=%d waiting\n", Env::rank, tid);
    origin_value = 0;
    result_value = 0;
    
    
    for(int i = 1; i < Env::nranks; i++) {
        int target_rank = (Env::rank + i) % Env::nranks;
        printf("0.Rank=%d tid=%d target_rank=%d i = %d\n", Env::rank, tid,  target_rank, i);
        int target_disp = Env::rank;
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target_rank, 0, Env::thread_windows[tid]);
        MPI_Win_sync(Env::thread_windows[tid]);
        MPI_Fetch_and_op(&origin_value, &result_value, MPI_INT, target_rank, Env::nranks, MPI_NO_OP, Env::thread_windows[tid]);
        MPI_Win_sync(Env::thread_windows[tid]);
        MPI_Win_unlock(target_rank, Env::thread_windows[tid]);  
        printf("0.Rank=%d tid=%d ret=%d target_rank=%d\n", Env::rank, tid, result_value, target_rank);
        if(result_value != -1) {
            int plus_one_rank = 1;
            int my_rank_plus_one = Env::rank+1;
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target_rank, 0, Env::thread_windows[tid]);
            MPI_Fetch_and_op(&plus_one_rank, &result_value, MPI_INT, target_rank, Env::nranks, MPI_SUM, Env::thread_windows[tid]);    
            printf("1.Rank=%d tid=%d ret=%d\n", Env::rank, tid, result_value);
            MPI_Fetch_and_op(&my_rank_plus_one,   &result_value, MPI_INT, target_rank, target_disp, MPI_REPLACE, Env::thread_windows[tid]);
            printf("2.Rank=%d tid=%d ret=%d\n", Env::rank, tid, result_value);
            MPI_Win_flush(target_rank, Env::thread_windows[tid]);
            MPI_Win_sync(Env::thread_windows[tid]);
            MPI_Win_unlock(target_rank, Env::thread_windows[tid]);    
            
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target_rank, 0, Env::thread_windows[tid]);
            //MPI_Win_fence((MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE), Env::thread_windows[tid]);
            MPI_Get(&origin_value, 1, MPI_INT, target_rank, Env::nranks, 1, MPI_INT, Env::thread_windows[tid]);
            printf("3.Rank=%d tid=%d ret=%d\n", Env::rank, tid, result_value);
            //MPI_Win_fence(MPI_MODE_NOSUCCEED, Env::thread_windows[tid]);
            MPI_Win_unlock(target_rank, Env::thread_windows[tid]);    
            
            int flag = 0;
            int source_rank = target_rank;
            while(!flag) {
                MPI_Iprobe(source_rank, tid, Env::thread_communicators[tid], &flag, &status);
            }
            int th_csc_handshake;
            MPI_Recv(&th_csc_handshake, 1, MPI_INT, source_rank , tid, Env::thread_communicators[tid], &status);

            printf("rank=%d tid=%d <--- %d: MPI_Recv handshake is done (%d)\n", Env::rank, tid, source_rank, th_csc_handshake);
            
            //MPI_Get_count( &status, MPI_INT, &count );
            //if(count != 1)
            //{
            //    errs++;
            //}
            
        }
        //else {
          //  MPI_Win_unlock(target_rank, Env::thread_windows[tid]);    
        //}
        //MPI_Win_sync(Env::thread_windows[tid]);
        
        
    }
    */
    
    
    printf("Rank=%d, tid=%d Done\n", Env::rank, tid);
    pthread_barrier_wait(&Env::thread_barrier);
    Env::barrier();
    
    if (!Env::rank and !tid) {
        printf("%d:\n", Env::rank);
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
    /*
    if (Env::rank==1 and !tid) {
        printf("%d:\n", Env::rank);
        for(int j = 0; j < Env::nthreads; j++) {
            printf("%d: ", j);
            for(int i =0;i < Env::nranks+1; i++) {
                printf("%d ", *(Env::idle_threads[j] + i));
            }
            printf("\n");
        }
        printf("\n");
    }
    pthread_barrier_wait(&Env::thread_barrier);
    Env::barrier();
    */
    /*
    if (Env::rank==2 and !tid) {
        printf("%d:\n", Env::rank);
        for(int j = 0; j < Env::nthreads; j++) {
            printf("%d: ", j);
            for(int i =0;i < Env::nranks+1; i++) {
                printf("%d ", *(Env::idle_threads[j] + i));
            }
            printf("\n");
        }
        printf("\n");
    }
    pthread_barrier_wait(&Env::thread_barrier);
    Env::barrier();
    if (Env::rank==3 and !tid) {
        printf("%d:\n", Env::rank);
        for(int j = 0; j < Env::nthreads; j++) {
            printf("%d: ", j);
            for(int i =0;i < Env::nranks+1; i++) {
                printf("%d ", *(Env::idle_threads[j] + i));
            }
            printf("\n");
        }
        printf("\n");
    }
    pthread_barrier_wait(&Env::thread_barrier);
    Env::barrier();
    */
    
    //*/
    
    
    /*
    for(uint32_t i = 0; i < Env::nranks-1; i++) {
        int32_t r = (Env::rank + 1 + i) % Env::nranks;
        if(r != Env::rank) {
            char req = 1;
            MPI_Send(req, 1, MPI_CHAR, r, Env::rank, MPI_COMM_WORLD);
        }
    }
    */
    
    auto finish = std::chrono::high_resolution_clock::now();
    //if(!tid) Env::exec_time = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish - start).count())/1e9;
    Env::execution_time[tid] = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish - start).count())/1e9;

    C_tile = inputFeatures->tiles[Env::thread_rowgroup[tid]][0];
    auto& C_spmat = C_tile.spmat;
    bool passed = validate_prediction(C_spmat, trueCategories, C_tile.start_row, tid);
    if(passed) {
        if(!tid) Logging::print(Logging::LOG_LEVEL::INFO, "Challenge PASSED.\n");
    }
    else {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Challenge FAILED.\n");
    }
    
    pthread_barrier_wait(&Env::thread_barrier);        
    Env::barrier();
}

template<typename Weight>
bool Net<Weight>::check_for_idle_ranks(int32_t tid) {
    //int ranks_sum = *(Env::idle_threads[tid] + Env::nranks);
    //if(ranks_sum) {
        
        int window_host_rank = 0;
        int num_follower_ranks = 0;
        int some_val = 0;
        int some_res = 0;
        int some_cmp = 0;
        /*
        if(window_host_rank == Env::rank) {
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, window_host_rank, 0, Env::thread_windows[tid]);
            MPI_Win_sync(Env::thread_windows[tid]);
            MPI_Win_unlock(window_host_rank, Env::thread_windows[tid]);
        }
        */
        //MPI_Win_lock(MPI_LOCK_EXCLUSIVE, window_host_rank, 0, Env::thread_windows[tid]);
        MPI_Win_lock_all(0, Env::thread_windows[tid]);
        //MPI_Win_sync(Env::thread_windows[tid]);
        //MPI_Win_flush(window_host_rank, Env::thread_windows[tid]);
        MPI_Fetch_and_op(&some_val, &some_res, MPI_INT, window_host_rank, Env::nranks, MPI_NO_OP, Env::thread_windows[tid]);
        //MPI_Win_flush(window_host_rank, Env::thread_windows[tid]);
        MPI_Win_flush_all(Env::thread_windows[tid]);
        num_follower_ranks = some_res;
        //printf("rank=%d tid=%d: queue: num_follower_ranks=%d\n", Env::rank, tid, num_follower_ranks);
        if(num_follower_ranks) {
            printf("rank=%d tid=%d check_for_idle_ranks: num=%d\n", Env::rank, tid, num_follower_ranks);
            for(int i = 0; i < Env::nranks; i++) {
                if(i != Env::rank) {
                    some_val = 0;
                    some_res = 0;
                    some_cmp = 1;
                    MPI_Compare_and_swap(&some_val, &some_cmp, &some_res, MPI_INT, window_host_rank, i, Env::thread_windows[tid]);
                    //MPI_Win_flush(window_host_rank, Env::thread_windows[tid]);
                    MPI_Win_flush_all(Env::thread_windows[tid]);
                    
                    //printf("i=%d %d\n", i, some_res);
                    if(some_res){
                        some_val = num_follower_ranks - 1;
                        MPI_Fetch_and_op(&some_val, &some_res, MPI_INT, window_host_rank, Env::nranks, MPI_REPLACE, Env::thread_windows[tid]);
                        //MPI_Win_flush(window_host_rank, Env::thread_windows[tid]);
                        MPI_Win_flush_all(Env::thread_windows[tid]);
                        
                        some_val = 0;
                        MPI_Fetch_and_op(&some_val, &some_res, MPI_INT, window_host_rank, i, MPI_REPLACE, Env::thread_windows[tid]);
                        //MPI_Win_flush(window_host_rank, Env::thread_windows[tid]);
                        MPI_Win_flush_all(Env::thread_windows[tid]);
                        
                        int follower_rank = i;
                        int th_csc_handshake = 1;
                        //printf("rank=%d tid=%d --> %d: MP_Send handshake started\n", Env::rank, tid, follower_rank);
                        MPI_Send(&th_csc_handshake, 1, MPI_INT, follower_rank, follower_rank, Env::thread_communicators[tid]);   
                        MPI_Fetch_and_op(&some_val, &some_res, MPI_INT, window_host_rank, Env::nranks, MPI_NO_OP, Env::thread_windows[tid]);
                        MPI_Win_flush_all(Env::thread_windows[tid]);
                        printf("rank=%d tid=%d check_for_idle_ranks: --> rank=%d MP_Send handshake (%d/%d)\n", Env::rank, tid, follower_rank, th_csc_handshake, some_res);
                        //break;
                    }
                }   
            }
        }
        //MPI_Win_flush_all(Env::thread_windows[tid]);
        //MPI_Win_sync(Env::thread_windows[tid]);

        //MPI_Win_unlock(window_host_rank, Env::thread_windows[tid]);
        MPI_Win_unlock_all(Env::thread_windows[tid]);
        /*
        //MPI_Win_flush(Env::rank, Env::thread_windows[tid]);
        //MPI_Win_sync(Env::thread_windows[tid]);
        //printf(">>Rank=%d tid=%d ret=%d\n", Env::rank, tid, return_value);
        int target_rank = -1;
       //return_value = *(Env::idle_threads[tid] + Env::nranks);
       //int* thread_window = Env::idle_threads[tid];
       //int& idle_rank_is_waiting = thread_window[Env::nranks];
       //if(return_value) {
       if(idle_rank_is_waiting) {
           printf("Rank=%d, waitingN=%d\n", Env::rank, idle_rank_is_waiting);
            for(int i = 1; i < Env::nranks; i++) {
                int target_rank_index = (Env::rank - i + Env::nranks) % Env::nranks;
                //int target_rank_public = *(Env::idle_threads[tid] + target_rank_index) - 1;
                int& target_rank_public = thread_window[target_rank_index];
                //
                int target_rank_private = (Env::rank - i + Env::nranks) % Env::nranks;
                printf("Rank=%d tid=%d i=%d pub=%d priv=%d\n", Env::rank, tid, i, target_rank_public, target_rank_private);
                if((target_rank_public-1) == target_rank_private) {
                    
                    target_rank = target_rank_private;
                    printf("Target rank = %d\n", target_rank);
                    
                    // *(Env::idle_threads[tid] + target_rank_index) = 0;
                    // *(Env::idle_threads[tid] + Env::nranks) = *(Env::idle_threads[tid] + Env::nranks) - 1;
                    int minus_one_rank = idle_rank_is_waiting - 1;
                    int no_rank = 0;
                    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, Env::rank, 0, Env::thread_windows[tid]);
                    
                    MPI_Fetch_and_op(&minus_one_rank, &return_value, MPI_INT, Env::rank, Env::nranks, MPI_REPLACE, Env::thread_windows[tid]);
                    
                    MPI_Fetch_and_op(&no_rank, &return_value, MPI_INT, Env::rank, target_rank_index, MPI_REPLACE, Env::thread_windows[tid]);
                    MPI_Win_flush(Env::rank, Env::thread_windows[tid]);
                    MPI_Win_sync(Env::thread_windows[tid]);
                    //target_rank_public = 0;
                    //idle_rank_is_waiting--; 
                    MPI_Win_unlock(Env::rank, Env::thread_windows[tid]);            
                    break;
                }
            }
            if(target_rank >= 0) {
                
                int th_csc_handshake = 1;
                MPI_Send(&th_csc_handshake, 1, MPI_INT, target_rank, tid, Env::thread_communicators[tid]);
                printf("rank=%d tid=%d --> %d: MPI_Send handshake is done\n", Env::rank, tid, target_rank);
                //MPI_Send(&data, 1, MPI_INT, target_rank, tid, thread_communicators[tid]);
            }

            
            
        }
        */
        
    //}
    return(false);
}

template<typename Weight>
int32_t Net<Weight>::add_to_my_followers(std::vector<int32_t>& my_threads, const uint32_t start_layer, const uint32_t ncols, const int32_t tid) {
    uint32_t num_threads = my_threads.size();
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
        }
        
        pthread_barrier_init(&Env::thread_barriers[tid], NULL, num_threads);
        
        pthread_cond_broadcast(&Env::thread_cond); 
        pthread_mutex_unlock(&Env::thread_mutex);
    }
    return(num_threads);
}


template<typename Weight>
void Net<Weight>::add_to_my_followers(std::vector<int32_t>& my_threads, const uint32_t start_layer, const uint32_t ncols, const int32_t leader, const int32_t tid) {  
    uint32_t old_num_threads = 0;
    uint32_t new_num_threads = 0;
    uint32_t num_threads = 0;
    if(tid == leader) {
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
}

template<typename Weight>
bool Net<Weight>::add_to_idle_threads(std::vector<int32_t>& my_threads, const int32_t tid) {
    bool status = true;
    if(Env::follower_threads.size() != (uint32_t) Env::nthreads) {
        pthread_mutex_lock(&Env::thread_mutex);
            Env::follower_threads.push_back(tid);
            my_threads.erase(my_threads.begin(), my_threads.end());
            Env::threads[tid].leader = -1;
            //Env::follower_to_leader[tid] = -1;
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
    return(status);
}

template<typename Weight>
void Net<Weight>::hybrid(std::vector<int32_t>& my_threads, const uint32_t rowgroup, const uint32_t layer, const int32_t leader, const int32_t tid) {
    double start_time = 0;
    uint64_t nnz   = 0;
    uint32_t nrows = inputFeatures->tile_height;
    uint32_t ncols = layers[0]->ncols;
    
    struct Tile<Weight> A_tile;
    struct Tile<Weight> B_tile;
    struct Tile<Weight> C_tile;
    
    std::shared_ptr<struct Data_Block<Weight>> s_spa = spaWeightVec[tid];  
    struct Env::thread_struct& thread_st = Env::threads[tid];
    bool old_thread = false;
    for(uint32_t l = layer; l < maxLayers; l++) {
        start_time = Env::tic();
            add_to_my_followers(my_threads, l, ncols, leader, tid);
            Env::decrease_num_threads(1, leader, tid);
            Env::init_num_threads(my_threads.size(), leader, tid);
        Env::hybrid_probe_time[tid] += Env::toc(start_time);     
            
        start_time = Env::tic();
            if(not(l%2)) {
                A_tile = inputFeatures->tiles[rowgroup][0];
                C_tile = output->tiles[rowgroup][0];
            }
            else {
                A_tile = output->tiles[rowgroup][0];
                C_tile = inputFeatures->tiles[rowgroup][0];
            }

            std::shared_ptr<struct CSC<Weight>> A_CSC = A_tile.spmat;
            std::shared_ptr<struct CSC<Weight>> C_CSC = C_tile.spmat;
            B_tile = layers[l]->tiles[0][0];
            std::shared_ptr<struct CSC<Weight>> B_CSC = B_tile.spmat;
            std::shared_ptr<struct Data_Block<Weight>> b_bias = biasWeightVecs[l];  

            const uint32_t start_col = Env::threads[tid].start_col;// Env::follower_threads_info[leader][tid].start_col;
            const uint32_t end_col   = Env::threads[tid].end_col; //Env::follower_threads_info[leader][tid].end_col;
            std::tie(thread_st.off_nnz, nrows, ncols) =  spmm_symb(A_CSC, B_CSC, s_spa, start_col, end_col, tid);
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
    }
}
#endif 
