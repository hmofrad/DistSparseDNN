/*
 * net.hpp: Neural network base class 
 * (c) Mohammad Hasanzadeh Mofrad, 2020
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */
 
#ifndef NET_HPP
#define NET_HPP

#include "triple.hpp"
#include "tiling.hpp"
#include "spops.hpp"
#include <deque>
#include "hashers.hpp"

/* Input x layers */
enum PARALLELISM_TYPE {_DATA_X_DATA_, _DATA_X_MODEL_, DATA_THEN_MODEL, _MANAGER_X_WORKER_, _WORK_X_STEALING_, _P_SIZE_};
const char* PARALLELISM_TYPES[] = {"_DATA_X_DATA_", "_DATA_X_MODEL_", "DATA_THEN_MODEL", "_MANAGER_X_WORKER_", "_WORK_X_STEALING_"};

enum SCHEDULING_TYPE {_EARLIEST_FIRST_, _SLOWER_FIRST_, _FASTER_FIRST_, _NONE_};
const char* SCHEDULING_TYPES[] = {"_EARLIEST_FIRST_", "_SLOWER_FIRST_", "_FASTER_FIRST_", "_NONE_"};

template<typename Weight>
class Net {
    public:
        Net() {};
        ~Net() {};
        
        Net(const uint32_t input_ninstanses_, const uint32_t input_nfeatures_, const std::string input_file,
            const uint32_t ncategories, const VALUE_TYPE category_type_, const  std::string category_file, 
            const uint32_t nneurons_, const uint32_t nmax_layers_, const  std::vector<std::string> layer_files,
            const Weight bias_value, const VALUE_TYPE bias_type, const std::vector<std::string> bias_files,
            Weight(*noop_function_)(Weight),
            Weight(*activation_function_)(Weight),
            const std::string classifier_,
            const FILE_TYPE file_type = FILE_TYPE::_BINARY_,
            const COMPRESSED_FORMAT input_compression_type_ = COMPRESSED_FORMAT::_CSC_,
            const COMPRESSED_FORMAT layer_compression_type_ = COMPRESSED_FORMAT::_CSC_,
            const PARALLELISM_TYPE parallelism_type_  = PARALLELISM_TYPE::DATA_THEN_MODEL,
            const HASHING_TYPE hashing_type_ = HASHING_TYPE::_BOTH_);

        std::unique_ptr<struct Tiling<Weight>> input_features = nullptr;
        std::vector<uint32_t> true_categories;
        std::vector<std::unique_ptr<struct Tiling<Weight>>> layers;
        
        std::vector<std::shared_ptr<struct Data_Block<Weight>>> bias_vectors;
        std::vector<std::shared_ptr<struct Data_Block<Weight>>> spa_vectors;
        
        std::unique_ptr<struct Tiling<Weight>> output = nullptr;
        
        uint32_t input_ninstanses = 0;
        uint32_t input_nfeatures = 0;
        uint64_t input_nnzs = 0;
        uint32_t ncategories = 0;
        uint32_t nneurons = 0;
        uint32_t nmax_layers = 0;
        
        VALUE_TYPE category_type = VALUE_TYPE::_NONZERO_INSTANCES_ONLY_;
        
        Weight (*noop_function)(Weight);
        Weight (*activation_function)(Weight);
        std::string classifier;
        
        uint32_t predicted_nistances;
        

        uint32_t split_factor = 8;
        bool numa_queues = true;
        uint32_t schduling_threshold = 4;
        
        //COMPRESSED_FORMAT compression_type       = COMPRESSED_FORMAT::_CSC_;
        COMPRESSED_FORMAT input_compression_type = COMPRESSED_FORMAT::_CSC_;
        COMPRESSED_FORMAT layer_compression_type = COMPRESSED_FORMAT::_CSC_;
        //MULTIPLICATION_TYPE multiplication_type  = MULTIPLICATION_TYPE::_COMPRESSED_X_COMPRESSED_;
        PARALLELISM_TYPE parallelism_type        = PARALLELISM_TYPE::DATA_THEN_MODEL;
        SCHEDULING_TYPE scheduling_type          = _SLOWER_FIRST_;
        float recruiting_ratio = .3;
        HASHING_TYPE hashing_type = HASHING_TYPE::_BOTH_; 
        std::vector<std::shared_ptr<struct TwoDHasher>> hashers;
        std::shared_ptr<struct TwoDHasher> input_hasher;
        std::shared_ptr<struct TwoDHasher> layer_hasher;

        
        void execute();
        void inference(const int32_t tid);
        void printTimes();
        void data_x_model(const int32_t tid);
        void data_x_data(const int32_t tid);
        void hybrid_x_hybrid(const int32_t tid);
        void manager_x_worker(const int32_t tid);
        void work_x_stealing(const int32_t tid);
        uint32_t hybrid_x_data(std::deque<int32_t>& leader_owned_threads, const int32_t my_rowgroup, const int32_t tid);
        void hybrid_x_model(std::deque<int32_t>& leader_owned_threads, const uint32_t leader_rowgroup, const uint32_t leader_start_layer, const uint32_t leader_current_layer, const int32_t leader_tid, const int32_t tid);
        bool add_to_idle_threads(std::deque<int32_t>& leader_owned_threads, const int32_t tid);
        bool add_to_my_follower_threads(std::deque<int32_t>& leader_owned_threads, const uint32_t leader_rowgroup, const uint32_t leader_start_layer, const uint32_t leader_current_layer, const uint32_t nrows, const uint32_t ncols, const int32_t leader, const int32_t tid);
        bool thread_scheduling(std::deque<int32_t>& leader_owned_threads, const uint32_t leader_rowgroup, std::deque<int32_t>& follower_threads, int32_t socket_id, const uint32_t leader_start_layer, const uint32_t leader_current_layer, const uint32_t nrows, const uint32_t ncols, const int32_t leader, const int32_t tid);
};

template<typename Weight>
Net<Weight>::Net(const uint32_t input_ninstanses_, const uint32_t input_nfeatures_, const std::string input_file,
                 const uint32_t ncategories_, const VALUE_TYPE category_type_, const std::string category_file, 
                 const uint32_t nneurons_, const uint32_t nmax_layers_, const std::vector<std::string> layer_files,
                 const Weight bias_value, const VALUE_TYPE bias_type, const std::vector<std::string> bias_files,
                 Weight(*noop_function_)(Weight), Weight(*activation_function_)(Weight), const std::string classifier_,
                 const FILE_TYPE file_type, 
                 const COMPRESSED_FORMAT input_compression_type_, const COMPRESSED_FORMAT layer_compression_type_, 
                 const PARALLELISM_TYPE parallelism_type_,  const HASHING_TYPE hashing_type_)
                     : input_ninstanses(input_ninstanses_), input_nfeatures(input_nfeatures_), ncategories(ncategories_),
                       nneurons(nneurons_), nmax_layers(nmax_layers_), category_type(category_type_),
                       noop_function(noop_function_), activation_function(activation_function_), classifier(classifier_),
                       input_compression_type(input_compression_type_), layer_compression_type(layer_compression_type_), parallelism_type(parallelism_type_),
                       hashing_type(hashing_type_) {
                           
    auto start = std::chrono::high_resolution_clock::now();
    input_ninstanses+=2;
    input_ninstanses += (input_ninstanses % Env::nthreads) ? (Env::nthreads - (input_ninstanses % Env::nthreads)) : 0; 
    input_nfeatures+=2;
    input_nfeatures += (input_nfeatures % Env::nthreads) ? (Env::nthreads - (input_nfeatures % Env::nthreads)) : 0; 
    nneurons+=2;
    nneurons += (nneurons % Env::nthreads) ? (Env::nthreads - (nneurons % Env::nthreads)) : 0; 
    scheduling_type = (parallelism_type != PARALLELISM_TYPE::DATA_THEN_MODEL) ? SCHEDULING_TYPE::_NONE_ : scheduling_type;
    hashers.push_back(std::move(std::make_shared<struct TwoDHasher>(hashing_type, true, input_ninstanses, input_nfeatures, 1, 1)));
        
    if(not(((input_compression_type == COMPRESSED_FORMAT::_UDC_) and (layer_compression_type == COMPRESSED_FORMAT::_UDC_)) or // Dense x Dense
           ((input_compression_type == COMPRESSED_FORMAT::_UDC_) and (layer_compression_type == COMPRESSED_FORMAT::_CSC_)) or // Dense x Compressed
           ((input_compression_type == COMPRESSED_FORMAT::_CSC_) and (layer_compression_type == COMPRESSED_FORMAT::_CSC_)) or // Compressed x Compressed (CSC)
           ((input_compression_type == COMPRESSED_FORMAT::_CSR_) and (layer_compression_type == COMPRESSED_FORMAT::_CSR_)))){ // Compressed x Compressed (CSR)
        Logging::print(Logging::LOG_LEVEL::ERROR, "[%sx%s] multiplication not implemented\n", COMPRESSED_FORMATS[input_compression_type], COMPRESSED_FORMATS[layer_compression_type]);
        std::exit(Env::finalize());
    }
        
    
    input_nnzs = IO::get_nnzs<Weight>(input_file, file_type, hashers[0], input_ninstanses);
    if(parallelism_type == PARALLELISM_TYPE::_DATA_X_MODEL_) {
        input_features = std::move(std::make_unique<Tiling<Weight>>(Env::nranks, Env::nranks, 1, Env::nranks, 
                                                                    input_nnzs, input_ninstanses, input_nfeatures, 
                                                                    input_file, file_type, 
                                                                    TILING_TYPE::_1D_ROW_, input_compression_type, hashers[0]));
    }
    else if((parallelism_type == PARALLELISM_TYPE::_MANAGER_X_WORKER_) or (parallelism_type == PARALLELISM_TYPE::_WORK_X_STEALING_)) {
        input_features = std::move(std::make_unique<Tiling<Weight>>(Env::nranks * Env::nthreads * split_factor, Env::nranks * Env::nthreads * split_factor, 1, Env::nranks,
                                                                    Env::nthreads, Env::nranks * Env::nthreads, 
                                                                    input_nnzs, input_ninstanses, input_nfeatures, 
                                                                    input_file, file_type, 
                                                                    TILING_TYPE::_1D_ROW_, input_compression_type, hashers[0]));
       Env::threads_rowgroups = input_features->set_threads_indices();
       Env::rank_rowgroups = input_features->set_rank_indices();  
    }
    else {
        input_features = std::move(std::make_unique<Tiling<Weight>>(Env::nranks * Env::nthreads, Env::nranks * Env::nthreads, 1, Env::nranks,
                                                                    Env::nthreads, Env::nranks * Env::nthreads, 
                                                                    input_nnzs, input_ninstanses, input_nfeatures, 
                                                                    input_file, file_type, 
                                                                    TILING_TYPE::_1D_ROW_, input_compression_type, hashers[0]));
        Env::thread_rowgroup = input_features->set_thread_index();                                                           
    }
    
    input_ninstanses = input_features->nrows;
    input_nfeatures  = input_features->ncols;
    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Processing the category files for %d neurons and %d layers.\n", nneurons, nmax_layers); 
    predicted_nistances = IO::read_file_iv<uint32_t>(category_file, file_type, hashers[0], true, category_type, true_categories, input_features->nrows);

    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Processing %d layer files (silent).\n", nmax_layers); 
    //nmax_layers = 2;
    layers.resize(nmax_layers);
    bias_vectors.resize(nmax_layers);
    
    uint64_t layer_nnzs = 0;
    uint32_t layer_nrows = 0, layer_ncols = 0;
    for(uint32_t i = 0; i < nmax_layers; i++) {
        if(i == 0) { layer_nrows = input_nfeatures; layer_ncols = nneurons; }
        else if(i < nmax_layers-1) { layer_nrows = nneurons; layer_ncols = nneurons; }
        else { layer_nrows = nneurons; layer_ncols = ncategories ? ncategories : nneurons; }
        std::string layer_file = layer_files[i];
        hashers.push_back(std::move(std::make_shared<struct TwoDHasher>(hashing_type, false, layer_nrows, layer_ncols, 1, 1)));
        layer_nnzs = IO::get_nnzs<Weight>(layer_file, file_type, hashers[i+1], layer_nrows);
        layers[i] = std::move(std::make_unique<Tiling<Weight>>(1, 1, 1, 1, 
                                                               layer_nnzs, layer_nrows, layer_ncols, 
                                                               layer_file, file_type, 
                                                               TILING_TYPE::_1D_COL_, layer_compression_type, hashers[i+1]));
        bias_vectors[i] = std::move(std::make_shared<struct Data_Block<Weight>>(layer_ncols, Env::rank_socket_id));
        if(bias_type == VALUE_TYPE::_CONSTANT_) {                
            Weight* b_A = bias_vectors[i]->ptr;
            for(uint32_t j = 0; j < layer_ncols; j++) { b_A[j] = bias_value; }
        }
        else if(bias_type == VALUE_TYPE::_INSTANCE_AND_VALUE_PAIRS_) {
            std::string bias_file = bias_files[i];
            std::vector<Weight> bias_values;
            uint32_t c = IO::read_file_iv<Weight>(bias_file, file_type, hashers[i+1], false, bias_type, bias_values, layer_ncols);
            Weight* b_A = bias_vectors[i]->ptr;
            for(uint32_t j = 0; j < layer_ncols; j++) { b_A[j] = bias_values[j]; }
        }
        Logging::enabled = false; 
        if(i%10==0 and Env::rank == 0) { printf("|"); }
    }
    Logging::enabled = true;
    Logging::print(Logging::LOG_LEVEL::VOID, "\n"); 
    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Done reading %d layer files.\n", nmax_layers); 
    Env::barrier();

    spa_vectors.resize(Env::nthreads);
    for(int32_t i = 0; i < Env::nthreads; i++) {
        if((input_compression_type == COMPRESSED_FORMAT::_UDC_) or (input_compression_type == COMPRESSED_FORMAT::_CSC_)) {
            uint32_t max_height = input_features->get_tile_info_max("height");
            spa_vectors[i] = std::move(std::make_shared<struct Data_Block<Weight>>(max_height, Env::threads_socket_id[i]));    
        }
        else if(input_compression_type == COMPRESSED_FORMAT::_CSR_) {
            uint32_t max_width = input_features->get_tile_info_max("width");
            max_width = (nneurons > max_width) ? nneurons : max_width;
            spa_vectors[i] = std::move(std::make_shared<struct Data_Block<Weight>>(max_width, Env::threads_socket_id[i]));    
        }
        else {
            Logging::print(Logging::LOG_LEVEL::ERROR, "%s compression not implemented\n", COMPRESSED_FORMATS[input_compression_type]);
            std::exit(Env::finalize());
        }
    }

    if(parallelism_type == PARALLELISM_TYPE::_DATA_X_MODEL_) {
        output = std::move(std::make_unique<Tiling<Weight>>(Env::nranks, Env::nranks, 1, Env::nranks, 
                                                            0, input_ninstanses, nneurons, 
                                                            TILING_TYPE::_1D_ROW_, input_compression_type, hashers[0]));
    }
    else if((parallelism_type == PARALLELISM_TYPE::_MANAGER_X_WORKER_) or (parallelism_type == PARALLELISM_TYPE::_WORK_X_STEALING_)) {
        output = std::move(std::make_unique<Tiling<Weight>>(Env::nranks * Env::nthreads * split_factor, Env::nranks * Env::nthreads * split_factor, 1, Env::nranks, 
                                                            Env::nthreads, Env::nranks * Env::nthreads, 
                                                            0, input_ninstanses, nneurons, 
                                                            TILING_TYPE::_1D_ROW_, input_compression_type, hashers[0]));
    }
    else {
        output = std::move(std::make_unique<Tiling<Weight>>(Env::nranks * Env::nthreads, Env::nranks * Env::nthreads, 1, Env::nranks, 
                                                            Env::nthreads, Env::nranks * Env::nthreads, 
                                                            0, input_ninstanses, nneurons, 
                                                            TILING_TYPE::_1D_ROW_, input_compression_type, hashers[0]));
    }
    output->set_tile_info(input_features->tiles);

    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Running the inference method [Input compression=%s|Layer compression=%s|Parallelism=%s|Scheduling=%s|Hashing=%s].\n", 
                       COMPRESSED_FORMATS[input_compression_type], COMPRESSED_FORMATS[layer_compression_type], PARALLELISM_TYPES[parallelism_type], SCHEDULING_TYPES[scheduling_type], HASHING_TYPES[hashing_type]); 
    auto finish = std::chrono::high_resolution_clock::now();
    Env::io_time = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish-start).count())/1e9;
    Env::barrier();
    
    execute();

    finish = std::chrono::high_resolution_clock::now();
    Env::end_to_end_time = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish-start).count())/1e9;
    Env::barrier();
    
    printTimes();
}

template<typename Weight>
void Net<Weight>::printTimes() {
    Env::barrier();
    Logging::print(Logging::LOG_LEVEL::INFO, "Time: min, max, mean, std_dev, sum\n");
    double min = 0.0, max = 0.0, mean = 0.0, std_dev = 0.0, sum = 0.0;
    if(Env::nranks == 1) {        
        Env::stats<double>(Env::execution_time, min, max, mean, std_dev, sum);
        Logging::print(Logging::LOG_LEVEL::VOID, "Exe time: %.3f %.3f %.3f %.3f\n", min, max, mean, std_dev, sum);
        Logging::print(Logging::LOG_LEVEL::VOID, "I/O time: %.3f\n", Env::io_time);
        Logging::print(Logging::LOG_LEVEL::VOID, "Run time: %.3f\n", Env::end_to_end_time);
    }
    else {
        int index = std::distance(Env::execution_time.begin(), std::max_element(Env::execution_time.begin(), Env::execution_time.end()));
        double exec_time = Env::execution_time[index];
        std::tie(min, max, mean, std_dev, sum) =  Env::statistics<double>(exec_time);
        Logging::print(Logging::LOG_LEVEL::VOID, "Exe time: %.3f %.3f %.3f %.3f\n", min, max, mean, std_dev, sum);
        
        std::tie(min, max, mean, std_dev, sum) =  Env::statistics<double>(Env::io_time);
        Logging::print(Logging::LOG_LEVEL::VOID, "I/O time: %.3f %.3f %.3f %.3f\n", min, max, mean, std_dev, sum);
        std::tie(min, max, mean, std_dev, sum) =  Env::statistics<double>(Env::end_to_end_time);
        Logging::print(Logging::LOG_LEVEL::VOID, "Run time: %.3f %.3f %.3f %.3f\n", min, max, mean, std_dev, sum);
    }
}

template<typename Weight>
void Net<Weight>::execute() {
    std::vector<std::thread> threads;
    
    for(int i = 0; i < Env::nthreads; i++) {
        threads.push_back(std::thread(&Net<Weight>::inference, this, i));
    }
    
    for(std::thread& th: threads) {
        th.join();
    }
}

template<typename Weight>
void Net<Weight>::inference(const int32_t tid) {
    if(Env::NUMA_ALLOC) { (void)Env::set_thread_affinity(tid); }
    
    if(parallelism_type == PARALLELISM_TYPE::_DATA_X_MODEL_) { data_x_model(tid); }
    else if(parallelism_type == PARALLELISM_TYPE::_DATA_X_DATA_) { data_x_data(tid); }
    else if(parallelism_type == PARALLELISM_TYPE::DATA_THEN_MODEL) { hybrid_x_hybrid(tid); }
    else if(parallelism_type == PARALLELISM_TYPE::_MANAGER_X_WORKER_) { manager_x_worker(tid); }    
    else if(parallelism_type == PARALLELISM_TYPE::_WORK_X_STEALING_) { work_x_stealing(tid); }
}

template<typename Weight>
void Net<Weight>::data_x_model(const int32_t tid) {
    auto start_t = std::chrono::high_resolution_clock::now();  
    uint32_t leader_rowgroup = Env::rank;
    const int32_t leader_tid = 0;
    struct Env::thread_struct& thread_st = Env::threads[tid];

    uint32_t A_nrows = 0, B_nrows = 0, B_ncols = 0;
    uint32_t start = 0, end = 0;
    uint32_t sub_start = 0, sub_end = 0;
    uint32_t l = 0;
    for (uint32_t l = 0; l < nmax_layers; l++) {    
        struct Tile<Weight>& A_tile = (input_compression_type == COMPRESSED_FORMAT::_UDC_) ? (not(l%2)) ? input_features->tiles[leader_rowgroup][0]
                                                                                                  : output->tiles[leader_rowgroup][0] 
                                                                                     : input_features->tiles[leader_rowgroup][0];    
        std::shared_ptr<struct Compressed_Format<Weight>>& A_SPMAT = A_tile.spmat;
        struct Tile<Weight>& B_tile = layers[l]->tiles[0][0];
        std::shared_ptr<struct Compressed_Format<Weight>>& B_SPMAT = B_tile.spmat;
        struct Tile<Weight>& C_tile = (input_compression_type == COMPRESSED_FORMAT::_UDC_) ? (not(l%2)) ? output->tiles[leader_rowgroup][0] 
                                                                                                  : input_features->tiles[leader_rowgroup][0]
                                                                                     : output->tiles[leader_rowgroup][0];
        std::shared_ptr<struct Compressed_Format<Weight>>& C_SPMAT = C_tile.spmat;
        std::shared_ptr<struct Data_Block<Weight>>& s_spa = spa_vectors[tid];
        std::shared_ptr<struct Data_Block<Weight>>& b_bias = bias_vectors[l];
        
        A_nrows = A_SPMAT->nrows;
        B_nrows = B_SPMAT->nrows;
        B_ncols = B_SPMAT->ncols;
        //printf("Tid=%d l=%d A[%d] B[%d %d]\n", tid, l, A_nrows, B_nrows, B_ncols);
        if(tid == leader_tid) {
            for(int32_t i = 0; i < Env::nthreads; i++) {
                Env::threads[i].start_row = ((B_nrows/Env::nthreads) * i);    
                Env::threads[i].end_row = (i == (Env::nthreads-1)) ? B_nrows : ((B_nrows/Env::nthreads) * (i+1));    
                
                Env::threads[i].start_col = ((B_ncols/Env::nthreads) * i);    
                Env::threads[i].end_col = (i == (Env::nthreads-1)) ? B_ncols : ((B_ncols/Env::nthreads) * (i+1));    
            }    
        }    
        pthread_barrier_wait(&Env::thread_barrier);
        
        if((layer_compression_type == COMPRESSED_FORMAT::_UDC_) or (layer_compression_type == COMPRESSED_FORMAT::_CSC_)) {
            start = Env::threads[tid].start_col;
            end = Env::threads[tid].end_col;
            sub_start = 0, sub_end   = 0;
        }
        else {
            Logging::print(Logging::LOG_LEVEL::ERROR, "%s compression not implemented\n", COMPRESSED_FORMATS[layer_compression_type]);
            std::exit(Env::finalize());
        }
        bool last_layer = (category_type == VALUE_TYPE::_INSTANCE_AND_VALUE_PAIRS_) and (l==nmax_layers-1);
        data_x_model_1_iter(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, noop_function, activation_function,
                            A_nrows, B_ncols, 
                            start, end, 
                            sub_start, sub_end, 
                            thread_st, last_layer, input_compression_type, layer_compression_type, leader_tid, tid); 
        //if(!Env::rank and !tid) printf("Total checksum=51009.396646, Total count=282192\n");
        //break;
    }
    auto finish_t = std::chrono::high_resolution_clock::now();
    Env::execution_time[tid] = (double)(std::chrono::duration_cast<std::chrono::nanoseconds>(finish_t - start_t).count())/1e9;
    
    
    struct Tile<Weight>& C_tile = (layer_compression_type == COMPRESSED_FORMAT::_UDC_) ? (not((l-1)%2)) ? output->tiles[leader_rowgroup][0] 
                                                                                                  : input_features->tiles[leader_rowgroup][0]
                                                                                 : input_features->tiles[leader_rowgroup][0];
    const std::shared_ptr<struct Compressed_Format<Weight>>& C_SPMAT = C_tile.spmat;
    data_x_model_validate_prediction(C_SPMAT, C_tile.start_row, true_categories, predicted_nistances, category_type, classifier, leader_tid, tid);
}

template<typename Weight>
void Net<Weight>::data_x_data(const int32_t tid) {
    auto start_t = std::chrono::high_resolution_clock::now();  
    uint32_t leader_rowgroup = Env::thread_rowgroup[tid];
    int32_t leader_tid = 0;
    struct Env::thread_struct& thread_st = Env::threads[tid];
    uint32_t A_nrows = 0, B_nrows = 0, B_ncols = 0;
    uint32_t start = 0, end = 0;
    const uint32_t off = 0;
    uint32_t l = 0;
    for (l = 0; l < nmax_layers; l++) {
        struct Tile<Weight>& A_tile = (not(l%2)) ? input_features->tiles[leader_rowgroup][0]
                                                 : output->tiles[leader_rowgroup][0];
        std::shared_ptr<struct Compressed_Format<Weight>>& A_SPMAT = A_tile.spmat;
        struct Tile<Weight>& B_tile = layers[l]->tiles[0][0];    
        std::shared_ptr<struct Compressed_Format<Weight>>& B_SPMAT = B_tile.spmat;
        struct Tile<Weight>& C_tile = (not(l%2)) ? output->tiles[leader_rowgroup][0]
                                                 : input_features->tiles[leader_rowgroup][0];
        std::shared_ptr<struct Compressed_Format<Weight>>& C_SPMAT = C_tile.spmat;
        std::shared_ptr<struct Data_Block<Weight>>& s_spa = spa_vectors[tid];
        std::shared_ptr<struct Data_Block<Weight>>& b_bias = bias_vectors[l];

        A_nrows = A_SPMAT->nrows;
        B_nrows = B_SPMAT->nrows;
        B_ncols = B_SPMAT->ncols;
        
        if((input_compression_type == COMPRESSED_FORMAT::_UDC_) or (input_compression_type == COMPRESSED_FORMAT::_CSC_)) { end = B_ncols; }
        else if (input_compression_type == COMPRESSED_FORMAT::_CSR_) { end = A_nrows; }
        else {
            Logging::print(Logging::LOG_LEVEL::ERROR, "%s compression not implemented\n", COMPRESSED_FORMATS[input_compression_type]);
            std::exit(Env::finalize());
        }
        bool last_layer = (category_type == VALUE_TYPE::_INSTANCE_AND_VALUE_PAIRS_) and (l==nmax_layers-1);
        data_x_data_1_iter(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, noop_function, activation_function,
                           A_nrows, B_ncols, start, end, off, 
                           thread_st, last_layer, input_compression_type, layer_compression_type, leader_tid, tid); 
    }
    auto finish_t = std::chrono::high_resolution_clock::now();
    Env::execution_time[tid] = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish_t - start_t).count())/1e9;
    
    struct Tile<Weight>& C_tile = (not((l-1)%2)) ? output->tiles[leader_rowgroup][0] 
                                             : input_features->tiles[leader_rowgroup][0];
    const std::shared_ptr<struct Compressed_Format<Weight>> C_SPMAT = C_tile.spmat;
    data_x_data_validate_prediction(C_SPMAT, C_tile.start_row, true_categories, predicted_nistances, category_type, classifier, leader_tid, tid);
}

template<typename Weight>
void Net<Weight>::hybrid_x_hybrid(const int32_t tid) {
    auto start_t = std::chrono::high_resolution_clock::now();  
    uint32_t my_rowgroup = Env::thread_rowgroup[tid];
    int32_t leader_tid = 0;
    uint32_t my_start_layer = hybrid_x_data(Env::my_threads[tid], my_rowgroup, tid);
    if(my_start_layer < nmax_layers) hybrid_x_model(Env::my_threads[tid], my_rowgroup, my_start_layer, my_start_layer, tid, tid);
    while(add_to_idle_threads(Env::my_threads[tid], tid)) {
        const int32_t leader = Env::threads[tid].leader;
        uint32_t leader_rowgroup = Env::threads[tid].rowgroup;
        uint32_t leader_start_layer = Env::threads[tid].start_layer;
        uint32_t leader_current_layer = Env::threads[tid].current_layer;
        hybrid_x_model(Env::my_threads[tid], leader_rowgroup, leader_start_layer, leader_current_layer, leader, tid);
    }
    auto finish_t = std::chrono::high_resolution_clock::now();
    Env::execution_time[tid] = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish_t - start_t).count())/1e9;
    uint32_t layer_index = (my_start_layer == nmax_layers) ? nmax_layers-1 : my_start_layer;
    
    struct Tile<Weight>& C_tile = (not(layer_index%2)) ? output->tiles[my_rowgroup][0]
                                                       : input_features->tiles[my_rowgroup][0];
    std::shared_ptr<struct Compressed_Format<Weight>> C_SPMAT = C_tile.spmat;
    data_x_data_validate_prediction(C_SPMAT, C_tile.start_row, true_categories, predicted_nistances, category_type, classifier, leader_tid, tid);
}

template<typename Weight>
uint32_t Net<Weight>::hybrid_x_data(std::deque<int32_t>& leader_owned_threads, const int32_t my_rowgroup, const int32_t tid) {
    int32_t sid = (numa_queues) ? Env::threads_socket_id[tid] : Env::rank_socket_id;
    int32_t leader_tid = 0;
    struct Env::thread_struct& thread_st = Env::threads[tid];
    uint32_t A_nrows = 0, B_nrows = 0, B_ncols = 0;
    uint32_t start = 0, end = 0;
    const uint32_t off = 0;
    bool breaking = false;
    uint32_t l = 0; 
    for (l = 0; l < nmax_layers; l++) {
        //printf("1.tid=%d l=%d\n", tid, l);
        struct Tile<Weight>& A_tile = (not(l%2)) ? input_features->tiles[my_rowgroup][0]
                                                 : output->tiles[my_rowgroup][0];
        std::shared_ptr<struct Compressed_Format<Weight>> A_SPMAT = A_tile.spmat;
        struct Tile<Weight>& B_tile = layers[l]->tiles[0][0];    
        std::shared_ptr<struct Compressed_Format<Weight>> B_SPMAT = B_tile.spmat;
        struct Tile<Weight>& C_tile = (not(l%2)) ? output->tiles[my_rowgroup][0]
                                                 : input_features->tiles[my_rowgroup][0];
        std::shared_ptr<struct Compressed_Format<Weight>>& C_SPMAT = C_tile.spmat;
        std::shared_ptr<struct Data_Block<Weight>>& s_spa = spa_vectors[tid];
        std::shared_ptr<struct Data_Block<Weight>>& b_bias = bias_vectors[l];      
        
        A_nrows = A_SPMAT->nrows;
        B_nrows = B_SPMAT->nrows;
        B_ncols = B_SPMAT->ncols;
        
        if((input_compression_type == COMPRESSED_FORMAT::_CSC_) and (layer_compression_type == COMPRESSED_FORMAT::_CSC_)) { end = B_ncols; }
        else if ((input_compression_type == COMPRESSED_FORMAT::_CSR_) and (layer_compression_type == COMPRESSED_FORMAT::_CSR_)) { end = A_nrows; }
        else {
            Logging::print(Logging::LOG_LEVEL::ERROR, "[%sx%s] multiplication not implemented\n", COMPRESSED_FORMATS[input_compression_type], COMPRESSED_FORMATS[layer_compression_type]);
            std::exit(Env::finalize());
        }

        uint32_t A_ncols = A_SPMAT->ncols;
        if((l >= nmax_layers*recruiting_ratio) and add_to_my_follower_threads(leader_owned_threads, my_rowgroup, l, l, A_nrows, B_ncols, tid, tid)) { break; }

        bool last_layer = (category_type == VALUE_TYPE::_INSTANCE_AND_VALUE_PAIRS_) and (l==nmax_layers-1);
        data_x_data_1_iter(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, noop_function, activation_function,
                           A_nrows, B_ncols, start, end, off, 
                           thread_st, last_layer, input_compression_type, layer_compression_type, leader_tid, tid); 
                           
        Env::scores[sid][tid]++;     
    }
    return(l);
}

template<typename Weight>
void Net<Weight>::hybrid_x_model(std::deque<int32_t>& leader_owned_threads, const uint32_t leader_rowgroup, const uint32_t leader_start_layer, const uint32_t leader_current_layer, const int32_t leader_tid, const int32_t tid) {
    int32_t sid = (numa_queues) ? Env::threads_socket_id[tid] : Env::rank_socket_id;
    struct Env::thread_struct& thread_st = Env::threads[tid];    
    
    uint32_t A_nrows = 0, B_nrows = 0, B_ncols = 0;
    uint32_t start = 0, end = 0;
    const uint32_t off = 0;  
    uint32_t B_ncols_prev = 0;
    struct Tile<Weight>& A_tile = (not(leader_start_layer%2)) ? input_features->tiles[leader_rowgroup][0]
                                                           : output->tiles[leader_rowgroup][0];
    struct Tile<Weight>& C_tile = (not(leader_start_layer%2)) ? output->tiles[leader_rowgroup][0]
                                                           : input_features->tiles[leader_rowgroup][0];

    B_ncols = layers[leader_current_layer]->tiles[0][0].spmat->ncols;
    for (uint32_t l = leader_current_layer; l < nmax_layers; l++) {
        std::shared_ptr<struct Compressed_Format<Weight>>& A_SPMAT = A_tile.spmat;
        struct Tile<Weight>& B_tile = layers[l]->tiles[0][0];
        std::shared_ptr<struct Compressed_Format<Weight>>& B_SPMAT = B_tile.spmat;
        std::shared_ptr<struct Compressed_Format<Weight>>& C_SPMAT = C_tile.spmat;
        std::shared_ptr<struct Data_Block<Weight>>& s_spa = spa_vectors[tid];
        std::shared_ptr<struct Data_Block<Weight>>& b_bias = bias_vectors[l];
        
        B_ncols_prev = B_ncols;
        A_nrows = A_SPMAT->nrows;
        B_nrows = B_SPMAT->nrows;
        B_ncols = B_SPMAT->ncols;
        
        add_to_my_follower_threads(leader_owned_threads, leader_rowgroup, leader_start_layer, l, A_nrows, B_ncols, leader_tid, tid);
        double start_time = Env::tic();   
            Env::decrease_num_threads(1, leader_tid, tid);
            Env::init_num_threads(leader_owned_threads.size(), leader_tid, tid);
        Env::hybrid_probe_time[tid] += Env::toc(start_time);     
        
        if((input_compression_type == COMPRESSED_FORMAT::_CSC_) and (layer_compression_type == COMPRESSED_FORMAT::_CSC_)) {
            if(B_ncols != B_ncols_prev) {                
                if(tid == leader_tid){
                    uint32_t num_threads = leader_owned_threads.size();
                    if(num_threads > B_ncols) {
                        for(uint32_t i = 0; i < num_threads; i++) {
                            int32_t t = leader_owned_threads[i];
                            Env::threads[t].start_col = (i<B_ncols) ? i : 0;
                            Env::threads[t].end_col   = (i<B_ncols) ? i+1 : 0;
                        }
                    }
                    else {
                        for(uint32_t i = 0; i < num_threads; i++) {
                            int32_t t = leader_owned_threads[i];
                            Env::threads[t].start_col = ((B_ncols/num_threads) * i);
                            Env::threads[t].end_col   = (i == (num_threads-1)) ? B_ncols : ((B_ncols/num_threads) * (i+1));
                        }
                    }
                }
                pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
            }
            start = Env::threads[tid].start_col;
            end = Env::threads[tid].end_col;
        }
        else if((input_compression_type == COMPRESSED_FORMAT::_CSR_) and (layer_compression_type == COMPRESSED_FORMAT::_CSR_)) {
            start = Env::threads[tid].start_row;
            end = Env::threads[tid].end_row;
        }
        else {
            Logging::print(Logging::LOG_LEVEL::ERROR, "[%sx%s] multiplication not implemented\n", COMPRESSED_FORMATS[input_compression_type], COMPRESSED_FORMATS[layer_compression_type]);
            std::exit(Env::finalize());
        }
        uint32_t A_ncols = A_SPMAT->ncols;
        bool last_layer = (category_type == VALUE_TYPE::_INSTANCE_AND_VALUE_PAIRS_) and (l==nmax_layers-1);
        data_x_model_hybrid_1_iter(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, noop_function, activation_function,
               A_nrows, B_ncols, start, end, off,
               leader_owned_threads, thread_st, last_layer, input_compression_type, layer_compression_type, leader_tid, tid);
               
       if(tid == leader_tid) Env::scores[sid][tid]++;
    }
}

template<typename Weight>
bool Net<Weight>::thread_scheduling(std::deque<int32_t>& leader_owned_threads, const uint32_t leader_rowgroup, std::deque<int32_t>& follower_threads, int32_t socket_id, const uint32_t leader_start_layer, const uint32_t leader_current_layer, const uint32_t nrows, const uint32_t ncols, const int32_t leader, const int32_t tid) {  
    //uint32_t k1 = B_ncols%num_threads; uint32_t k1_len = (B_ncols+num_threads-1)/num_threads;
    //uint32_t k2 = num_threads-k1; uint32_t k2_len = B_ncols/num_threads;
    bool found = false;
    uint32_t num_threads = leader_owned_threads.size();
    uint32_t old_num_threads = leader_owned_threads.size();
    uint32_t num_new_threads = 0;
    uint32_t min_score_value = (uint32_t) INT32_MAX;
    uint32_t max_score_value = 0;
    int32_t sid = (numa_queues) ? Env::threads_socket_id[tid] : Env::rank_socket_id;
    
    if(!follower_threads.empty()) {
        pthread_mutex_lock(&Env::numa_thread_mutex[socket_id]);  
        if(!follower_threads.empty()) {
            if((sid == socket_id) and ((scheduling_type == SCHEDULING_TYPE::_SLOWER_FIRST_) or (scheduling_type == SCHEDULING_TYPE::_FASTER_FIRST_))) { 
                for(std::vector<uint32_t>::iterator it = Env::scores[socket_id].begin() + Env::queue_indices[socket_id].first ; it != Env::scores[socket_id].begin() + Env::queue_indices[socket_id].second ; it++) {
                    if((*it >= nmax_layers) or (*it == 0)) continue;
                    
                    if(*it > max_score_value) {
                        max_score_value = *it;
                    }
                    if(*it < min_score_value) {
                        min_score_value = *it;
                    }
                    
                }
            }
            
            bool pick = (((sid == socket_id) and ((scheduling_type == SCHEDULING_TYPE::_EARLIEST_FIRST_) or 
                                                   ((scheduling_type == SCHEDULING_TYPE::_SLOWER_FIRST_) and ((Env::scores[socket_id][tid] - min_score_value) < schduling_threshold)) or
                                                   ((scheduling_type == SCHEDULING_TYPE::_FASTER_FIRST_) and ((max_score_value  - Env::scores[socket_id][tid]) < schduling_threshold)))) or
                         (sid != socket_id));
            //printf("Rank=%d tid=%2d layer=%d min=%d max=%d pick=%d\n", Env::rank, tid, start_layer, min_score_value, max_score_value, pick);
            //uint32_t nworking = Env::nthreads_per_socket[socket_id] - Env::numa_num_finished_threads[socket_id];
            //uint32_t nfinished = Env::numa_num_finished_threads[socket_id];
            //uint32_t nhelping = Env::numa_num_finished_threads[socket_id] - follower_threads.size();
            //uint32_t nidles = follower_threads.size();
            //printf("Rank=%d tid=%2d layer=%d nworking=%d nfinished=%d nhelping=%d nidles=%d min=%d max=%d pick=%d\n", Env::rank, tid, start_layer, nworking, nfinished, nhelping, nidles, min_score_value, max_score_value, pick);
            if(pick) {
                if(leader_owned_threads.empty()) {
                    leader_owned_threads.push_back(tid);
                    num_threads++;
                }

                num_threads += follower_threads.size();
                leader_owned_threads.insert(leader_owned_threads.end(), follower_threads.begin(), follower_threads.end());
                follower_threads.erase(follower_threads.begin(), follower_threads.end());
                
                if(input_compression_type == COMPRESSED_FORMAT::_CSR_) {
                    if(num_threads > nrows) {
                        for(uint32_t i = 0; i < num_threads; i++) {
                            int32_t t = leader_owned_threads[i];
                            Env::threads[t].index = i;
                            Env::threads[t].leader = tid;
                            Env::threads[t].rowgroup = leader_rowgroup;
                            Env::threads[t].start_layer = leader_start_layer;
                            Env::threads[t].current_layer = leader_current_layer;
                            Env::threads[t].start_row = (i<nrows) ? i : 0;
                            Env::threads[t].end_row   = (i<nrows) ? i+1 : 0;
                            Env::threads[t].idx_nnz = 0;
                            Env::threads[t].off_nnz = 0;
                            Env::threads[t].dis_nnz = 0;
                        }      
                    }
                    else {
                        for(uint32_t i = 0; i < num_threads; i++) {
                            int32_t t = leader_owned_threads[i];
                            Env::threads[t].index = i;
                            Env::threads[t].leader = tid;
                            Env::threads[t].rowgroup = leader_rowgroup;
                            Env::threads[t].start_layer = leader_start_layer;
                            Env::threads[t].current_layer = leader_current_layer;
                            Env::threads[t].start_row = ((nrows/num_threads) * i);
                            Env::threads[t].end_row   = (i == (num_threads-1)) ? nrows : ((nrows/num_threads) * (i+1));
                            Env::threads[t].idx_nnz = 0;
                            Env::threads[t].off_nnz = 0;
                            Env::threads[t].dis_nnz = 0;
                        }                     
                    }
                }
                else if(input_compression_type == COMPRESSED_FORMAT::_CSC_) {
                    if(num_threads > ncols) {
                        for(uint32_t i = 0; i < num_threads; i++) {
                            int32_t t = leader_owned_threads[i];
                            Env::threads[t].index = i;
                            Env::threads[t].leader = tid;
                            Env::threads[t].rowgroup = leader_rowgroup;
                            Env::threads[t].start_layer = leader_start_layer;
                            Env::threads[t].current_layer = leader_current_layer;
                            Env::threads[t].start_col = (i<ncols) ? i : 0;
                            Env::threads[t].end_col   = (i<ncols) ? i+1 : 0;
                            Env::threads[t].idx_nnz = 0;
                            Env::threads[t].off_nnz = 0;
                            Env::threads[t].dis_nnz = 0;
                        }
                    }
                    else {
                        for(uint32_t i = 0; i < num_threads; i++) {
                            int32_t t = leader_owned_threads[i];
                            Env::threads[t].index = i;
                            Env::threads[t].leader = tid;
                            Env::threads[t].rowgroup = leader_rowgroup;
                            Env::threads[t].start_layer = leader_start_layer;
                            Env::threads[t].current_layer = leader_current_layer;
                            Env::threads[t].start_col = ((ncols/num_threads) * i);
                            Env::threads[t].end_col   = (i == (num_threads-1)) ? ncols : ((ncols/num_threads) * (i+1));
                            Env::threads[t].idx_nnz = 0;
                            Env::threads[t].off_nnz = 0;
                            Env::threads[t].dis_nnz = 0;
                        }
                    }
                }                    
                int ret1 = pthread_barrier_destroy(&Env::thread_barriers[tid]);
                int ret = pthread_barrier_init(&Env::thread_barriers[tid], NULL, num_threads);
                num_new_threads = num_threads - old_num_threads;
                Env::increase_num_threads(num_new_threads, leader, tid);
                //printf("tid=%d/%d n=%d old=%d new=%d dest=%d init=%d\n", leader, tid, num_threads, old_num_threads, num_new_threads, ret1, ret);
                pthread_cond_broadcast(&Env::numa_thread_cond[socket_id]); 
                
                found = true;
            }
        }
        pthread_mutex_unlock(&Env::numa_thread_mutex[socket_id]);
    }
    return(found);
}


template<typename Weight>
bool Net<Weight>::add_to_my_follower_threads(std::deque<int32_t>& leader_owned_threads, const uint32_t leader_rowgroup, const uint32_t leader_start_layer, const uint32_t leader_current_layer, const uint32_t nrows, const uint32_t ncols, const int32_t leader_tid, const int32_t tid) {  
    bool found = false;
    if(tid == leader_tid) {
        double start_time = 0;
        start_time = Env::tic();
        int32_t sid = (numa_queues) ? Env::threads_socket_id[tid] : Env::rank_socket_id;
        if(numa_queues) {
            for(int32_t s = 0; s < Env::nsockets; s++) {
                int32_t si = (s + Env::threads_socket_id[tid]) % Env::nsockets;
                if((si == sid) or (Env::nthreads_per_socket[si] and (Env::numa_follower_threads[si].size() == (uint32_t) Env::nthreads_per_socket[si]))) {
                    found |= thread_scheduling(leader_owned_threads, leader_rowgroup, Env::numa_follower_threads[si], si, leader_start_layer, leader_current_layer, nrows, ncols, leader_tid, tid);
                }
            }
        }
        else {
            found = thread_scheduling(leader_owned_threads, leader_rowgroup, Env::numa_follower_threads[sid], sid, leader_start_layer, leader_current_layer, nrows, ncols, leader_tid, tid);
        }
        Env::hybrid_probe_time[tid] += Env::toc(start_time);  
    }

    return(found);
}

template<typename Weight>
bool Net<Weight>::add_to_idle_threads(std::deque<int32_t>& leader_owned_threads, const int32_t tid) {
    uint32_t sid = (numa_queues) ? Env::threads_socket_id[tid] : Env::rank_socket_id;
    bool status = true;
    uint32_t all_done = 0;
    
    pthread_mutex_lock(&Env::numa_thread_mutex[sid]);
    Env::numa_follower_threads[sid].push_back(tid);
    if(not leader_owned_threads.empty()) leader_owned_threads.erase(leader_owned_threads.begin(), leader_owned_threads.end());
    
    Env::threads[tid].leader = -1;
        
    for(std::deque<int32_t>& numa_thread: Env::numa_follower_threads) {
        all_done += numa_thread.size();
    }
    
    if(all_done == (uint32_t) Env::nthreads) {         
        pthread_mutex_unlock(&Env::numa_thread_mutex[sid]);   
        for(int32_t s = 0; s < Env::nsockets; s++) {
            pthread_mutex_lock(&Env::numa_thread_mutex[s]);
            pthread_cond_broadcast(&Env::numa_thread_cond[s]);   
            pthread_mutex_unlock(&Env::numa_thread_mutex[s]);  
        }
        status = false;
    }
    else {
        pthread_cond_wait(&Env::numa_thread_cond[sid], &Env::numa_thread_mutex[sid]); 
        pthread_mutex_unlock(&Env::numa_thread_mutex[sid]); 
        all_done = 0;
        for(std::deque<int32_t>& numa_thread: Env::numa_follower_threads) all_done += numa_thread.size();
        if(all_done == (uint32_t) Env::nthreads) status = false;
    }
    return(status);
}

template<typename Weight>
void Net<Weight>::manager_x_worker(const int32_t tid) {
    auto start_t = std::chrono::high_resolution_clock::now(); 
    uint32_t leader_rowgroup = 0;
    int32_t leader_tid = 0;
    struct Env::thread_struct& thread_st = Env::threads[tid];
    uint32_t A_nrows = 0, B_nrows = 0, B_ncols = 0;
    uint32_t start = 0, end = 0;
    const uint32_t off = 0;
    while(!Env::rank_rowgroups.empty()) {
        pthread_mutex_lock(&Env::thread_mutex_q);  
        if(!Env::rank_rowgroups.empty()) {
            leader_rowgroup = Env::rank_rowgroups.front();
            Env::processed_rowgroups.push_back(leader_rowgroup);
            Env::rank_rowgroups.pop_front();
            pthread_mutex_unlock(&Env::thread_mutex_q);
        }
        else {
            pthread_mutex_unlock(&Env::thread_mutex_q);
            break;
        }
        
        for (uint32_t l = 0; l < nmax_layers; l++) {
            struct Tile<Weight>& A_tile = (not(l%2)) ? input_features->tiles[leader_rowgroup][0]
                                                     : output->tiles[leader_rowgroup][0];
            std::shared_ptr<struct Compressed_Format<Weight>>& A_SPMAT = A_tile.spmat;
            struct Tile<Weight>& B_tile = layers[l]->tiles[0][0];    
            std::shared_ptr<struct Compressed_Format<Weight>>& B_SPMAT = B_tile.spmat;
            struct Tile<Weight>& C_tile = (not(l%2)) ? output->tiles[leader_rowgroup][0]
                                                     : input_features->tiles[leader_rowgroup][0];
            std::shared_ptr<struct Compressed_Format<Weight>>& C_SPMAT = C_tile.spmat;
            std::shared_ptr<struct Data_Block<Weight>>& s_spa = spa_vectors[tid];
            std::shared_ptr<struct Data_Block<Weight>>& b_bias = bias_vectors[l];    
            
            A_nrows = A_SPMAT->nrows;
            B_nrows = B_SPMAT->nrows;
            B_ncols = B_SPMAT->ncols;
            
            if((input_compression_type == COMPRESSED_FORMAT::_CSC_) and (layer_compression_type == COMPRESSED_FORMAT::_CSC_)) { end = B_ncols; }
            else if ((input_compression_type == COMPRESSED_FORMAT::_CSR_) and (layer_compression_type == COMPRESSED_FORMAT::_CSR_)) { end = A_nrows; }
            else {
                Logging::print(Logging::LOG_LEVEL::ERROR, "[%sx%s] multiplication not implemented\n", COMPRESSED_FORMATS[input_compression_type], COMPRESSED_FORMATS[layer_compression_type]);
                std::exit(Env::finalize());
            }
            bool last_layer = (category_type == VALUE_TYPE::_INSTANCE_AND_VALUE_PAIRS_) and (l==nmax_layers-1);
            data_x_data_1_iter(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, noop_function, activation_function,
                               A_nrows, B_ncols, start, end, off, 
                               thread_st, last_layer, input_compression_type, layer_compression_type, leader_tid, tid);       
        }   
    }
    auto finish_t = std::chrono::high_resolution_clock::now();
    Env::execution_time[tid] = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish_t - start_t).count())/1e9;

    std::vector<std::vector<struct Tile<Weight>>> C_tiles = (not((nmax_layers-1)%2)) ? output->tiles : input_features->tiles;
    manager_x_worker_validate_prediction(C_tiles, true_categories, predicted_nistances, category_type, classifier, leader_tid, tid);
}


template<typename Weight>
void Net<Weight>::work_x_stealing(const int32_t tid) {
    auto start_t = std::chrono::high_resolution_clock::now();  
    uint32_t leader_rowgroup = 0;
    int32_t leader_tid = 0;
    struct Env::thread_struct& thread_st = Env::threads[tid];
    uint32_t A_nrows = 0, B_nrows = 0, B_ncols = 0;
    uint32_t start = 0, end = 0;
    const uint32_t off = 0;
    while(true) {
        bool found = false;
        for(int32_t i = 0; i < Env::nthreads; i++) {
            uint32_t t = (tid+i) % Env::nthreads;
            if(!Env::threads_rowgroups[t].empty()) {
                pthread_mutex_lock(&Env::thread_mutexes_qs[t]);
                if(!Env::threads_rowgroups[t].empty()) {                    
                    leader_rowgroup = Env::threads_rowgroups[t].front();
                    Env::processed_rowgroups_per_thread[t].push_back(leader_rowgroup);
                    Env::threads_rowgroups[t].pop_front();
                    found = true;
                }
                pthread_mutex_unlock(&Env::thread_mutexes_qs[t]);   
                if(found) break;
            }
        }
        if(not found) break;
        
        for (uint32_t l = 0; l < nmax_layers; l++) {
            struct Tile<Weight>& A_tile = (not(l%2)) ? input_features->tiles[leader_rowgroup][0]
                                                     : output->tiles[leader_rowgroup][0];
            std::shared_ptr<struct Compressed_Format<Weight>>& A_SPMAT = A_tile.spmat;
            struct Tile<Weight>& B_tile = layers[l]->tiles[0][0];    
            std::shared_ptr<struct Compressed_Format<Weight>>& B_SPMAT = B_tile.spmat;
            struct Tile<Weight>& C_tile = (not(l%2)) ? output->tiles[leader_rowgroup][0]
                                                     : input_features->tiles[leader_rowgroup][0];
            std::shared_ptr<struct Compressed_Format<Weight>>& C_SPMAT = C_tile.spmat;
            std::shared_ptr<struct Data_Block<Weight>>& s_spa = spa_vectors[tid];
            std::shared_ptr<struct Data_Block<Weight>>& b_bias = bias_vectors[l];  
        
            A_nrows = A_SPMAT->nrows;
            B_nrows = B_SPMAT->nrows;
            B_ncols = B_SPMAT->ncols;
            
            if((input_compression_type == COMPRESSED_FORMAT::_CSC_) and (layer_compression_type == COMPRESSED_FORMAT::_CSC_)) { end = B_ncols; }
            else if ((input_compression_type == COMPRESSED_FORMAT::_CSR_) and (layer_compression_type == COMPRESSED_FORMAT::_CSR_)) { end = A_nrows; }
            else {
                Logging::print(Logging::LOG_LEVEL::ERROR, "[%sx%s] multiplication not implemented\n", COMPRESSED_FORMATS[input_compression_type], COMPRESSED_FORMATS[layer_compression_type]);
                std::exit(Env::finalize());
            }
            bool last_layer = (category_type == VALUE_TYPE::_INSTANCE_AND_VALUE_PAIRS_) and (l==nmax_layers-1);
            data_x_data_1_iter(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, noop_function, activation_function,
                               A_nrows, B_ncols, start, end, off, 
                               thread_st, last_layer, input_compression_type, layer_compression_type, leader_tid, tid);       
        }   
    }

    auto finish_t = std::chrono::high_resolution_clock::now();
    Env::execution_time[tid] = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish_t - start_t).count())/1e9;
    
    std::vector<std::vector<struct Tile<Weight>>> C_tiles = (not((nmax_layers-1)%2)) ? output->tiles : input_features->tiles;
    work_x_stealing_validate_prediction(C_tiles, true_categories, predicted_nistances, category_type, classifier, leader_tid, tid);
}

#endif 
