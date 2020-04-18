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
enum PARALLELISM_TYPE {_DATA_X_MODEL_, _DATA_X_DATA_, _HYBRID_X_HYBRID_ , _HYBRID_X_HYBRID_1_, _MANAGER_X_WORKER_, _WORK_X_STEALING_, _SIZE_};
const char* PARALLELISM_TYPES[] = {"_DATA_X_MODEL_", "_DATA_X_DATA_", "_HYBRID_X_HYBRID_", "_HYBRID_X_HYBRID_1_", "_MANAGER_X_WORKER_", "_WORK_X_STEALING_"};

enum SCHEDULING_TYPE {_EARLIEST_FIRST_, _SLOWER_FIRST_, _FASTER_FIRST_, _NONE_};
const char* SCHEDULING_TYPES[] = {"_EARLIEST_FIRST_", "_SLOWER_FIRST_", "_FASTER_FIRST_", "_NONE_"};

template<typename Weight>
class Net {
    public:
        Net() {};
        ~Net() {};
        
        Net(const uint32_t NinputInstanses_, const uint32_t Nneurons_,
            const std::string inputFile_prefix, const uint32_t maxLayers_, const std::string layerFile_prefix,
            const PARALLELISM_TYPE parallelism_type_  = PARALLELISM_TYPE::_HYBRID_X_HYBRID_,
            const COMPRESSED_FORMAT compression_type = COMPRESSED_FORMAT::_CSC_,
            const HASHING_TYPE hashing_type_ = HASHING_TYPE::_BOTH_,
            const INPUT_TYPE input_type = INPUT_TYPE::_BINARY_);

        std::unique_ptr<struct Tiling<Weight>> inputFeatures = nullptr;
        std::vector<uint32_t> trueCategories;
        std::vector<std::vector<std::unique_ptr<struct Tiling<Weight>>>> layers;
        
        std::vector<std::vector<std::shared_ptr<struct Data_Block<Weight>>>> biasWeightVecs;
        std::vector<std::shared_ptr<struct Data_Block<Weight>>> spaWeightVec;
        
        std::unique_ptr<struct Tiling<Weight>> output = nullptr;
        
        uint64_t nedges;
        uint32_t NinputInstanses;        
        uint32_t Nneurons;
        Weight biasValue;
        uint32_t maxLayers;
        int32_t nCategories;
        
        PARALLELISM_TYPE parallelism_type;
        SCHEDULING_TYPE scheduling_type = _SLOWER_FIRST_;
        bool repartition = false;
        bool replication = false;
        uint32_t split_factor = 8;
        bool numa_queues = true;
        uint32_t schduling_threshold = 4;
        
        COMPRESSED_FORMAT compression_type;
        bool dual_spmat = false;
        float recruiting_ratio = .3;
        
        HASHING_TYPE hashing_type;        
        std::shared_ptr<struct TwoDHasher> input_hasher;
        std::shared_ptr<struct TwoDHasher> layer_hasher;

        void printTimes();
        void printTimesExcel();

        void printTimesExcel1();
        void execute();
        void inferenceReLU(const int32_t tid);
        
        void data_x_model(const int32_t tid);
        void data_x_data(const int32_t tid);
        void hybrid_x_hybrid(const int32_t tid);
        void hybrid_x_hybrid_1(const int32_t tid);
        void manager_x_worker(const int32_t tid);
        void work_x_stealing(const int32_t tid);
        uint32_t hybrid_x_data(std::deque<int32_t>& my_threads, const int32_t my_rowgroup, const int32_t tid);
        void hybrid_x_model(std::deque<int32_t>& my_threads, const uint32_t my_rowgroup, const uint32_t leader_start_layer, const int32_t leader_tid, const int32_t tid);
        bool add_to_idle_threads(std::deque<int32_t>& my_threads, const int32_t tid);
        bool    add_to_my_follower_threads(std::deque<int32_t>& my_threads, const uint32_t my_rowgroup, const uint32_t start_layer, const uint32_t start_row, const uint32_t nrows, const uint32_t ncols, const int32_t leader, const int32_t tid);
        bool    thread_scheduling(std::deque<int32_t>& my_threads, const uint32_t my_rowgroup, std::deque<int32_t>& follower_threads, int32_t socket_id, const uint32_t start_layer, const uint32_t start_row, const uint32_t nrows, const uint32_t ncols, const int32_t leader, const int32_t tid);
};

template<typename Weight>
Net<Weight>::Net(const uint32_t NinputInstanses_, const uint32_t Nneurons_,
                 const std::string inputFile_prefix, const uint32_t maxLayers_, const std::string layerFile_prefix,
                 const PARALLELISM_TYPE parallelism_type_, const COMPRESSED_FORMAT compression_type_, 
                 const HASHING_TYPE hashing_type_, const INPUT_TYPE input_type) 
                     : NinputInstanses(NinputInstanses_), Nneurons(Nneurons_), 
                       maxLayers(maxLayers_), parallelism_type(parallelism_type_), compression_type(compression_type_), hashing_type(hashing_type_) {
    
    auto start = std::chrono::high_resolution_clock::now();
    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Processing input feature file for %d neurons and %s\n", Nneurons, PARALLELISM_TYPES[parallelism_type]);  
    std::vector<Weight> neuralNetBias = {-0.3,-0.35,-0.4,-0.45};
    std::vector<uint32_t> NneuronsVector = {1024, 4096, 16384, 65536};    
    uint32_t idxN = std::distance(NneuronsVector.begin(), std::find(NneuronsVector.begin(), NneuronsVector.end(), Nneurons));
    if(idxN >= NneuronsVector.size()) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Invalid number of neurons %d\n", Nneurons);
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
    Net::nedges = nnz;
    //nrows = ((NinputInstanses + 2) > nrows) ? (NinputInstanses + 2) : nrows; 
    nrows = NinputInstanses + 2;
    
    ncols = ((Nneurons + 2) > ncols) ? (Nneurons + 2) : ncols;
    
    ncols += (ncols % Env::nthreads) ? (Env::nthreads - (ncols % Env::nthreads)) : 0;  
    
    if((parallelism_type != PARALLELISM_TYPE::_HYBRID_X_HYBRID_) and (parallelism_type != PARALLELISM_TYPE::_HYBRID_X_HYBRID_1_)) {
        scheduling_type = SCHEDULING_TYPE::_NONE_;
    }

    if(replication and (Env::nthreads_per_socket[Env::rank_socket_id] == (uint32_t) Env::nthreads)) {
            replication = false;
    }
    
    long nbuckets_rows = 1;
    long nbuckets_cols = 1;
    
    input_hasher = std::move(std::make_shared<struct TwoDHasher>(hashing_type, true, nrows, ncols, nbuckets_rows, nbuckets_cols));
    
    uint64_t total_memory = nnz*sizeof(uint32_t) + nnz*sizeof(Weight);
    if(compression_type == COMPRESSED_FORMAT::_CSC_) {
        total_memory += ncols*sizeof(uint32_t);
    }
    else if(compression_type == COMPRESSED_FORMAT::_CSR_) {
        total_memory += nrows*sizeof(uint32_t);
    }
    
    if (parallelism_type == PARALLELISM_TYPE::_HYBRID_X_HYBRID_) {
        split_factor=1;
    }
    else {
        split_factor = (total_memory+Env::L3_CACHE_SIZE-1)/Env::L3_CACHE_SIZE;
    }
    //split_factor=1;
    if(parallelism_type == PARALLELISM_TYPE::_DATA_X_MODEL_) {
        inputFeatures = std::move(std::make_unique<Tiling<Weight>>(Env::nranks, Env::nranks, 1, Env::nranks, 
                                                                   nnz, nrows, ncols, 
                                                                   feature_file, input_type, 
                                                                   TILING_TYPE::_1D_ROW_, compression_type,
                                                                   input_hasher, false, repartition));
    }
    else if((parallelism_type == PARALLELISM_TYPE::_HYBRID_X_HYBRID_1_) or (parallelism_type == PARALLELISM_TYPE::_MANAGER_X_WORKER_) or (parallelism_type == PARALLELISM_TYPE::_WORK_X_STEALING_)) {
        inputFeatures = std::move(std::make_unique<Tiling<Weight>>(Env::nranks * Env::nthreads * split_factor, Env::nranks * Env::nthreads * split_factor, 1, Env::nranks,
                                                                   Env::nthreads, Env::nranks * Env::nthreads, 
                                                                   nnz, nrows, ncols, 
                                                                   feature_file, input_type, 
                                                                   TILING_TYPE::_1D_ROW_, compression_type,
                                                                   input_hasher, false, repartition));        
       inputFeatures->set_threads_indices();
       inputFeatures->set_rank_indices();  
    }
    else {
        inputFeatures = std::move(std::make_unique<Tiling<Weight>>(Env::nranks * Env::nthreads, Env::nranks * Env::nthreads, 1, Env::nranks,
                                                                   Env::nthreads, Env::nranks * Env::nthreads, 
                                                                   nnz, nrows, ncols, 
                                                                   feature_file, input_type, 
                                                                   TILING_TYPE::_1D_ROW_, compression_type,
                                                                   input_hasher, false, repartition));
        inputFeatures->set_thread_index();                                                           
    }
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Processing the category files for %d neurons and %d layers.\n", Nneurons, maxLayers); 
    std::vector<uint32_t> maxLayersVector = {120, 480, 1920};
    uint32_t idxL = std::distance(maxLayersVector.begin(), std::find(maxLayersVector.begin(), maxLayersVector.end(), maxLayers));
    if(idxL >= maxLayersVector.size()) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Invalid number of layers %d\n", maxLayers);
        std::exit(Env::finalize());
    }
    std::string categoryFile = layerFile_prefix + "/neuron" + std::to_string(Nneurons) + "-l" + std::to_string(maxLayers);
    categoryFile += (input_type == INPUT_TYPE::_TEXT_) ? "-categories.tsv" : "-categories.bin";
    
    if(INPUT_TYPE::_TEXT_ == input_type) {
        nCategories = IO::text_file_categories(categoryFile, trueCategories, inputFeatures->nrows);
    }
    else {
        nCategories = IO::binary_file_categories(categoryFile, trueCategories, inputFeatures->nrows, input_hasher);
    }
    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Processing %d layer files (silent).\n", maxLayers); 
    //maxLayers = 2;
    
    layers.resize(Env::nsockets);
    biasWeightVecs.resize(Env::nsockets);
    for(int32_t s = 0; s < Env::nsockets; s++) {
        layers[s].resize(maxLayers);
        biasWeightVecs[s].resize(maxLayers);
    }
    
    if((parallelism_type != PARALLELISM_TYPE::_HYBRID_X_HYBRID_) and (dual_spmat == true)) {
        dual_spmat = false;
    }

    for(uint32_t i = 0; i < maxLayers; i++) {
        std::string layerFile = layerFile_prefix + "/neuron" + std::to_string(Nneurons) + "/n" + std::to_string(Nneurons) + "-l" + std::to_string(i+1);
        layerFile += (input_type == INPUT_TYPE::_TEXT_) ? ".tsv" : ".bin";
        if(i == 0) {
            std::tie(nnz, nrows, ncols) = (INPUT_TYPE::_TEXT_ == input_type) ? IO::text_file_stat<Weight>(layerFile)
                                                                         : IO::binary_file_stat<Weight>(layerFile);   

            nrows = (inputFeatures->ncols > nrows) ? inputFeatures->ncols : nrows; 
            ncols = (inputFeatures->ncols > ncols) ? inputFeatures->ncols : ncols;                 
            
                
            nbuckets_rows = nbuckets_cols;
            nbuckets_cols = nbuckets_cols; // dummy            
            
            //nbuckets_rows = Env::nthreads * 512;
            //nbuckets_cols = Env::nthreads * 512;
             //printf("%d %lu %d %lu\n", nrows, nbuckets_rows, ncols, nbuckets_cols ); std::exit(0);
            layer_hasher = std::move(std::make_shared<struct TwoDHasher>(hashing_type, false, nrows, ncols, nbuckets_rows, nbuckets_cols));
            Env::barrier();
        }
        
        bool enable_dual_spmat = (dual_spmat and (i >= maxLayers*recruiting_ratio)) ? true : false;
        for(int32_t s = 0; s < Env::nsockets; s++) {
            if(replication or (s == Env::rank_socket_id)) {
                /*
                if(parallelism_type == PARALLELISM_TYPE::_DATA_X_MODEL_) {
                    layers[s][i] = std::move(std::make_unique<Tiling<Weight>>(Env::nthreads, 1, Env::nthreads, 1, 
                                                                           Env::nthreads, Env::nthreads, 
                                                                           nnz, nrows, ncols, 
                                                                           layerFile, input_type, 
                                                                           TILING_TYPE::_1D_COL_, compression_type,
                                                                           layer_hasher, repartition));
                }
                else {
                */
                layers[s][i] = std::move(std::make_unique<Tiling<Weight>>(1, 1, 1, 1, 
                                                                       nnz, nrows, ncols, 
                                                                       layerFile, input_type, 
                                                                       TILING_TYPE::_1D_COL_, compression_type,
                                                                       layer_hasher, enable_dual_spmat, false));
                
                //}                
                biasWeightVecs[s][i] = std::move(std::make_shared<struct Data_Block<Weight>>(inputFeatures->ncols, s));
            }
        }    
        Logging::enabled = false; 
        if(i%10==0) printf("|"); 
    }
    Logging::enabled = true;
    Logging::print(Logging::LOG_LEVEL::VOID, "\n"); 
    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Done reading %d layer files.\n", maxLayers); 
    Env::barrier();
    
    for(int32_t s = 0; s < Env::nsockets; s++) {
        if(replication or (s == Env::rank_socket_id)) {
            for(uint32_t i = 0; i < maxLayers; i++) {
                Weight* b_A = biasWeightVecs[s][i]->ptr;
                for(uint32_t i = 0; i < inputFeatures->ncols; i++) {
                    b_A[i] = biasValue;
                }
            }
        }
    }

    spaWeightVec.resize(Env::nthreads);
    for(int32_t i = 0; i < Env::nthreads; i++) {
        if(parallelism_type == PARALLELISM_TYPE::_DATA_X_MODEL_) {
            if(compression_type == COMPRESSED_FORMAT::_CSC_) {
                uint32_t tile_height = (repartition) ? inputFeatures->get_tile_info_max("height") : inputFeatures->get_tile_info("height", 0);
                spaWeightVec[i] = std::move(std::make_shared<struct Data_Block<Weight>>(tile_height, Env::threads_socket_id[i]));    
            }
            else if(compression_type == COMPRESSED_FORMAT::_CSR_) {
                int32_t sid = Env::rank_socket_id;
                uint32_t tile_height = (repartition) ? layers[sid][0]->get_tile_info_max("height") : layers[sid][0]->get_tile_info("height", 0);
                spaWeightVec[i] = std::move(std::make_shared<struct Data_Block<Weight>>(tile_height, Env::threads_socket_id[i]));
            }
            else {
                Logging::print(Logging::LOG_LEVEL::ERROR, "%s compression not implemented\n", COMPRESSED_FORMATS[compression_type]);
                std::exit(Env::finalize());
            }
        }
        else if(parallelism_type == PARALLELISM_TYPE::_HYBRID_X_HYBRID_) {
            if((compression_type == COMPRESSED_FORMAT::_CSC_) or (compression_type == COMPRESSED_FORMAT::_CSR_)) {
                int32_t sid = Env::rank_socket_id;
                uint32_t tile_height1 = (repartition) ? inputFeatures->get_tile_info_max("height") : inputFeatures->get_tile_info("height", 0);
                uint32_t tile_height2 = (repartition) ? layers[sid][0]->get_tile_info_max("height") : layers[sid][0]->get_tile_info("height", 0);
                uint32_t tile_height = (tile_height1 > tile_height2) ? tile_height1 : tile_height2;
                spaWeightVec[i] = std::move(std::make_shared<struct Data_Block<Weight>>(tile_height, Env::threads_socket_id[i]));
            }
            else {
                Logging::print(Logging::LOG_LEVEL::ERROR, "%s compression not implemented\n", COMPRESSED_FORMATS[compression_type]);
                std::exit(Env::finalize());
            }
        }
        else {
            if(compression_type == COMPRESSED_FORMAT::_CSC_) {
                uint32_t tile_height = (repartition) ? inputFeatures->get_tile_info_max("height") : inputFeatures->get_tile_info("height", i);
                spaWeightVec[i] = std::move(std::make_shared<struct Data_Block<Weight>>(tile_height, Env::threads_socket_id[i]));
            }
            else if(compression_type == COMPRESSED_FORMAT::_CSR_) {   
                int32_t sid = Env::rank_socket_id;
                uint32_t tile_height = (repartition) ? layers[sid][0]->get_tile_info_max("height") : layers[sid][0]->get_tile_info("height", 0);
                spaWeightVec[i] = std::move(std::make_shared<struct Data_Block<Weight>>(tile_height, Env::threads_socket_id[i]));
            }
            else {
                Logging::print(Logging::LOG_LEVEL::ERROR, "%s compression not implemented\n", COMPRESSED_FORMATS[compression_type]);
                std::exit(Env::finalize());
            }
        }
    }

    if(parallelism_type == PARALLELISM_TYPE::_DATA_X_MODEL_) {
        output = std::move(std::make_unique<Tiling<Weight>>(Env::nranks, Env::nranks, 1, Env::nranks, 
                                                            0, inputFeatures->nrows, inputFeatures->ncols, 
                                                            TILING_TYPE::_1D_ROW_, compression_type,
                                                            input_hasher, false, repartition));
    }
    else if((parallelism_type == PARALLELISM_TYPE::_HYBRID_X_HYBRID_1_) or (parallelism_type == PARALLELISM_TYPE::_MANAGER_X_WORKER_) or (parallelism_type == PARALLELISM_TYPE::_WORK_X_STEALING_)) {
        output = std::move(std::make_unique<Tiling<Weight>>(Env::nranks * Env::nthreads * split_factor, Env::nranks * Env::nthreads * split_factor, 1, Env::nranks, 
                                                            Env::nthreads, Env::nranks * Env::nthreads, 
                                                            0, inputFeatures->nrows, inputFeatures->ncols, 
                                                            TILING_TYPE::_1D_ROW_, compression_type,
                                                            input_hasher, false, repartition));
    }
    else {
        output = std::move(std::make_unique<Tiling<Weight>>(Env::nranks * Env::nthreads, Env::nranks * Env::nthreads, 1, Env::nranks, 
                                                            Env::nthreads, Env::nranks * Env::nthreads, 
                                                            0, inputFeatures->nrows, inputFeatures->ncols, 
                                                            TILING_TYPE::_1D_ROW_, compression_type,
                                                            input_hasher, false, repartition));
    }
    output->set_tile_info(inputFeatures->tiles);

    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Running the inferenceReLU method [Compression=%s(%d)|Parallelism=%s|Scheduling=%s|Hashing=%s].\n", 
                   COMPRESSED_FORMATS[compression_type], dual_spmat, PARALLELISM_TYPES[parallelism_type], SCHEDULING_TYPES[scheduling_type], HASHING_TYPES[hashing_type]); 
    auto finish = std::chrono::high_resolution_clock::now();
    Env::io_time = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish-start).count())/1e9;
    Env::barrier();
    Env::global_time = Env::tic();
    
    execute();

    finish = std::chrono::high_resolution_clock::now();
    Env::end_to_end_time = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish-start).count())/1e9;
    Env::barrier();
    
    
        if(Env::nranks == 1)
            printTimesExcel();
        else 
            printTimesExcel1();
    
}

void stats(const std::vector<double> vec, double& sum, double& mean, double& std_dev, double& min, double& max) {
    sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    mean = sum / vec.size();
    double sq_sum = std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0);
    std_dev = std::sqrt(sq_sum / vec.size() - mean * mean);
    std::pair bounds = std::minmax_element(vec.begin(), vec.end());
    min = *bounds.first;
    max = *bounds.second;
}

void annotate() {
    /*
   for(auto dc: Env::data_counters) {
        for(auto d: dc) { 
            printf("%f %d %d %d %lu\n", d.time, d.rank, d.tid, d.layer, d.nnz_bytes);
        }
    } 
    return;
    */
    std::vector<struct Env::data_counter> annotated;
    uint32_t s = Env::data_counters[0].size();
    uint32_t sz = s * Env::nthreads;
    uint32_t k = 0;
    std::vector<uint32_t> indices(Env::nthreads, 0);
    while(k < sz) {
        int min_idx = 0;
        double min_val = std::numeric_limits<float>::max();
        for(int i = 0; i < Env::nthreads; i++) {
            if(indices[i] < s && Env::data_counters[i][indices[i]].time < min_val) {
                min_val = Env::data_counters[i][indices[i]].time;
                min_idx = i;
            }
        }
        auto d = Env::data_counters[min_idx][indices[min_idx]];
        //printf("%f %d %d %d\n", d.time, d.rank, d.tid, d.layer);
        annotated.push_back(Env::data_counters[min_idx][indices[min_idx]]);

        indices[min_idx]++;
        k++;
    }
    
    
    //for(auto& a: annotated) {
      //  printf("%d %d %d %d %lu %lu\n", (uint32_t) ceil(a.time/1e6), a.rank, a.tid, a.layer, a.b_bytes, a.c_bytes);
    //}
    
    uint32_t t = (uint32_t) ceil(annotated.front().time/1e6);
    std::vector<int> l;
    l.push_back(annotated.front().layer);
    uint64_t b_nbytes = annotated.front().b_bytes;
    uint64_t c_nbytes = annotated.front().c_bytes;
    int r = 100;
    bool p = true;
    for(uint32_t i = 1; i < annotated.size(); i++) {
        if(i%r == 0) {
            p = true;
        }
        uint32_t t1 = (uint32_t) ceil(annotated[i].time/1e6);
        if(t == t1) {
            //if(std::find(l.begin(), l.end(), annotated[i].layer) == l.end()) {
                l.push_back(annotated[i].layer);
                c_nbytes += annotated[i].c_bytes;
            //}
        }
        if((t != t1) or (i+1 == annotated.size())) {
            std::sort(l.begin(), l.end());
            auto last = std::unique(l.begin(), l.end());
            l.erase(last, l.end());
            //printf("i=%d time=%d b_nbyets=%lu c_bytes=%lu\n", i, t, l.size() * b_nbytes, c_nbytes);
            if(p) {
                printf("%d %lu %lu\n", i, l.size() * b_nbytes, c_nbytes);
                p = false;
            }
            //for(auto ll: l) {printf("%d ", ll);} printf("\n");
            t = (uint32_t) ceil(annotated[i].time/1e6);
            l.clear();
            l.shrink_to_fit();
            l.push_back(annotated[i].layer);
            c_nbytes = annotated[i].c_bytes;
        }
    }
    
    
}

#include <unordered_map>
#include <map>
void annotate1() {
    
    for(uint32_t i = 0; i < Env::data_counters[0].size(); i++) {
        uint64_t b_nbytes = Env::data_counters[0][i].b_bytes;
        uint64_t c_nbytes = Env::data_counters[0][i].c_bytes;
        printf("%d %lu %lu\n", i, b_nbytes, c_nbytes);
    }
}

void annotate2() {
    /*
   for(auto dc: Env::data_counters) {
        for(auto d: dc) { 
            printf("%d %f %d\n", d.tid, d.time, d.layer);
        }
    }
    */
    
    uint32_t s = 0;
    for(int i = 0; i < Env::nthreads; i++) {
        s+= Env::data_counters[i].size();   
    }


    std::vector<struct Env::data_counter> annotated;
    uint32_t sz = s * Env::nthreads;
    uint32_t k = 0;
    std::vector<uint32_t> indices(Env::nthreads, 0);
    while(k < s) {
        int min_idx = 0;
        double min_val = std::numeric_limits<float>::max();
        for(int i = 0; i < Env::nthreads; i++) {
            if(indices[i] < Env::data_counters[i].size() && Env::data_counters[i][indices[i]].time < min_val) {
                min_val = Env::data_counters[i][indices[i]].time;
                min_idx = i;
            }
        }
        auto d = Env::data_counters[min_idx][indices[min_idx]];
        //printf("%f %d %d %d\n", d.time, d.rank, d.tid, d.layer);
        annotated.push_back(Env::data_counters[min_idx][indices[min_idx]]);

        indices[min_idx]++;
        k++;
    }

    /*
    for(uint32_t i = 0; i < annotated.size(); i++) {
        printf("%f %d %d\n", annotated[i].time, annotated[i].tid, annotated[i].layer);
    }
    */
    
    uint32_t t = (uint32_t) ceil(annotated.front().time/1e6);
    std::map<int,int> tids;
    std::map<int,int> tids1;
    
    
    
    uint32_t i=0;
    for(i = 0; i < annotated.size(); i++) {
        double time = annotated[i].time;
        int tid = annotated[i].tid;
        int nhelpers = annotated[i].layer;
        if(tids.size() == (uint32_t) Env::nthreads) {
            tids1.clear();
            tids1 = tids;
            int counts1 = 0;
            
            for(auto t: tids) {
                counts1 += t.second==1;
            }
            
            printf("%f %d %d\n",time, counts1, Env::nthreads-counts1);
            /*
            std::vector<uint32_t> temp;
            for(auto t: tids) {
                temp.push_back(t.second);
            }
            
            std::sort(temp.begin(), temp.end());

            std::vector<int> counts(Env::nthreads+1);
            
            for(uint32_t j = 0; j < temp.size(); j++) {
                counts[temp[j]]++;
            }
            */
            /*
            bool tf = false;
            for(auto t: tids) {
                if(not t.second) {tf = true; break;}
            }
            printf("%d ", i);
            if(tf) {
            for(auto t: tids) {
                printf("%d ", t.second);
            }
            }
            printf("\n");
            */
            /*
            for(uint32_t j = 1; j < counts.size(); j++) {
                //if(counts[j])  printf("%d ", counts[j]);
            }
            */
            
            tids.clear();
        }
        else {
            tids[tid]=nhelpers;
        }
    }
    
    
    if(not tids.empty()) {
        double time = annotated.back().time;
        int counts1 = 0;
        
        for(auto t: tids) {
            counts1 += t.second==1;
        }
        
        printf("%f %d %d\n",time, counts1, Env::nthreads-counts1);
        
        /*
        bool tf = false;
        for(auto t: tids1) {
            if(not t.second) {tf = true; break;}
        }
        printf("%d ", i);
        
        if(tf) {
        
        for(auto t: tids1) {
            printf("%d ", t.second);
        }
        
        }
        printf("\n");
        */
    }
    
    
    
    
    //tids.push_back(annotated.front().tid);
    //uint64_t b_nbytes = annotated.front().b_bytes;
    //uint64_t c_nbytes = annotated.front().c_bytes;
    /*
    uint32_t nhelpers = annoteted.front().layer;
    int r = 100;
    bool p = true;
    for(uint32_t i = 1; i < annotated.size(); i++) {
        if(i%r == 0) {
            p = true;
        }
        uint32_t t1 = (uint32_t) ceil(annotated[i].time/1e6);
        if(t == t1) {
            //if(std::find(l.begin(), l.end(), annotated[i].layer) == l.end()) {
                tids.push_back(annotated[i].tid);
                //c_nbytes += annotated[i].c_bytes;
                nhelpers += annotated[i].layer;
            //}
        }
        if((t != t1) or (i+1 == annotated.size())) {
            std::sort(tids.begin(), tids.end());
            auto last = std::unique(tids.begin(), tids.end());
            tids.erase(last, tids.end());
            //printf("i=%d time=%d b_nbyets=%lu c_bytes=%lu\n", i, t, l.size() * b_nbytes, c_nbytes);
            if(p) {
                printf("%d %lu %lu\n", i, l.size() * b_nbytes, c_nbytes);
                p = false;
            }
            //for(auto ll: l) {printf("%d ", ll);} printf("\n");
            t = (uint32_t) ceil(annotated[i].time/1e6);
            tids.clear();
            tids.shrink_to_fit();
            tids.push_back(annotated[i].tids);
        }
    }
*/
    
}
template<typename Weight>
void Net<Weight>::printTimesExcel() {
    Env::barrier();
    
    double sum = 0.0, mean = 0.0, std_dev = 0.0, min = 0.0, max = 0.0;
    stats(Env::execution_time, sum, mean, std_dev, min, max);
    Logging::print(Logging::LOG_LEVEL::VOID, "exec time: %.3f %.3f %.3f\n", min, max, sum);
    
    //annotate2();
    
    /*
    stats(Env::spmm_symb_time, sum, mean, std_dev, min, max);
    Logging::print(Logging::LOG_LEVEL::VOID, "%.3f %.3f %.3f ", min, max, sum);
    
    stats(Env::spmm_real_time, sum, mean, std_dev, min, max);
    Logging::print(Logging::LOG_LEVEL::VOID, "%.3f %.3f %.3f ", min, max, sum);
    
    stats(Env::memory_allocation_time, sum, mean, std_dev, min, max);
    Logging::print(Logging::LOG_LEVEL::VOID, "%.3f %.3f %.3f ", min, max, sum);

    stats(Env::hybrid_probe_time, sum, mean, std_dev, min, max);
    Logging::print(Logging::LOG_LEVEL::VOID, "%.3f %.3f %.3f\n", min, max, sum);
    */
    /*
    if(Env::nnzs[0].empty()) return;
    
    for(int j = 0; j < Env::nthreads; j++) {
        for(int i = 0; i < 20; i++) {
            printf("%d ", Env::nnzs[j][i]);
        }
        printf("\n");
    }
    printf("\n");
    if(Env::times[0].empty()) return;
    for(int j = 0; j < Env::nthreads; j++) {
        for(int i = 0; i < 20; i++) {
            printf("%f ", Env::times[j][i]/1e9);
        }
        printf("\n");
    }
    */
    /*
    int m = Env::nnzs[0].size();
    for(int i = 0; i < m; i++) {
        std::vector<double> temp;
        for(int j = 0; j < Env::nthreads; j++) {
            temp.push_back(Env::nnzs[j][i]);
        }
        stats(temp, sum, mean, std_dev, min, max);
        Logging::print(Logging::LOG_LEVEL::VOID, "%.3f ", mean);
    }
    Logging::print(Logging::LOG_LEVEL::VOID, "\n\n");
    for(int i = 0; i < m; i++) {
        std::vector<double> temp;
        for(int j = 0; j < Env::nthreads; j++) {
            temp.push_back(Env::nnzs[j][i]);
        }
        stats(temp, sum, mean, std_dev, min, max);
        Logging::print(Logging::LOG_LEVEL::VOID, "%.3f ", std_dev);
    }
    */
    
    /*
    stats(t, sum, mean, std_dev, min, max);
    Logging::print(Logging::LOG_LEVEL::VOID, "Front: mean=%.3f stdev=%.3f sum=%.3f min=%.3f max=%.3f\n", mean, std_dev, sum, min, max);
    
    t.clear();
    t.shrink_to_fit();
    for(int i = 0; i < Env::nthreads; i++) {
        t.push_back(Env::nnzs[i].back());
    }
    stats(t, sum, mean, std_dev, min, max);
    Logging::print(Logging::LOG_LEVEL::VOID, "Front: mean=%.3f stdev=%.3f sum=%.3f min=%.3f max=%.3f\n", mean, std_dev, sum, min, max);
    */
    
    /*
    if(parallelism_type == PARALLELISM_TYPE::_DATA_X_MODEL_) {
        annotate1();
        
    }
    if(parallelism_type == PARALLELISM_TYPE::_DATA_X_DATA_) {
        annotate();
    }
    */
}



template<typename Weight>
void Net<Weight>::printTimesExcel1() {
    Env::barrier();
    
    double sum = 0.0, mean = 0.0, std_dev = 0.0, min = 0.0, max = 0.0;
    /*
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics<double>(Env::io_time);
    Logging::print(Logging::LOG_LEVEL::VOID, "i/o time: mean, std_dev, min, max\n");
    Logging::print(Logging::LOG_LEVEL::VOID, "          %.3f %.3f %.3f %.3f\n", mean, std_dev, min, max);
    */
    int index = std::distance(Env::execution_time.begin(), std::max_element(Env::execution_time.begin(), Env::execution_time.end()));
    double exec_time = Env::execution_time[index];
    
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics<double>(exec_time);
    Logging::print(Logging::LOG_LEVEL::VOID, "Exec time: %.3f %.3f %.3f\n", min, max, sum);
    //return;
    
    
    double spmm_sym_time = Env::spmm_symb_time[index];
    double spmm_time = Env::spmm_real_time[index];
    double memory_time = Env::memory_allocation_time[index];
    double hybrid_time = Env::hybrid_probe_time[index];
    
    /*
    if(exec_time == max) {
        printf("time: %.3f %.3f %.3f %.3f %.3f %.3f\n", exec_time, spmm_sym_time, spmm_time, memory_time, hybrid_time, exec_time-(spmm_sym_time + spmm_time + memory_time + hybrid_time));
    }
    */
    
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
    /*
    Logging::print(Logging::LOG_LEVEL::VOID, "           mean, std_dev, min, max\n");
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics<double>(exec_time);
    double max_exec_time = max;
    Logging::print(Logging::LOG_LEVEL::VOID, "Exec time: %.3f %.3f %.3f %.3f ", mean, std_dev, min, max);
    */
    
    /*
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics<double>(spmm_sym_time);
    Logging::print(Logging::LOG_LEVEL::VOID, "%.3f ", max);
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics<double>(spmm_time);
    Logging::print(Logging::LOG_LEVEL::VOID, "%.3f ", max);
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics<double>(memory_time);
    Logging::print(Logging::LOG_LEVEL::VOID, "%.3f ", max);
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics<double>(hybrid_time);
    Logging::print(Logging::LOG_LEVEL::VOID, "%.3f\n", max);
    */
    
    /*
    //uint64_t DNNedges = inputFeatures->get_info("nedges");
    uint64_t DNNedges = Net::nedges;
    uint64_t DNNConns = NinputInstanses * DNNedges;
    double inference_rate = (double) DNNConns / exec_time;
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics<double>(inference_rate/1e9);
    //Logging::print(Logging::LOG_LEVEL::VOID, "Infe time: %.3f %.3f %.3f %.3f\n", mean, std_dev, min, max);
    Logging::print(Logging::LOG_LEVEL::VOID, "%.3f %.3f %.3f %.3f\n", mean, std_dev, min, max);
    */
    //double min_exec_rate = (double) (NinputInstanses * DNNedges) /max_exec_time;
    //Logging::print(Logging::LOG_LEVEL::VOID, "Run time: %f (sec), run rate: %f (1e9 edges/sec)\n", max_exec_time, min_exec_rate/1e9);
    
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
    if(parallelism_type == PARALLELISM_TYPE::_HYBRID_X_HYBRID_1_) {
        hybrid_x_hybrid_1(tid);
    }
    else if(parallelism_type == PARALLELISM_TYPE::_MANAGER_X_WORKER_) {
        manager_x_worker(tid);
    }    
    else if(parallelism_type == PARALLELISM_TYPE::_WORK_X_STEALING_) {
        work_x_stealing(tid);
    }
}

template<typename Weight>
void Net<Weight>::data_x_model(const int32_t tid) {
    int32_t sid = (replication) ? Env::threads_socket_id[tid] : Env::rank_socket_id;
    uint32_t leader_rowgroup = Env::rank;
    const int32_t leader_tid = 0;
    struct Env::thread_struct& thread_st = Env::threads[tid];

    struct Tile<Weight>& A_tile = inputFeatures->tiles[leader_rowgroup][0];
    struct Tile<Weight>& C_tile = output->tiles[leader_rowgroup][0];
    std::shared_ptr<struct Data_Block<Weight>> s_spa = spaWeightVec[tid];
    std::shared_ptr<struct Data_Block<Weight>> b_bias;
    
    uint32_t nrows = A_tile.spmat->nrows;
    const uint32_t ncols = layers[sid][0]->ncols;
    uint32_t start, end;
    uint32_t sub_start = 0, sub_end = 0;
    
    if(tid == leader_tid) {	
        for(int32_t i = 0; i < Env::nthreads; i++) {	
            Env::threads[i].start_col = ((ncols/Env::nthreads) * i);	
            Env::threads[i].end_col = (i == (Env::nthreads-1)) ? ncols : ((ncols/Env::nthreads) * (i+1));	
            
            Env::threads[i].start_row = ((nrows/Env::nthreads) * i);	
            Env::threads[i].end_row = (i == (Env::nthreads-1)) ? nrows : ((nrows/Env::nthreads) * (i+1));	
        }	
    }	
    pthread_barrier_wait(&Env::thread_barrier);
    
    if(compression_type == COMPRESSED_FORMAT::_CSC_) {
        start = Env::threads[tid].start_col;
        end = Env::threads[tid].end_col;
        sub_start = 0;
        sub_end   = 0;
        
        /*
        B_start = 0;
        B_end = layers[sid][0]->ncols;
        B_sub_start = 0;
        B_sub_end   = 0;
        */
    }
    else if(compression_type == COMPRESSED_FORMAT::_CSR_) {
        start = Env::threads[tid].start_row;
        end = Env::threads[tid].end_row;
        sub_start = 0;
        sub_end   = 0;
        
        /*
        B_start = 0;
        B_end = layers[sid][0]->nrows;
        B_sub_start = layers[sid][0]->start_row;
        B_sub_end   = layers[sid][0]->end_row;
        */
    }
    else {
        Logging::print(Logging::LOG_LEVEL::ERROR, "%s compression not implemented\n", COMPRESSED_FORMATS[compression_type]);
        std::exit(Env::finalize());
    }
    pthread_barrier_wait(&Env::thread_barrier);
    
    auto start_t = std::chrono::high_resolution_clock::now();  
    for (uint32_t l = 0; l < maxLayers; l++) {
        //auto now1 = std::chrono::high_resolution_clock::now();
        std::shared_ptr<struct Compressed_Format<Weight>> A_SPMAT = A_tile.spmat;
        
        //struct Tile<Weight>& B_tile = layers[sid][l]->tiles[0][tid];
        struct Tile<Weight>& B_tile = layers[sid][l]->tiles[0][0];
        
        std::shared_ptr<struct Compressed_Format<Weight>> B_SPMAT = B_tile.spmat;
        std::shared_ptr<struct Compressed_Format<Weight>> C_SPMAT = C_tile.spmat;
        
        b_bias = biasWeightVecs[sid][l];
        nrows = A_SPMAT->nrows;
        
        
        /*
        B_start_col = 0;
        B_end_col = B_CSC->ncols;
        B_sub_start_col = B_tile.start_col;
        B_sub_end_col   = B_tile.end_col;
        */
        
        //B_start_col = Env::threads[tid].start_col;	
        //B_end_col = Env::threads[tid].end_col;	
        //B_sub_start_col = 0;	
        //B_sub_end_col   = 0;
        
        
        data_x_model_1_iter(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, 
                            nrows, ncols, 
                            start, end, 
                            sub_start, sub_end, 
                            thread_st, leader_tid, tid); 
        
        /*
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(now - start).count());
        uint64_t B_SIZE = B_CSC->JA_blk->nbytes + B_CSC->IA_blk->nbytes + B_CSC->A_blk->nbytes;
        uint64_t C_SIZE = C_CSC->JA_blk->nbytes + C_CSC->IA_blk->nbytes + C_CSC->A_blk->nbytes;
        //printf("time %f: Rank=%d tid=%2d layer=%3d nnz=%d B=%lu C=%lu\n", elapsed, Env::rank, tid, l, Env::nnzs[tid][l], B_SIZE, C_SIZE);
        if(tid == leader_tid)
            Env::data_counters[tid].push_back({elapsed, Env::rank, tid, l, (B_CSC->IA_blk->nitems*4*2+ncols*4), (uint64_t) (C_CSC->IA_blk->nitems*4*2+ncols*4)});
        */     

       /*     
        auto now2 = std::chrono::high_resolution_clock::now();
        double elapsed = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(now2 - now1).count());
        Env::times[tid].push_back(elapsed);
        */
        
    }
    auto finish_t = std::chrono::high_resolution_clock::now();
    Env::execution_time[tid] = (double)(std::chrono::duration_cast<std::chrono::nanoseconds>(finish_t - start_t).count())/1e9;

    const std::shared_ptr<struct Compressed_Format<Weight>> A_SPMAT = A_tile.spmat;
    data_x_model_validate_prediction(A_SPMAT, A_tile.start_row, trueCategories, nCategories, leader_tid, tid);
}

template<typename Weight>
void Net<Weight>::data_x_data(const int32_t tid) {
    int32_t sid = (replication) ? Env::threads_socket_id[tid] : Env::rank_socket_id;
    uint32_t leader_rowgroup = Env::thread_rowgroup[tid];
    int32_t leader_tid = 0;
    struct Env::thread_struct& thread_st = Env::threads[tid];
    
    std::shared_ptr<struct Data_Block<Weight>> s_spa = spaWeightVec[tid];
    std::shared_ptr<struct Data_Block<Weight>> b_bias;
    
    uint32_t nrows = inputFeatures->tiles[leader_rowgroup][0].spmat->nrows;
    const uint32_t ncols = layers[sid][0]->ncols;
    uint32_t start, end;
    const uint32_t off = 0;
    
    if(compression_type == COMPRESSED_FORMAT::_CSC_) {
        start = 0;
        end = ncols;
    }
    else if(compression_type == COMPRESSED_FORMAT::_CSR_) {
        start = 0;
        end = nrows;
    }
    
    auto start_t = std::chrono::high_resolution_clock::now();  
    for (uint32_t l = 0; l < maxLayers; l++) {
        //auto now1 = std::chrono::high_resolution_clock::now();
        
        struct Tile<Weight>& A_tile = (not(l%2)) ? inputFeatures->tiles[leader_rowgroup][0]
                                                 : output->tiles[leader_rowgroup][0];
        std::shared_ptr<struct Compressed_Format<Weight>> A_SPMAT = A_tile.spmat;
        struct Tile<Weight>& B_tile = layers[sid][l]->tiles[0][0];    
        std::shared_ptr<struct Compressed_Format<Weight>> B_SPMAT = B_tile.spmat;
        struct Tile<Weight>& C_tile = (not(l%2)) ? output->tiles[leader_rowgroup][0]
                                                 : inputFeatures->tiles[leader_rowgroup][0];
        std::shared_ptr<struct Compressed_Format<Weight>> C_SPMAT = C_tile.spmat;
        b_bias = biasWeightVecs[sid][l];  

        data_x_data_1_iter(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, 
                           nrows, ncols, start, end, off, 
                           thread_st, leader_tid, tid);  
                         
        //auto now = std::chrono::high_resolution_clock::now();
        //double elapsed = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(now - start).count());
        //uint64_t B_SIZE = B_CSC->JA_blk->nbytes + B_CSC->IA_blk->nbytes + B_CSC->A_blk->nbytes;
        //uint64_t C_SIZE = C_CSC->JA_blk->nbytes + C_CSC->IA_blk->nbytes + C_CSC->A_blk->nbytes;
        //printf("time %f: Rank=%d tid=%2d layer=%3d nnz=%d B=%lu C=%lu\n", elapsed, Env::rank, tid, l, Env::nnzs[tid][l], B_SIZE, C_SIZE);
        //Env::data_counters[tid].push_back({elapsed, Env::rank, tid, l, (B_CSC->IA_blk->nitems*4*2+ncols*4), (uint64_t) (C_CSC->IA_blk->nitems*4*2+ncols*4)});
        
        
        /*        
        auto now2 = std::chrono::high_resolution_clock::now();
        double elapsed = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(now2 - now1).count());
        Env::times[tid].push_back(elapsed);
        */
       
    }
    auto finish_t = std::chrono::high_resolution_clock::now();
    Env::execution_time[tid] = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish_t - start_t).count())/1e9;
    
    struct Tile<Weight>& A_tile = inputFeatures->tiles[leader_rowgroup][0];
    const std::shared_ptr<struct Compressed_Format<Weight>> A_SPMAT = A_tile.spmat;
    data_x_data_validate_prediction(A_SPMAT, A_tile.start_row, trueCategories, nCategories, leader_tid, tid);
}

template<typename Weight>
void Net<Weight>::hybrid_x_hybrid_1(const int32_t tid) {
    int32_t leader_tid = 0;
    bool has_dual_spmat = false;
    auto start_t = std::chrono::high_resolution_clock::now();  
    for(uint32_t my_rowgroup: Env::threads_rowgroups[tid]) {
        Env::processed_rowgroups_per_thread[tid].push_back(my_rowgroup);
        //printf("hybrid_x_hybrid_1: Rank=%d tid=%d rowgroup=%d %lu\n", Env::rank, tid, my_rowgroup, Env::my_threads[tid].size());
        //printf("0.tid=%d, my_rowgroup=%d\n", tid, my_rowgroup);
        uint32_t my_start_layer = hybrid_x_data(Env::my_threads[tid], my_rowgroup, tid);
        //printf("1.tid=%d my_start_layer=%d\n", tid, my_start_layer);
        
        if(my_start_layer < maxLayers) {
            if(dual_spmat) {      
                has_dual_spmat = true;
                if(compression_type == COMPRESSED_FORMAT::_CSC_) {
                    struct Tile<Weight>& A_tile = inputFeatures->tiles[my_rowgroup][0];
                    struct Tile<Weight>& C_tile = output->tiles[my_rowgroup][0];
                    A_tile.spmat1 = std::make_shared<struct CSR<Weight>>(A_tile.spmat, A_tile.start_row, A_tile.end_row, A_tile.start_col, A_tile.end_col, Env::threads_socket_id[tid]);
                    C_tile.spmat1 = std::make_shared<struct CSR<Weight>>(0, C_tile.height, C_tile.width, Env::threads_socket_id[tid]);
                }
                else if(compression_type == COMPRESSED_FORMAT::_CSR_) {
                    struct Tile<Weight>& A_tile = inputFeatures->tiles[my_rowgroup][0];
                    struct Tile<Weight>& C_tile = output->tiles[my_rowgroup][0];
                    
                    A_tile.spmat1 = std::make_shared<struct CSC<Weight>>(A_tile.spmat, A_tile.start_row, A_tile.end_row, A_tile.start_col, A_tile.end_col, Env::threads_socket_id[tid]);
                    C_tile.spmat1 = std::make_shared<struct CSC<Weight>>(0, C_tile.height, C_tile.width, Env::threads_socket_id[tid]);
                }
            }
            hybrid_x_model(Env::my_threads[tid], my_rowgroup, my_start_layer, tid, tid);
        }
       // printf("2.tid=%d num_threads=%lu\n", tid, Env::my_threads[tid].size());
        while(add_to_idle_threads(Env::my_threads[tid], tid)) {
            
            const int32_t leader = Env::threads[tid].leader;
            uint32_t leader_rowgroup = Env::threads[tid].rowgroup;
            uint32_t leader_start_layer = Env::threads[tid].start_layer;
            //printf(">>tid=%d leader=%d leader_rowgroup=%d leader_start_layer=%d\n", tid, leader, leader_rowgroup, leader_start_layer);
            hybrid_x_model(Env::my_threads[tid], leader_rowgroup, leader_start_layer, leader, tid);
        }
        
        pthread_barrier_wait(&Env::thread_barrier);
        if(tid == 0) {
            for(int32_t s = 0; s < Env::nsockets; s++) {
                pthread_mutex_lock(&Env::numa_thread_mutex[s]);
                Env::numa_follower_threads[s].clear();
                Env::numa_follower_threads[s].shrink_to_fit();
                pthread_mutex_unlock(&Env::numa_thread_mutex[s]);  
            }
        }
        //printf("3.tid=%d num_threads=%lu\n", tid, Env::my_threads[tid].size());
    }
    
    auto finish_t = std::chrono::high_resolution_clock::now();
    Env::execution_time[tid] = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish_t - start_t).count())/1e9;

    work_x_stealing_validate_prediction(inputFeatures->tiles, trueCategories, nCategories, leader_tid, tid);
    
    
}

template<typename Weight>
void Net<Weight>::hybrid_x_hybrid(const int32_t tid) {
    uint32_t my_rowgroup = Env::thread_rowgroup[tid];
    int32_t leader_tid = 0;
    bool has_dual_spmat = false;
    Env::global_time = Env::tic();
    auto start_t = std::chrono::high_resolution_clock::now();  
    
    uint32_t my_start_layer = hybrid_x_data(Env::my_threads[tid], my_rowgroup, tid);
    if(my_start_layer < maxLayers) {
        if(dual_spmat) {      
            has_dual_spmat = true;
            if(compression_type == COMPRESSED_FORMAT::_CSC_) {
                struct Tile<Weight>& A_tile = inputFeatures->tiles[my_rowgroup][0];
                struct Tile<Weight>& C_tile = output->tiles[my_rowgroup][0];
                A_tile.spmat1 = std::make_shared<struct CSR<Weight>>(A_tile.spmat, A_tile.start_row, A_tile.end_row, A_tile.start_col, A_tile.end_col, Env::threads_socket_id[tid]);
                C_tile.spmat1 = std::make_shared<struct CSR<Weight>>(0, C_tile.height, C_tile.width, Env::threads_socket_id[tid]);
            }
            else if(compression_type == COMPRESSED_FORMAT::_CSR_) {
                struct Tile<Weight>& A_tile = inputFeatures->tiles[my_rowgroup][0];
                struct Tile<Weight>& C_tile = output->tiles[my_rowgroup][0];
                
                A_tile.spmat1 = std::make_shared<struct CSC<Weight>>(A_tile.spmat, A_tile.start_row, A_tile.end_row, A_tile.start_col, A_tile.end_col, Env::threads_socket_id[tid]);
                C_tile.spmat1 = std::make_shared<struct CSC<Weight>>(0, C_tile.height, C_tile.width, Env::threads_socket_id[tid]);
            }
        }
        hybrid_x_model(Env::my_threads[tid], my_rowgroup, my_start_layer, tid, tid);
    }

    while(add_to_idle_threads(Env::my_threads[tid], tid)) {
        const int32_t leader = Env::threads[tid].leader;
        uint32_t leader_rowgroup = Env::threads[tid].rowgroup;
        uint32_t leader_start_layer = Env::threads[tid].start_layer;
        hybrid_x_model(Env::my_threads[tid], leader_rowgroup, leader_start_layer, leader, tid);
    }
    
    auto finish_t = std::chrono::high_resolution_clock::now();
    Env::execution_time[tid] = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish_t - start_t).count())/1e9;

    struct Tile<Weight>& A_tile = inputFeatures->tiles[my_rowgroup][0];
    std::shared_ptr<struct Compressed_Format<Weight>> A_SPMAT;    
    if(has_dual_spmat) {
        A_SPMAT = A_tile.spmat1;
    }
    else {
        A_SPMAT = A_tile.spmat;
    }
    data_x_data_validate_prediction(A_SPMAT, A_tile.start_row, trueCategories, nCategories, leader_tid, tid);
}

template<typename Weight>
uint32_t Net<Weight>::hybrid_x_data(std::deque<int32_t>& my_threads, const int32_t my_rowgroup, const int32_t tid) {
    int32_t sid = (replication) ? Env::threads_socket_id[tid] : Env::rank_socket_id;
    int32_t sid1 = (numa_queues) ? Env::threads_socket_id[tid] : Env::rank_socket_id;
    
    //uint32_t my_rowgroup = Env::thread_rowgroup[tid];
    int32_t leader_tid = 0;
    struct Env::thread_struct& thread_st = Env::threads[tid];
    
    std::shared_ptr<struct Data_Block<Weight>> s_spa = spaWeightVec[tid];
    std::shared_ptr<struct Data_Block<Weight>> b_bias;
    
    uint32_t nrows = inputFeatures->tiles[my_rowgroup][0].spmat->nrows;
    const uint32_t ncols = layers[sid][0]->ncols;
    uint32_t start, end;
    const uint32_t off = 0;
    uint32_t start_row = 0;
    if(compression_type == COMPRESSED_FORMAT::_CSC_) {
        start = 0;
        end = ncols;
    }
    else if(compression_type == COMPRESSED_FORMAT::_CSR_) {
        start = 0;
        end = nrows;
    }
    bool breaking = false;
    uint32_t l = 0; 
    for (l = 0; l < maxLayers; l++) {
        if((l >= maxLayers*recruiting_ratio) and add_to_my_follower_threads(my_threads, my_rowgroup, l, 0, nrows, ncols, tid, tid)) {
            if(not(l%2)) 
                break;
            else     
                breaking = true;
        }
        struct Tile<Weight>& A_tile = (not(l%2)) ? inputFeatures->tiles[my_rowgroup][0]
                                             : output->tiles[my_rowgroup][0];
        std::shared_ptr<struct Compressed_Format<Weight>> A_SPMAT = A_tile.spmat;
        struct Tile<Weight>& B_tile = layers[sid][l]->tiles[0][0];    
        std::shared_ptr<struct Compressed_Format<Weight>> B_SPMAT = B_tile.spmat;
        struct Tile<Weight>& C_tile = (not(l%2)) ? output->tiles[my_rowgroup][0]
                                                 : inputFeatures->tiles[my_rowgroup][0];
        std::shared_ptr<struct Compressed_Format<Weight>> C_SPMAT = C_tile.spmat;
        b_bias = biasWeightVecs[sid][l];      
        
        data_x_data_1_iter(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, 
                           nrows, ncols, start, end, off, 
                           thread_st, leader_tid, tid); 
        
        Env::scores[sid1][tid]++;     

        double elapsed = Env::toc(Env::global_time);
        Env::data_counters[tid].push_back({elapsed, Env::rank, tid, 1, 0,0});        
        
        if(breaking) break;
    }
    return(l);
}

template<typename Weight>
void Net<Weight>::hybrid_x_model(std::deque<int32_t>& my_threads, const uint32_t my_rowgroup, const uint32_t leader_start_layer, const int32_t leader_tid, const int32_t tid) {
    int32_t sid = (replication) ? Env::threads_socket_id[tid] : Env::rank_socket_id;
    int32_t sid1 = (numa_queues) ? Env::threads_socket_id[tid] : Env::rank_socket_id;
    struct Env::thread_struct& thread_st = Env::threads[tid];    
    struct Tile<Weight>& A_tile =  inputFeatures->tiles[my_rowgroup][0];
    struct Tile<Weight>& C_tile = output->tiles[my_rowgroup][0];
    std::shared_ptr<struct Data_Block<Weight>> s_spa = spaWeightVec[tid];
    std::shared_ptr<struct Data_Block<Weight>> b_bias;
    
    std::shared_ptr<struct Compressed_Format<Weight>> A_SPMAT = nullptr;
    std::shared_ptr<struct Compressed_Format<Weight>> B_SPMAT = nullptr;
    std::shared_ptr<struct Compressed_Format<Weight>> C_SPMAT = nullptr;
    
    uint32_t nrows = A_tile.spmat->nrows;
    const uint32_t ncols = layers[sid][0]->ncols;
    uint32_t start, end;
    const uint32_t off = 0;   
    double start_time = 0;
    for (uint32_t l = leader_start_layer; l < maxLayers; l++) {
        (void)add_to_my_follower_threads(my_threads, my_rowgroup, l, 0, nrows, ncols, leader_tid, tid);
        start_time = Env::tic();   
            Env::decrease_num_threads(1, leader_tid, tid);
            Env::init_num_threads(my_threads.size(), leader_tid, tid);
        Env::hybrid_probe_time[tid] += Env::toc(start_time);     

        struct Tile<Weight>& B_tile = layers[sid][l]->tiles[0][0];
        
        b_bias = biasWeightVecs[sid][l];
        if(dual_spmat) {
            A_SPMAT = A_tile.spmat1;
            B_SPMAT = B_tile.spmat1;
            C_SPMAT = C_tile.spmat1;
            if(compression_type == COMPRESSED_FORMAT::_CSC_) {
                start = Env::threads[tid].start_row;
                end = Env::threads[tid].end_row;
            }
            else if(compression_type == COMPRESSED_FORMAT::_CSR_) {
                start = Env::threads[tid].start_col;
                end = Env::threads[tid].end_col;
            }
        }
        else {
            if(compression_type == COMPRESSED_FORMAT::_CSC_) {
                A_SPMAT = A_tile.spmat;
                B_SPMAT = B_tile.spmat;
                C_SPMAT = C_tile.spmat;
                start = Env::threads[tid].start_col;
                end = Env::threads[tid].end_col;
            }
            else if(compression_type == COMPRESSED_FORMAT::_CSR_) {
                A_SPMAT = A_tile.spmat;
                B_SPMAT = B_tile.spmat;
                C_SPMAT = C_tile.spmat;
                start = Env::threads[tid].start_row;
                end = Env::threads[tid].end_row;
            }
        }
   // printf("3.tid=%d %d leader_rowgroup=%d leader=%d \n", tid, l, my_rowgroup, leader_tid);
        data_x_model_hybrid_1_iter(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, 
               nrows, ncols, start, end, off,
               my_threads, thread_st, leader_tid, tid);
     //    printf("4.tid=%d %d\n", tid, l);                          
       if(tid == leader_tid) Env::scores[sid1][tid]++;
       
       double elapsed = Env::toc(Env::global_time);
       if(tid == leader_tid) {
         
            Env::data_counters[tid].push_back({elapsed, Env::rank, tid, (uint32_t)my_threads.size(), 0,0});        
       }
       else {
            Env::data_counters[tid].push_back({elapsed, Env::rank, tid, 0, 0,0});          
       }
       
    }
}

template<typename Weight>
bool Net<Weight>::thread_scheduling(std::deque<int32_t>& my_threads, const uint32_t my_rowgroup, std::deque<int32_t>& follower_threads, int32_t socket_id, const uint32_t start_layer, const uint32_t start_row, const uint32_t nrows, const uint32_t ncols, const int32_t leader, const int32_t tid) {  
    bool found = false;
    uint32_t num_threads = my_threads.size();
    uint32_t old_num_threads = my_threads.size();
    uint32_t num_new_threads = 0;
    uint32_t min_score_value = (uint32_t) INT32_MAX;
    uint32_t max_score_value = 0;
    int32_t sid1 = (numa_queues) ? Env::threads_socket_id[tid] : Env::rank_socket_id;
    
    if(!follower_threads.empty()) {
        pthread_mutex_lock(&Env::numa_thread_mutex[socket_id]);  
        if(!follower_threads.empty()) {
            if((sid1 == socket_id) and ((scheduling_type == SCHEDULING_TYPE::_SLOWER_FIRST_) or (scheduling_type == SCHEDULING_TYPE::_FASTER_FIRST_))) { 
                for(std::vector<uint32_t>::iterator it = Env::scores[socket_id].begin() + Env::queue_indices[socket_id].first ; it != Env::scores[socket_id].begin() + Env::queue_indices[socket_id].second ; it++) {
                    if((*it >= maxLayers) or (*it == 0)) continue;
                    
                    if(*it > max_score_value) {
                        max_score_value = *it;
                    }
                    if(*it < min_score_value) {
                        min_score_value = *it;
                    }
                    
                }
            }
            
            bool pick = (((sid1 == socket_id) and ((scheduling_type == SCHEDULING_TYPE::_EARLIEST_FIRST_) or 
                                                   ((scheduling_type == SCHEDULING_TYPE::_SLOWER_FIRST_) and ((Env::scores[socket_id][tid] - min_score_value) < schduling_threshold)) or
                                                   ((scheduling_type == SCHEDULING_TYPE::_FASTER_FIRST_) and ((max_score_value  - Env::scores[socket_id][tid]) < schduling_threshold)))) or
                         (sid1 != socket_id));
            //printf("Rank=%d tid=%2d layer=%d min=%d max=%d pick=%d\n", Env::rank, tid, start_layer, min_score_value, max_score_value, pick);
            //uint32_t nworking = Env::nthreads_per_socket[socket_id] - Env::numa_num_finished_threads[socket_id];
            //uint32_t nfinished = Env::numa_num_finished_threads[socket_id];
            //uint32_t nhelping = Env::numa_num_finished_threads[socket_id] - follower_threads.size();
            //uint32_t nidles = follower_threads.size();
            //printf("Rank=%d tid=%2d layer=%d nworking=%d nfinished=%d nhelping=%d nidles=%d min=%d max=%d pick=%d\n", Env::rank, tid, start_layer, nworking, nfinished, nhelping, nidles, min_score_value, max_score_value, pick);
            if(pick) {
                if(my_threads.empty()) {
                    my_threads.push_back(tid);
                    num_threads++;
                }

                num_threads += follower_threads.size();
                my_threads.insert(my_threads.end(), follower_threads.begin(), follower_threads.end());
                follower_threads.erase(follower_threads.begin(), follower_threads.end());
                
                if((dual_spmat and compression_type == COMPRESSED_FORMAT::_CSC_) or (not dual_spmat and compression_type == COMPRESSED_FORMAT::_CSR_)) {
                    for(uint32_t i = 0; i < num_threads; i++) {
                        int32_t t = my_threads[i];
                        Env::threads[t].index = i;
                        Env::threads[t].leader = tid;
                        Env::threads[t].rowgroup = my_rowgroup;
                        Env::threads[t].start_layer = start_layer;
                        Env::threads[t].start_row = start_row + ((nrows/num_threads) * i);
                        Env::threads[t].end_row   = (i == (num_threads-1)) ? start_row+nrows : start_row+((nrows/num_threads) * (i+1));
                    }                     
                }
                else if((dual_spmat and compression_type == COMPRESSED_FORMAT::_CSR_) or (not dual_spmat and compression_type == COMPRESSED_FORMAT::_CSC_)) {
                    for(uint32_t i = 0; i < num_threads; i++) {
                        int32_t t = my_threads[i];
                        Env::threads[t].index = i;
                        Env::threads[t].leader = tid;
                        Env::threads[t].rowgroup = my_rowgroup;
                        Env::threads[t].start_layer = start_layer;
                        Env::threads[t].start_col = ((ncols/num_threads) * i);
                        Env::threads[t].end_col   = (i == (num_threads-1)) ? ncols : ((ncols/num_threads) * (i+1));
                    }
                }                    
                pthread_barrier_destroy(&Env::thread_barriers[tid]);
                pthread_barrier_init(&Env::thread_barriers[tid], NULL, num_threads);

                num_new_threads = num_threads - old_num_threads;
                Env::increase_num_threads(num_new_threads, leader, tid);
                
                pthread_cond_broadcast(&Env::numa_thread_cond[socket_id]); 
                
                found = true;
            }
        }
        pthread_mutex_unlock(&Env::numa_thread_mutex[socket_id]);
    }
    return(found);
}


template<typename Weight>
bool Net<Weight>::add_to_my_follower_threads(std::deque<int32_t>& my_threads, const uint32_t my_rowgroup, const uint32_t start_layer, const uint32_t start_row, const uint32_t nrows, const uint32_t ncols, const int32_t leader_tid, const int32_t tid) {  
    bool found = false;
    if(tid == leader_tid) {
        double start_time = 0;
        start_time = Env::tic();
        int32_t sid1 = (numa_queues) ? Env::threads_socket_id[tid] : Env::rank_socket_id;
        if(numa_queues) {
            for(int32_t s = 0; s < Env::nsockets; s++) {
                int32_t si = (s + Env::threads_socket_id[tid]) % Env::nsockets;
                if((si == sid1) or (Env::nthreads_per_socket[si] and (Env::numa_follower_threads[si].size() == (uint32_t) Env::nthreads_per_socket[si]))) {
                    found |= thread_scheduling(my_threads, my_rowgroup, Env::numa_follower_threads[si], si, start_layer, start_row, nrows, ncols, leader_tid, tid);
                }
            }
        }
        else {
            found = thread_scheduling(my_threads, my_rowgroup, Env::numa_follower_threads[sid1], sid1, start_layer, start_row, nrows, ncols, leader_tid, tid);
        }
        Env::hybrid_probe_time[tid] += Env::toc(start_time);  
    }

    return(found);
}

template<typename Weight>
bool Net<Weight>::add_to_idle_threads(std::deque<int32_t>& my_threads, const int32_t tid) {
    uint32_t sid1 = (numa_queues) ? Env::threads_socket_id[tid] : Env::rank_socket_id;
    bool status = true;
    uint32_t all_done = 0;

    pthread_mutex_lock(&Env::numa_thread_mutex[sid1]);
    Env::numa_follower_threads[sid1].push_back(tid);
    if(not my_threads.empty()) {
        my_threads.erase(my_threads.begin(), my_threads.end());
    }
    Env::threads[tid].leader = -1;
        
    for(std::deque<int32_t>& numa_thread: Env::numa_follower_threads) {
        all_done += numa_thread.size();
    }
    

    
    if(all_done == (uint32_t) Env::nthreads) {         
        pthread_mutex_unlock(&Env::numa_thread_mutex[sid1]);   
        for(int32_t s = 0; s < Env::nsockets; s++) {
            pthread_mutex_lock(&Env::numa_thread_mutex[s]);
            
            
            pthread_cond_broadcast(&Env::numa_thread_cond[s]);   
            pthread_mutex_unlock(&Env::numa_thread_mutex[s]);  
        }

        
        status = false;
    }
    else {
        pthread_cond_wait(&Env::numa_thread_cond[sid1], &Env::numa_thread_mutex[sid1]); 
        pthread_mutex_unlock(&Env::numa_thread_mutex[sid1]); 
        
        all_done = 0;
        for(std::deque<int32_t>& numa_thread: Env::numa_follower_threads) {
            all_done += numa_thread.size();
        }
        
        if(all_done == (uint32_t) Env::nthreads) {
            status = false;
        }

    }
    
    
    return(status);
}


template<typename Weight>
void Net<Weight>::manager_x_worker(const int32_t tid) {
    int32_t sid = (replication) ? Env::threads_socket_id[tid] : Env::rank_socket_id;
    uint32_t leader_rowgroup = 0;
    int32_t leader_tid = 0;
    struct Env::thread_struct& thread_st = Env::threads[tid];
    int32_t last_follower_rank = -1;
    int32_t last_follower_thread = -1;
    
    std::shared_ptr<struct Data_Block<Weight>> s_spa = spaWeightVec[tid];
    std::shared_ptr<struct Data_Block<Weight>> b_bias;
    
    uint32_t nrows;
    const uint32_t ncols = layers[sid][0]->ncols;
    uint32_t start, end;
    const uint32_t off = 0;
    auto start_t = std::chrono::high_resolution_clock::now();  
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
        
        for (uint32_t l = 0; l < maxLayers; l++) {
            struct Tile<Weight>& A_tile = (not(l%2)) ? inputFeatures->tiles[leader_rowgroup][0]
                                                     : output->tiles[leader_rowgroup][0];
            std::shared_ptr<struct Compressed_Format<Weight>> A_SPMAT = A_tile.spmat;
            struct Tile<Weight>& B_tile = layers[sid][l]->tiles[0][0];    
            std::shared_ptr<struct Compressed_Format<Weight>> B_SPMAT = B_tile.spmat;
            struct Tile<Weight>& C_tile = (not(l%2)) ? output->tiles[leader_rowgroup][0]
                                                     : inputFeatures->tiles[leader_rowgroup][0];
            std::shared_ptr<struct Compressed_Format<Weight>> C_SPMAT = C_tile.spmat;
            b_bias = biasWeightVecs[sid][l];    
            
            nrows = A_SPMAT->nrows;
            if(compression_type == COMPRESSED_FORMAT::_CSC_) {
                start = 0;
                end = B_SPMAT->ncols;
            }
            else if(compression_type == COMPRESSED_FORMAT::_CSR_) {
                start = 0;
                end = A_SPMAT->nrows;
            }
            
            data_x_data_1_iter(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, 
                               nrows, ncols, start, end, off, 
                               thread_st, leader_tid, tid);       
        }   
    }
    auto finish_t = std::chrono::high_resolution_clock::now();
    Env::execution_time[tid] = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish_t - start_t).count())/1e9;

    manager_x_worker_validate_prediction(inputFeatures->tiles, trueCategories, nCategories, leader_tid, tid);
}


template<typename Weight>
void Net<Weight>::work_x_stealing(const int32_t tid) {
    int32_t sid = (replication) ? Env::threads_socket_id[tid] : Env::rank_socket_id;
    uint32_t leader_rowgroup = 0;
    int32_t leader_tid = 0;
    struct Env::thread_struct& thread_st = Env::threads[tid];
    
    std::shared_ptr<struct Data_Block<Weight>> s_spa = spaWeightVec[tid];
    std::shared_ptr<struct Data_Block<Weight>> b_bias;
    
    uint32_t nrows;
    const uint32_t ncols = layers[sid][0]->ncols;
    uint32_t start, end;
    const uint32_t off = 0;
    
    bool tiles_left = true;
    auto start_t = std::chrono::high_resolution_clock::now();  
    while(tiles_left) {
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
        tiles_left &= found;
        if(not tiles_left) continue;
        
        for (uint32_t l = 0; l < maxLayers; l++) {
            struct Tile<Weight>& A_tile = (not(l%2)) ? inputFeatures->tiles[leader_rowgroup][0]
                                                     : output->tiles[leader_rowgroup][0];
            std::shared_ptr<struct Compressed_Format<Weight>> A_SPMAT = A_tile.spmat;
            struct Tile<Weight>& B_tile = layers[sid][l]->tiles[0][0];    
            std::shared_ptr<struct Compressed_Format<Weight>> B_SPMAT = B_tile.spmat;
            struct Tile<Weight>& C_tile = (not(l%2)) ? output->tiles[leader_rowgroup][0]
                                                     : inputFeatures->tiles[leader_rowgroup][0];
            std::shared_ptr<struct Compressed_Format<Weight>> C_SPMAT = C_tile.spmat;
            b_bias = biasWeightVecs[sid][l];  
            
            nrows = A_SPMAT->nrows;
            if(compression_type == COMPRESSED_FORMAT::_CSC_) {
                start = 0;
                end = B_SPMAT->ncols;
            }
            else if(compression_type == COMPRESSED_FORMAT::_CSR_) {
                start = 0;
                end = A_SPMAT->nrows;
            }
            
            data_x_data_1_iter(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, 
                               nrows, ncols, start, end, off, 
                               thread_st, leader_tid, tid);       
        }   
    }

    auto finish_t = std::chrono::high_resolution_clock::now();
    Env::execution_time[tid] = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish_t - start_t).count())/1e9;

    work_x_stealing_validate_prediction(inputFeatures->tiles, trueCategories, nCategories, leader_tid, tid);
}

#endif 
