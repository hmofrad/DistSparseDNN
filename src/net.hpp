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

//template<typename Weight>
//Weight activation_function(Weight w) {return (w < 0) ? 0 : (w > 32) ? 32 : w;}

/* Input x layers */
enum PARALLELISM_TYPE {_DATA_X_MODEL_, _DATA_X_DATA_, _HYBRID_X_HYBRID_, _MANAGER_X_WORKER_, _WORK_X_STEALING_, _SIZE_};
const char* PARALLELISM_TYPES[] = {"_DATA_X_MODEL_", "_DATA_X_DATA_", "_HYBRID_X_HYBRID_", "_MANAGER_X_WORKER_", "_WORK_X_STEALING_"};

enum SCHEDULING_TYPE {_EARLIEST_FIRST_, _SLOWER_FIRST_, _FASTER_FIRST_, _NONE_};
const char* SCHEDULING_TYPES[] = {"_EARLIEST_FIRST_", "_SLOWER_FIRST_", "_FASTER_FIRST_", "_NONE_"};

template<typename Weight>
class Net {
    public:
        Net() {};
        ~Net() {};
        
        Net(const uint32_t input_ninstanses_, const uint32_t input_nfeatures_, const std::string feature_file,
			const uint32_t nneurons_, const uint32_t nmax_layers_, const  std::vector<std::string> layer_files,
            const Weight bias_value, const VALUE_TYPE bias_type, const std::vector<std::string> bias_files,
			const uint32_t ncategories, const VALUE_TYPE category_type_, const  std::string category_file, 
			Weight(*activation_function_)(Weight),
			const INPUT_TYPE input_type = INPUT_TYPE::_BINARY_,
            const PARALLELISM_TYPE parallelism_type_  = PARALLELISM_TYPE::_HYBRID_X_HYBRID_,
            const COMPRESSED_FORMAT compression_type_ = COMPRESSED_FORMAT::_CSR_,
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
		uint32_t nneurons = 0;
        uint32_t nmax_layers = 0;
        uint32_t ncategories = 0;
		VALUE_TYPE category_type = VALUE_TYPE::_NONZERO_INSTANCES_ONLY_;
		
		Weight (*activation_function)(Weight);
		
		uint32_t predicted_nistances;
        
        PARALLELISM_TYPE parallelism_type = PARALLELISM_TYPE::_HYBRID_X_HYBRID_;
        SCHEDULING_TYPE scheduling_type = _SLOWER_FIRST_;

        uint32_t split_factor = 8;
        bool numa_queues = true;
        uint32_t schduling_threshold = 4;
        
        COMPRESSED_FORMAT compression_type = COMPRESSED_FORMAT::_CSC_;
        //bool dual_spmat = false;
        float recruiting_ratio = .3;
        
        HASHING_TYPE hashing_type = HASHING_TYPE::_BOTH_; 
		std::vector<std::shared_ptr<struct TwoDHasher>> hashers;
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
        void manager_x_worker(const int32_t tid);
        void work_x_stealing(const int32_t tid);
        uint32_t hybrid_x_data(std::deque<int32_t>& my_threads, const int32_t my_rowgroup, const int32_t tid);
		void hybrid_x_model(std::deque<int32_t>& leader_owned_threads, const uint32_t leader_rowgroup, const uint32_t leader_start_layer, const int32_t leader_tid, const int32_t tid);
        bool add_to_idle_threads(std::deque<int32_t>& my_threads, const int32_t tid);
        bool add_to_my_follower_threads(std::deque<int32_t>& leader_owned_threads, const uint32_t leader_rowgroup, const uint32_t start_layer, const uint32_t start_row, const uint32_t nrows, const uint32_t ncols, const int32_t leader, const int32_t tid);
        bool thread_scheduling(std::deque<int32_t>& leader_owned_threads, const uint32_t leader_rowgroup, std::deque<int32_t>& follower_threads, int32_t socket_id, const uint32_t start_layer, const uint32_t start_row, const uint32_t nrows, const uint32_t ncols, const int32_t leader, const int32_t tid);
};



template<typename Weight>
Net<Weight>::Net(const uint32_t input_ninstanses_, const uint32_t input_nfeatures_, const std::string feature_file,
				 const uint32_t nneurons_, const uint32_t nmax_layers_, const std::vector<std::string> layer_files,
				 const Weight bias_value, const VALUE_TYPE bias_type, const std::vector<std::string> bias_files,
				 const uint32_t ncategories_, const VALUE_TYPE category_type_, const std::string category_file, 
				 Weight(*activation_function_)(Weight),
				 const INPUT_TYPE input_type, const PARALLELISM_TYPE parallelism_type_, 
				 const COMPRESSED_FORMAT compression_type_, const HASHING_TYPE hashing_type_)
				     : input_ninstanses(input_ninstanses_), input_nfeatures(input_nfeatures_), 
					   nneurons(nneurons_), nmax_layers(nmax_layers_), ncategories(ncategories_), category_type(category_type_),
					   activation_function(activation_function_),
					   parallelism_type(parallelism_type_), compression_type(compression_type_), hashing_type(hashing_type_) {
    auto start = std::chrono::high_resolution_clock::now();
	input_ninstanses+=2;
	input_ninstanses += (input_ninstanses % Env::nthreads) ? (Env::nthreads - (input_ninstanses % Env::nthreads)) : 0; 
	input_nfeatures+=2;
	input_nfeatures += (input_nfeatures % Env::nthreads) ? (Env::nthreads - (input_nfeatures % Env::nthreads)) : 0; 
	nneurons+=2;
	nneurons += (nneurons % Env::nthreads) ? (Env::nthreads - (nneurons % Env::nthreads)) : 0; 
	scheduling_type = (parallelism_type != PARALLELISM_TYPE::_HYBRID_X_HYBRID_) ? SCHEDULING_TYPE::_NONE_ : scheduling_type;
    hashers.push_back(std::move(std::make_shared<struct TwoDHasher>(hashing_type, true, input_ninstanses, input_nfeatures, 1, 1)));
    input_nnzs = IO::get_nnzs<Weight>(feature_file, input_type, hashers[0], input_ninstanses);
	 
	
    if(parallelism_type == PARALLELISM_TYPE::_DATA_X_MODEL_) {
        input_features = std::move(std::make_unique<Tiling<Weight>>(Env::nranks, Env::nranks, 1, Env::nranks, 
                                                                   input_nnzs, input_ninstanses, input_nfeatures, 
                                                                   feature_file, input_type, 
                                                                   TILING_TYPE::_1D_ROW_, compression_type, hashers[0]));
    }
    else if((parallelism_type == PARALLELISM_TYPE::_MANAGER_X_WORKER_) or (parallelism_type == PARALLELISM_TYPE::_WORK_X_STEALING_)) {
        input_features = std::move(std::make_unique<Tiling<Weight>>(Env::nranks * Env::nthreads * split_factor, Env::nranks * Env::nthreads * split_factor, 1, Env::nranks,
                                                                   Env::nthreads, Env::nranks * Env::nthreads, 
                                                                   input_nnzs, input_ninstanses, input_nfeatures, 
                                                                   feature_file, input_type, 
                                                                   TILING_TYPE::_1D_ROW_, compression_type, hashers[0]));
       Env::threads_rowgroups = input_features->set_threads_indices();
       Env::rank_rowgroups = input_features->set_rank_indices();  
    }
    else {
        input_features = std::move(std::make_unique<Tiling<Weight>>(Env::nranks * Env::nthreads, Env::nranks * Env::nthreads, 1, Env::nranks,
                                                                   Env::nthreads, Env::nranks * Env::nthreads, 
                                                                   input_nnzs, input_ninstanses, input_nfeatures, 
                                                                   feature_file, input_type, 
                                                                   TILING_TYPE::_1D_ROW_, compression_type, hashers[0]));
        Env::thread_rowgroup = input_features->set_thread_index();                                                           
    }
	
    input_ninstanses = input_features->nrows;
	input_nfeatures = input_features->ncols;
    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Processing the category files for %d neurons and %d layers.\n", nneurons, nmax_layers); 
	predicted_nistances = IO::read_file_iv<uint32_t>(category_file, input_type, hashers[0], category_type, true_categories, input_features->nrows);

    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Processing %d layer files (silent).\n", nmax_layers); 
    //nmax_layers = 2;
	layers.resize(nmax_layers);
	bias_vectors.resize(nmax_layers);
    //if((parallelism_type != PARALLELISM_TYPE::_HYBRID_X_HYBRID_) and (dual_spmat == true)) dual_spmat = false;
	uint64_t layer_nnzs = 0;
	uint32_t layer_nrows = 0, layer_ncols = 0;
    for(uint32_t i = 0; i < nmax_layers; i++) {
		if(i == 0) { layer_nrows = input_nfeatures; layer_ncols = nneurons; }
		else if(i < nmax_layers-1) { layer_nrows = nneurons; layer_ncols = nneurons; }
		else { layer_nrows = nneurons; layer_ncols = ncategories ? ncategories : nneurons; }
		std::string layer_file = layer_files[i];
		hashers.push_back(std::move(std::make_shared<struct TwoDHasher>(hashing_type, false, layer_nrows, layer_ncols, 1, 1)));
		layer_nnzs = IO::get_nnzs<Weight>(layer_file, input_type, hashers[i+1], layer_nrows);
		layers[i] = std::move(std::make_unique<Tiling<Weight>>(1, 1, 1, 1, 
															   layer_nnzs, layer_nrows, layer_ncols, 
															   layer_file, input_type, 
															   TILING_TYPE::_1D_COL_, compression_type, hashers[i+1]));
		bias_vectors[i] = std::move(std::make_shared<struct Data_Block<Weight>>(layer_ncols, Env::rank_socket_id));
		if(bias_type == VALUE_TYPE::_CONSTANT_) {				
			Weight* b_A = bias_vectors[i]->ptr;
			for(uint32_t j = 0; j < layer_ncols; j++) b_A[j] = bias_value;
		}
		else if(bias_type == VALUE_TYPE::_INSTANCE_AND_VALUE_PAIRS_) {
			std::string bias_file = bias_files[i];
			std::vector<Weight> bias_values;
			uint32_t c = IO::read_file_iv<Weight>(bias_file, input_type, hashers[i+1], bias_type, bias_values, layer_ncols);
			Weight* b_A = bias_vectors[i]->ptr;
			for(uint32_t j = 0; j < layer_ncols; j++) b_A[j] = bias_values[j];
		}
        Logging::enabled = false; 
        if(i%10==0) printf("|"); 
    }
    Logging::enabled = true;
    Logging::print(Logging::LOG_LEVEL::VOID, "\n"); 
    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Done reading %d layer files.\n", nmax_layers); 
    Env::barrier();

	spa_vectors.resize(Env::nthreads);
	for(int32_t i = 0; i < Env::nthreads; i++) {
		if(compression_type == COMPRESSED_FORMAT::_CSC_) {
			uint32_t max_height = input_features->get_tile_info_max("height");
			spa_vectors[i] = std::move(std::make_shared<struct Data_Block<Weight>>(max_height, Env::threads_socket_id[i]));    
		}
		else if(compression_type == COMPRESSED_FORMAT::_CSR_) {
			uint32_t max_width = input_features->get_tile_info_max("width");
			max_width = (nneurons > max_width) ? nneurons : max_width;
			spa_vectors[i] = std::move(std::make_shared<struct Data_Block<Weight>>(max_width, Env::threads_socket_id[i]));    
		}
		else {
			Logging::print(Logging::LOG_LEVEL::ERROR, "%s compression not implemented\n", COMPRESSED_FORMATS[compression_type]);
			std::exit(Env::finalize());
		}
	}
	
    if(parallelism_type == PARALLELISM_TYPE::_DATA_X_MODEL_) {
        output = std::move(std::make_unique<Tiling<Weight>>(Env::nranks, Env::nranks, 1, Env::nranks, 
                                                            0, input_ninstanses, nneurons, 
                                                            TILING_TYPE::_1D_ROW_, compression_type, hashers[0]));
    }
    else if((parallelism_type == PARALLELISM_TYPE::_MANAGER_X_WORKER_) or (parallelism_type == PARALLELISM_TYPE::_WORK_X_STEALING_)) {
        output = std::move(std::make_unique<Tiling<Weight>>(Env::nranks * Env::nthreads * split_factor, Env::nranks * Env::nthreads * split_factor, 1, Env::nranks, 
                                                            Env::nthreads, Env::nranks * Env::nthreads, 
                                                            0, input_ninstanses, nneurons, 
                                                            TILING_TYPE::_1D_ROW_, compression_type, hashers[0]));
    }
    else {
        output = std::move(std::make_unique<Tiling<Weight>>(Env::nranks * Env::nthreads, Env::nranks * Env::nthreads, 1, Env::nranks, 
                                                            Env::nthreads, Env::nranks * Env::nthreads, 
                                                            0, input_ninstanses, nneurons, 
                                                            TILING_TYPE::_1D_ROW_, compression_type, hashers[0]));
    }
    output->set_tile_info(input_features->tiles);

    Logging::print(Logging::LOG_LEVEL::INFO, "Neural network: Running the inferenceReLU method [Compression=%s|Parallelism=%s|Scheduling=%s|Hashing=%s].\n", 
                   COMPRESSED_FORMATS[compression_type], PARALLELISM_TYPES[parallelism_type], SCHEDULING_TYPES[scheduling_type], HASHING_TYPES[hashing_type]); 
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
    Logging::print(Logging::LOG_LEVEL::VOID, "exec time: %.3f %.3f %.3f %3f %3f\n", min, max, sum, mean, std_dev);
    
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
    //uint64_t DNNedges = input_features->get_info("nedges");
    uint64_t DNNedges = Net::nedges;
    uint64_t DNNConns = ninputinstanses * DNNedges;
    double inference_rate = (double) DNNConns / exec_time;
    std::tie(sum, mean, std_dev, min, max) =  Env::statistics<double>(inference_rate/1e9);
    //Logging::print(Logging::LOG_LEVEL::VOID, "Infe time: %.3f %.3f %.3f %.3f\n", mean, std_dev, min, max);
    Logging::print(Logging::LOG_LEVEL::VOID, "%.3f %.3f %.3f %.3f\n", mean, std_dev, min, max);
    */
    //double min_exec_rate = (double) (ninputinstanses * DNNedges) /max_exec_time;
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
    else if(parallelism_type == PARALLELISM_TYPE::_MANAGER_X_WORKER_) {
        manager_x_worker(tid);
    }    
    else if(parallelism_type == PARALLELISM_TYPE::_WORK_X_STEALING_) {
        work_x_stealing(tid);
    }
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

	struct Tile<Weight>& A_tile = input_features->tiles[leader_rowgroup][0];
    struct Tile<Weight>& C_tile = output->tiles[leader_rowgroup][0];
    for (uint32_t l = 0; l < nmax_layers; l++) {	
		std::shared_ptr<struct Compressed_Format<Weight>>& A_SPMAT = A_tile.spmat;
		struct Tile<Weight>& B_tile = layers[l]->tiles[0][0];
        std::shared_ptr<struct Compressed_Format<Weight>>& B_SPMAT = B_tile.spmat;
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
		
		if(compression_type == COMPRESSED_FORMAT::_CSC_) {
			start = Env::threads[tid].start_col;
			end = Env::threads[tid].end_col;
			sub_start = 0, sub_end   = 0;
		}
		else {
			Logging::print(Logging::LOG_LEVEL::ERROR, "%s compression not implemented\n", COMPRESSED_FORMATS[compression_type]);
			std::exit(Env::finalize());
		}

        data_x_model_1_iter(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, activation_function,
                            A_nrows, B_ncols, 
                            start, end, 
                            sub_start, sub_end, 
                            thread_st, leader_tid, tid); 
    }
    auto finish_t = std::chrono::high_resolution_clock::now();
    Env::execution_time[tid] = (double)(std::chrono::duration_cast<std::chrono::nanoseconds>(finish_t - start_t).count())/1e9;

    const std::shared_ptr<struct Compressed_Format<Weight>> C_SPMAT = A_tile.spmat;
    data_x_model_validate_prediction(C_SPMAT, C_tile.start_row, true_categories, predicted_nistances, category_type, leader_tid, tid);
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
        std::shared_ptr<struct Compressed_Format<Weight>> A_SPMAT = A_tile.spmat;
        struct Tile<Weight>& B_tile = layers[l]->tiles[0][0];    
        std::shared_ptr<struct Compressed_Format<Weight>> B_SPMAT = B_tile.spmat;
        struct Tile<Weight>& C_tile = (not(l%2)) ? output->tiles[leader_rowgroup][0]
                                                 : input_features->tiles[leader_rowgroup][0];
        std::shared_ptr<struct Compressed_Format<Weight>> C_SPMAT = C_tile.spmat;
		std::shared_ptr<struct Data_Block<Weight>>& s_spa = spa_vectors[tid];
		std::shared_ptr<struct Data_Block<Weight>>& b_bias = bias_vectors[l];

		A_nrows = A_SPMAT->nrows;
		B_nrows = B_SPMAT->nrows;
		B_ncols = B_SPMAT->ncols;
		
		if(compression_type == COMPRESSED_FORMAT::_CSC_) end = B_ncols;
		else if (compression_type == COMPRESSED_FORMAT::_CSR_) end = A_nrows;
		else {
			Logging::print(Logging::LOG_LEVEL::ERROR, "%s compression not implemented\n", COMPRESSED_FORMATS[compression_type]);
			std::exit(Env::finalize());
		}
		
		data_x_data_1_iter(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, activation_function,
						   A_nrows, B_ncols, start, end, off, 
                           thread_st, leader_tid, tid);
    }
    auto finish_t = std::chrono::high_resolution_clock::now();
    Env::execution_time[tid] = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish_t - start_t).count())/1e9;
    
    struct Tile<Weight>& C_tile = (not((l-1)%2)) ? output->tiles[leader_rowgroup][0] 
											 : input_features->tiles[leader_rowgroup][0];
    const std::shared_ptr<struct Compressed_Format<Weight>> C_SPMAT = C_tile.spmat;
    data_x_data_validate_prediction(C_SPMAT, C_tile.start_row, true_categories, predicted_nistances, category_type, leader_tid, tid);
}

template<typename Weight>
void Net<Weight>::hybrid_x_hybrid(const int32_t tid) {
	auto start_t = std::chrono::high_resolution_clock::now();  
    uint32_t my_rowgroup = Env::thread_rowgroup[tid];
    int32_t leader_tid = 0;
    Env::global_time = Env::tic();
    uint32_t my_start_layer = hybrid_x_data(Env::my_threads[tid], my_rowgroup, tid);
    if(my_start_layer < nmax_layers) hybrid_x_model(Env::my_threads[tid], my_rowgroup, my_start_layer, tid, tid);
    while(add_to_idle_threads(Env::my_threads[tid], tid)) {
        const int32_t leader = Env::threads[tid].leader;
        uint32_t leader_rowgroup = Env::threads[tid].rowgroup;
        uint32_t leader_start_layer = Env::threads[tid].start_layer;
        hybrid_x_model(Env::my_threads[tid], leader_rowgroup, leader_start_layer, leader, tid);
    }
	//printf("totaly done %d\n", tid);
    auto finish_t = std::chrono::high_resolution_clock::now();
    Env::execution_time[tid] = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish_t - start_t).count())/1e9;
	uint32_t layer_index = (my_start_layer == nmax_layers) ? nmax_layers-1 : my_start_layer;
	printf("tid=%d idx=%d my_start=%d max=%d\n", tid, layer_index, my_start_layer, nmax_layers);
	struct Tile<Weight>& C_tile = (not(layer_index%2)) ? output->tiles[my_rowgroup][0]
											           : input_features->tiles[my_rowgroup][0];
    //struct Tile<Weight>& A_tile = input_features->tiles[my_rowgroup][0];
    std::shared_ptr<struct Compressed_Format<Weight>> C_SPMAT = C_tile.spmat;
    data_x_data_validate_prediction(C_SPMAT, C_tile.start_row, true_categories, predicted_nistances, category_type, leader_tid, tid);
}

template<typename Weight>
uint32_t Net<Weight>::hybrid_x_data(std::deque<int32_t>& my_threads, const int32_t my_rowgroup, const int32_t tid) {
    int32_t sid = (numa_queues) ? Env::threads_socket_id[tid] : Env::rank_socket_id;
    int32_t leader_tid = 0;
    struct Env::thread_struct& thread_st = Env::threads[tid];
	uint32_t A_nrows = 0, B_nrows = 0, B_ncols = 0;
    uint32_t start = 0, end = 0;
    const uint32_t off = 0;
    bool breaking = false;
    uint32_t l = 0; 
    for (l = 0; l < nmax_layers; l++) {
		struct Tile<Weight>& A_tile = (not(l%2)) ? input_features->tiles[my_rowgroup][0]
                                                 : output->tiles[my_rowgroup][0];
        std::shared_ptr<struct Compressed_Format<Weight>> A_SPMAT = A_tile.spmat;
        struct Tile<Weight>& B_tile = layers[l]->tiles[0][0];    
        std::shared_ptr<struct Compressed_Format<Weight>> B_SPMAT = B_tile.spmat;
        struct Tile<Weight>& C_tile = (not(l%2)) ? output->tiles[my_rowgroup][0]
                                                 : input_features->tiles[my_rowgroup][0];
        std::shared_ptr<struct Compressed_Format<Weight>> C_SPMAT = C_tile.spmat;
		std::shared_ptr<struct Data_Block<Weight>> s_spa = spa_vectors[tid];
        std::shared_ptr<struct Data_Block<Weight>>& b_bias = bias_vectors[l];      
        
		A_nrows = A_SPMAT->nrows;
		B_nrows = B_SPMAT->nrows;
		B_ncols = B_SPMAT->ncols;

		if(compression_type == COMPRESSED_FORMAT::_CSC_) end = B_ncols;
		else if (compression_type == COMPRESSED_FORMAT::_CSR_) end = A_nrows;
		else {
			Logging::print(Logging::LOG_LEVEL::ERROR, "%s compression not implemented\n", COMPRESSED_FORMATS[compression_type]);
			std::exit(Env::finalize());
		}
		
		uint32_t A_ncols = A_SPMAT->ncols;
        if((l >= nmax_layers*recruiting_ratio) and add_to_my_follower_threads(my_threads, my_rowgroup, l, 0, A_nrows, B_ncols, tid, tid)) {
			printf("D:tid=%d layer=l=%d rg=%d A[%d %d] B[%d %d]\n", tid, l,  my_rowgroup, A_nrows, A_ncols, B_nrows, B_ncols);
			break;
		}
            //if(not(l%2)) break;
            //else breaking = true;
        //}
		//
		
		data_x_data_1_iter(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, activation_function,
                           A_nrows, B_ncols, start, end, off, 
                           thread_st, leader_tid, tid); 
						   
        Env::scores[sid][tid]++;     
        
		//if(breaking) break;
    }
    return(l);
}

template<typename Weight>
void Net<Weight>::hybrid_x_model(std::deque<int32_t>& leader_owned_threads, const uint32_t leader_rowgroup, const uint32_t leader_start_layer, const int32_t leader_tid, const int32_t tid) {
    int32_t sid = (numa_queues) ? Env::threads_socket_id[tid] : Env::rank_socket_id;
    struct Env::thread_struct& thread_st = Env::threads[tid];    
	
	uint32_t A_nrows = 0, B_nrows = 0, B_ncols = 0;
    uint32_t start = 0, end = 0;
    //uint32_t sub_start = 0, sub_end = 0;
	//uint32_t start_row = 0;
	const uint32_t off = 0;  
	uint32_t B_ncols_prev = 0;
	struct Tile<Weight>& A_tile = (not(leader_rowgroup%2)) ? input_features->tiles[leader_rowgroup][0]
											               : output->tiles[leader_rowgroup][0];
	struct Tile<Weight>& C_tile = (not(leader_rowgroup%2)) ? output->tiles[leader_rowgroup][0]
											               : input_features->tiles[leader_rowgroup][0];
	//start_row = A_tile.start_row;
	//struct Tile<Weight>& A_tile =  input_features->tiles[leader_rowgroup][0];
    //struct Tile<Weight>& C_tile = output->tiles[leader_rowgroup][0];
	B_ncols = layers[leader_start_layer]->tiles[0][0].spmat->ncols;
    for (uint32_t l = leader_start_layer; l < nmax_layers; l++) {
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
		
        bool new_followers=add_to_my_follower_threads(leader_owned_threads, leader_rowgroup, l, 0, A_nrows, B_ncols, leader_tid, tid);
        double start_time = Env::tic();   
            Env::decrease_num_threads(1, leader_tid, tid);
            Env::init_num_threads(leader_owned_threads.size(), leader_tid, tid);
        Env::hybrid_probe_time[tid] += Env::toc(start_time);     
		
		if(compression_type == COMPRESSED_FORMAT::_CSC_) {
			if(B_ncols != B_ncols_prev) {				
				if(tid == leader_tid){// and not new_followers and l != leader_start_layer) {
					printf("XXXXXXXXXXXXXXXX %d l=%d st=%d\n", tid, l , leader_start_layer);
					uint32_t num_threads = leader_owned_threads.size();
					if(num_threads > B_ncols) {
						for(uint32_t i = 0; i < num_threads; i++) {
							int32_t t = leader_owned_threads[i];
							Env::threads[t].start_row = (i<B_ncols) ? i : 0;
							Env::threads[t].end_row   = (i<B_ncols) ? i+1 : 0;
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
		else if(compression_type == COMPRESSED_FORMAT::_CSR_) {
			start = Env::threads[tid].start_row;
			end = Env::threads[tid].end_row;
		}
		else {
			Logging::print(Logging::LOG_LEVEL::ERROR, "%s compression not implemented\n", COMPRESSED_FORMATS[compression_type]);
			std::exit(Env::finalize());
		}
		uint32_t A_ncols = A_SPMAT->ncols;
		printf("M:tid=%d/%d/%lu l=%d r=%d A[%d %d] B[%d %d] [%d %d]\n", tid, leader_tid, leader_owned_threads.size(), l, leader_rowgroup, A_nrows, A_ncols, B_nrows, B_ncols, start, end);
        data_x_model_hybrid_1_iter(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, activation_function,
               A_nrows, B_ncols, start, end, off,
               leader_owned_threads, thread_st, leader_tid, tid);
		
       if(tid == leader_tid) Env::scores[sid][tid]++;
    }
	//printf("done hybrid %d\n", tid);
}

template<typename Weight>
bool Net<Weight>::thread_scheduling(std::deque<int32_t>& leader_owned_threads, const uint32_t leader_rowgroup, std::deque<int32_t>& follower_threads, int32_t socket_id, const uint32_t start_layer, const uint32_t start_row, const uint32_t nrows, const uint32_t ncols, const int32_t leader, const int32_t tid) {  
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
                
                if(compression_type == COMPRESSED_FORMAT::_CSR_) {
					if(num_threads > nrows) {
						for(uint32_t i = 0; i < num_threads; i++) {
							int32_t t = leader_owned_threads[i];
							Env::threads[t].index = i;
							Env::threads[t].leader = tid;
							Env::threads[t].rowgroup = leader_rowgroup;
							Env::threads[t].start_layer = start_layer;
							Env::threads[t].start_row = (i<nrows) ? i : 0;
							Env::threads[t].end_row   = (i<nrows) ? i+1 : 0;
						} 	 
					}
					else {
						for(uint32_t i = 0; i < num_threads; i++) {
							int32_t t = leader_owned_threads[i];
							Env::threads[t].index = i;
							Env::threads[t].leader = tid;
							Env::threads[t].rowgroup = leader_rowgroup;
							Env::threads[t].start_layer = start_layer;
							Env::threads[t].start_row = ((nrows/num_threads) * i);
							Env::threads[t].end_row   = (i == (num_threads-1)) ? nrows : ((nrows/num_threads) * (i+1));
						}                     
					}
                }
                else if(compression_type == COMPRESSED_FORMAT::_CSC_) {
					if(num_threads > ncols) {
						for(uint32_t i = 0; i < num_threads; i++) {
							int32_t t = leader_owned_threads[i];
							Env::threads[t].index = i;
							Env::threads[t].leader = tid;
							Env::threads[t].rowgroup = leader_rowgroup;
							Env::threads[t].start_layer = start_layer;
							Env::threads[t].start_col = (i<ncols) ? i : 0;
							Env::threads[t].end_col   = (i<ncols) ? i+1 : 0;
						}
					}
					else {
						for(uint32_t i = 0; i < num_threads; i++) {
							int32_t t = leader_owned_threads[i];
							Env::threads[t].index = i;
							Env::threads[t].leader = tid;
							Env::threads[t].rowgroup = leader_rowgroup;
							Env::threads[t].start_layer = start_layer;
							Env::threads[t].start_col = ((ncols/num_threads) * i);
							Env::threads[t].end_col   = (i == (num_threads-1)) ? ncols : ((ncols/num_threads) * (i+1));
						}
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
bool Net<Weight>::add_to_my_follower_threads(std::deque<int32_t>& leader_owned_threads, const uint32_t leader_rowgroup, const uint32_t start_layer, const uint32_t start_row, const uint32_t nrows, const uint32_t ncols, const int32_t leader_tid, const int32_t tid) {  
    bool found = false;
    if(tid == leader_tid) {
        double start_time = 0;
        start_time = Env::tic();
        int32_t sid = (numa_queues) ? Env::threads_socket_id[tid] : Env::rank_socket_id;
        if(numa_queues) {
            for(int32_t s = 0; s < Env::nsockets; s++) {
                int32_t si = (s + Env::threads_socket_id[tid]) % Env::nsockets;
                if((si == sid) or (Env::nthreads_per_socket[si] and (Env::numa_follower_threads[si].size() == (uint32_t) Env::nthreads_per_socket[si]))) {
                    found |= thread_scheduling(leader_owned_threads, leader_rowgroup, Env::numa_follower_threads[si], si, start_layer, start_row, nrows, ncols, leader_tid, tid);
                }
            }
        }
        else {
            found = thread_scheduling(leader_owned_threads, leader_rowgroup, Env::numa_follower_threads[sid], sid, start_layer, start_row, nrows, ncols, leader_tid, tid);
        }
        Env::hybrid_probe_time[tid] += Env::toc(start_time);  
    }

    return(found);
}

template<typename Weight>
bool Net<Weight>::add_to_idle_threads(std::deque<int32_t>& my_threads, const int32_t tid) {
    uint32_t sid = (numa_queues) ? Env::threads_socket_id[tid] : Env::rank_socket_id;
    bool status = true;
    uint32_t all_done = 0;

    pthread_mutex_lock(&Env::numa_thread_mutex[sid]);
    Env::numa_follower_threads[sid].push_back(tid);
    if(not my_threads.empty()) my_threads.erase(my_threads.begin(), my_threads.end());
    
    Env::threads[tid].leader = -1;
        
    for(std::deque<int32_t>& numa_thread: Env::numa_follower_threads) {
		all_done += numa_thread.size();
	}
    //printf("tid=%d all_done=%d nth=%d\n", tid, all_done, Env::nthreads);
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
            std::shared_ptr<struct Compressed_Format<Weight>> A_SPMAT = A_tile.spmat;
            struct Tile<Weight>& B_tile = layers[l]->tiles[0][0];    
            std::shared_ptr<struct Compressed_Format<Weight>> B_SPMAT = B_tile.spmat;
            struct Tile<Weight>& C_tile = (not(l%2)) ? output->tiles[leader_rowgroup][0]
                                                     : input_features->tiles[leader_rowgroup][0];
            std::shared_ptr<struct Compressed_Format<Weight>> C_SPMAT = C_tile.spmat;
            std::shared_ptr<struct Data_Block<Weight>>& s_spa = spa_vectors[tid];
			std::shared_ptr<struct Data_Block<Weight>>& b_bias = bias_vectors[l];    
            
			A_nrows = A_SPMAT->nrows;
			B_nrows = B_SPMAT->nrows;
			B_ncols = B_SPMAT->ncols;
			
            if(compression_type == COMPRESSED_FORMAT::_CSC_) end = B_ncols;
			else if (compression_type == COMPRESSED_FORMAT::_CSR_) end = A_nrows;
			else {
				Logging::print(Logging::LOG_LEVEL::ERROR, "%s compression not implemented\n", COMPRESSED_FORMATS[compression_type]);
				std::exit(Env::finalize());
			}
            
            data_x_data_1_iter(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, activation_function,
                               A_nrows, B_ncols, start, end, off, 
                               thread_st, leader_tid, tid);       
        }   
    }
    auto finish_t = std::chrono::high_resolution_clock::now();
    Env::execution_time[tid] = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish_t - start_t).count())/1e9;
    manager_x_worker_validate_prediction(input_features->tiles, true_categories, predicted_nistances, category_type, leader_tid, tid);
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
            std::shared_ptr<struct Compressed_Format<Weight>> A_SPMAT = A_tile.spmat;
            struct Tile<Weight>& B_tile = layers[l]->tiles[0][0];    
            std::shared_ptr<struct Compressed_Format<Weight>> B_SPMAT = B_tile.spmat;
            struct Tile<Weight>& C_tile = (not(l%2)) ? output->tiles[leader_rowgroup][0]
                                                     : input_features->tiles[leader_rowgroup][0];
            std::shared_ptr<struct Compressed_Format<Weight>> C_SPMAT = C_tile.spmat;
			std::shared_ptr<struct Data_Block<Weight>> s_spa = spa_vectors[tid];
			std::shared_ptr<struct Data_Block<Weight>> b_bias = bias_vectors[l];  
		
			A_nrows = A_SPMAT->nrows;
			B_nrows = B_SPMAT->nrows;
			B_ncols = B_SPMAT->ncols;
			
            if(compression_type == COMPRESSED_FORMAT::_CSC_) end = B_ncols;
			else if (compression_type == COMPRESSED_FORMAT::_CSR_) end = A_nrows;
			else {
				Logging::print(Logging::LOG_LEVEL::ERROR, "%s compression not implemented\n", COMPRESSED_FORMATS[compression_type]);
				std::exit(Env::finalize());
			}
            
            data_x_data_1_iter(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, activation_function,
                               A_nrows, B_ncols, start, end, off, 
                               thread_st, leader_tid, tid);       
        }   
    }

    auto finish_t = std::chrono::high_resolution_clock::now();
    Env::execution_time[tid] = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish_t - start_t).count())/1e9;

    work_x_stealing_validate_prediction(input_features->tiles, true_categories, predicted_nistances, category_type, leader_tid, tid);
}

#endif 
