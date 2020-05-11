/*
 * spmat.hpp: Sparse Matrix implementation 
 * (c) Mohammad Hasanzadeh Mofrad, 2020
 * (e) m.hasanzadeh.mofrad@gmail.com
 */
 
#ifndef SPMAT_HPP
#define SPMAT_HPP

#include <numeric>
#include <limits.h>
#include <tuple>

#include "allocator.hpp"
#include "triple.hpp"
#include "env.hpp"

enum COMPRESSED_FORMAT {_CSR_, _DCSR_, _TCSR_, _CSC_, _DCSC_, _TCSC_};
const char* COMPRESSED_FORMATS[] = {"_CSR_", "_DCSR_", "_TCSR_", "_CSC_", "_DCSC_", "_TCSC_"};


template<typename Weight>
struct Compressed_Format {
    public:
        Compressed_Format() {}
        virtual ~Compressed_Format() {}
        //virtual void populate(std::vector<struct Triple<Weight>>& triples) {Logging::print(Logging::LOG_LEVEL::ERROR, "Not implemented\n"); std::exit(Env::finalize());}
        // Fixed tile height and width; tile height = nrows / nranks or tile width = ncols / nranks 
        virtual void populate(std::vector<struct Triple<Weight>>& triples, const uint32_t tile_height, const uint32_t tile_width) {Logging::print(Logging::LOG_LEVEL::ERROR, "Not implemented\n"); std::exit(Env::finalize());}
        // If tile height and width are not necessarily multiples of nrows and ncols 
        //virtual void populate(std::vector<struct Triple<Weight>>& triples, const uint32_t start_row, const uint32_t tile_height, const uint32_t start_col, const uint32_t tile_width) {Logging::print(Logging::LOG_LEVEL::ERROR, "Not implemented\n"); std::exit(Env::finalize());}
        //virtual void populate_spa(Weight** spa, const Weight* bias, const uint32_t col,  uint64_t& index, const int32_t tid) {Logging::print(Logging::LOG_LEVEL::ERROR, "Not implemented\n"); std::exit(Env::finalize());}
		void populate_spa(Weight** spa, const Weight* bias, const uint32_t col,  uint64_t& index, Weight (*)(Weight), const int32_t tid){Logging::print(Logging::LOG_LEVEL::ERROR, "Not implemented\n"); std::exit(Env::finalize());}
        virtual void walk_dxm1(const bool one_rank, const int32_t leader_tid, const int32_t tid) {Logging::print(Logging::LOG_LEVEL::ERROR, "Not implemented\n"); std::exit(Env::finalize());}
        virtual void walk_dxm(const bool one_rank, const int32_t leader_tid, const int32_t tid) {Logging::print(Logging::LOG_LEVEL::ERROR, "Not implemented\n"); std::exit(Env::finalize());}
        virtual void walk_dxd(const bool one_rank, const int32_t leader_tid, const int32_t tid) {Logging::print(Logging::LOG_LEVEL::ERROR, "Not implemented\n"); std::exit(Env::finalize());}
        virtual void reallocate(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t leader_tid, const int32_t tid) {Logging::print(Logging::LOG_LEVEL::ERROR, "Not implemented\n"); std::exit(Env::finalize());}
        virtual void adjust(const int32_t tid) {Logging::print(Logging::LOG_LEVEL::ERROR, "Not implemented\n"); std::exit(Env::finalize());}
        virtual void adjust(const int32_t leader_tid, const int32_t tid) {Logging::print(Logging::LOG_LEVEL::ERROR, "Not implemented\n"); std::exit(Env::finalize());}
        virtual void adjust(const std::deque<int32_t> my_threads, const int32_t leader_tid, const int32_t tid) {Logging::print(Logging::LOG_LEVEL::ERROR, "Not implemented\n"); std::exit(Env::finalize());}
        virtual void repopulate(const std::shared_ptr<struct Compressed_Format<Weight>> other_spmat, const uint32_t dis_nnz, const int32_t leader_tid, const int32_t tid) {Logging::print(Logging::LOG_LEVEL::ERROR, "Not implemented\n"); std::exit(Env::finalize());}
        virtual void repopulate(const std::shared_ptr<struct Compressed_Format<Weight>> other_spmat, const std::deque<int32_t> my_threads, const int32_t leader_tid, const int32_t tid) {Logging::print(Logging::LOG_LEVEL::ERROR, "Not implemented\n"); std::exit(Env::finalize());}
        
        COMPRESSED_FORMAT compression_type;
        
        uint64_t nnz   = 0;
        uint64_t nnz_i = 0;
        uint32_t nrows = 0;
        uint32_t ncols = 0;
        
        std::shared_ptr<struct Data_Block<uint32_t>> IA_blk;
        std::shared_ptr<struct Data_Block<uint32_t>> JA_blk;
        std::shared_ptr<struct Data_Block<Weight>>   A_blk;
};

template<typename Weight>
struct CSR: public Compressed_Format<Weight> {
    public:
        CSR(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_);
        CSR(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t socket_id);
        //CSR(const std::shared_ptr<struct Compressed_Format<Weight>> other_spmat, const uint32_t start_row, const uint32_t tile_height, const uint32_t start_col, const uint32_t tile_width, const int32_t socket_id);
        ~CSR(){};
        
        //void populate(std::vector<struct Triple<Weight>>& triples);
        void populate(std::vector<struct Triple<Weight>>& triples, const uint32_t tile_height, const uint32_t tile_width);
        //void populate(std::vector<struct Triple<Weight>>& triples, const uint32_t start_row, const uint32_t tile_height, const uint32_t start_col, const uint32_t tile_width);
        //void populate_spa(Weight** spa, const Weight* bias, const uint32_t col,  uint64_t& index, const int32_t tid);
		void populate_spa(Weight** spa, const Weight* bias, const uint32_t col,  uint64_t& index, Weight (*)(Weight), const int32_t tid);
        void walk_dxm1(const bool one_rank, const int32_t leader_tid, const int32_t tid){};
        void walk_dxm(const bool one_rank, const int32_t leader_tid, const int32_t tid);
        void walk_dxd(const bool one_rank, const int32_t leader_tid, const int32_t tid);
        void reallocate(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t leader_tid, const int32_t tid);
        void adjust(const int32_t tid);
        void adjust(const int32_t leader_tid, const int32_t tid);
        void adjust(const std::deque<int32_t> my_threads, const int32_t leader_tid, const int32_t tid);
        void repopulate(const std::shared_ptr<struct Compressed_Format<Weight>> other_spmat, const uint32_t dis_nnz, const int32_t leader_tid, const int32_t tid);
        void repopulate(const std::shared_ptr<struct Compressed_Format<Weight>> other_spmat, const std::deque<int32_t> my_threads, const int32_t leader_tid, const int32_t tid);
        
        uint64_t nnz   = 0;
        uint64_t nnz_i = 0;
        uint32_t nrows = 0;
        uint32_t ncols = 0;
        
        std::shared_ptr<struct Data_Block<uint32_t>> IA_blk;
        std::shared_ptr<struct Data_Block<uint32_t>> JA_blk;
        std::shared_ptr<struct Data_Block<Weight>>   A_blk;
};

/* Compressed Sparse Column (CSC) */
template<typename Weight>
struct CSC: public Compressed_Format<Weight> {
    public:
        CSC(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_);
        CSC(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t socket_id);        
        //CSC(const std::shared_ptr<struct Compressed_Format<Weight>> other_spmat, const uint32_t start_row, const uint32_t tile_height, const uint32_t start_col, const uint32_t tile_width, const int32_t socket_id);
        ~CSC(){};
        
        //void populate(std::vector<struct Triple<Weight>>& triples);
        void populate(std::vector<struct Triple<Weight>>& triples, const uint32_t tile_height, const uint32_t tile_width);
        //void populate(std::vector<struct Triple<Weight>>& triples, const uint32_t start_row, const uint32_t tile_height, const uint32_t start_col, const uint32_t tile_width);
        //void populate_spa(Weight** spa, const Weight* bias, const uint32_t col,  uint64_t& index, const int32_t tid);
		void populate_spa(Weight** spa, const Weight* bias, const uint32_t col,  uint64_t& index, Weight (*)(Weight), const int32_t tid);
        void walk_dxm1(const bool one_rank, const int32_t leader_tid, const int32_t tid);
        void walk_dxm(const bool one_rank, const int32_t leader_tid, const int32_t tid);
        void walk_dxd(const bool one_rank, const int32_t leader_tid, const int32_t tid);
        void reallocate(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t leader_tid, const int32_t tid);
        void adjust(const int32_t tid);
        void adjust(const int32_t leader_tid, const int32_t tid);
        void adjust(const std::deque<int32_t> my_threads, const int32_t leader_tid, const int32_t tid);
        void repopulate(const std::shared_ptr<struct Compressed_Format<Weight>> other_spmat, const uint32_t dis_nnz, const int32_t leader_tid, const int32_t tid);
        void repopulate(const std::shared_ptr<struct Compressed_Format<Weight>> other_spmat, const std::deque<int32_t> my_threads, const int32_t leader_tid, const int32_t tid);
        
        uint64_t nnz   = 0;
        uint64_t nnz_i = 0;
        uint32_t nrows = 0;
        uint32_t ncols = 0;
        
        std::shared_ptr<struct Data_Block<uint32_t>> IA_blk;
        std::shared_ptr<struct Data_Block<uint32_t>> JA_blk;
        std::shared_ptr<struct Data_Block<Weight>>   A_blk;
};

/* Compressed Sparse Row (CSR) */
template<typename Weight>
CSR<Weight>::CSR(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t socket_id) {
    Compressed_Format<Weight>::compression_type = COMPRESSED_FORMAT::_CSR_;
    Compressed_Format<Weight>::nnz = nnz_;
    Compressed_Format<Weight>::nnz_i = nnz_;
    Compressed_Format<Weight>::nrows = nrows_; 
    Compressed_Format<Weight>::ncols = ncols_;
    
    CSR::compression_type = COMPRESSED_FORMAT::_CSR_;
    CSR::nnz = nnz_;
    CSR::nnz_i = nnz_;
    CSR::nrows = nrows_; 
    CSR::ncols = ncols_;
    
    CSR::IA_blk = std::move(std::make_shared<struct Data_Block<uint32_t>>((CSR::nrows + 1), socket_id));
    CSR::JA_blk = std::move(std::make_shared<struct Data_Block<uint32_t>>(CSR::nnz, socket_id));
    CSR::A_blk = std::move(std::make_shared<struct Data_Block<Weight>>(CSR::nnz, socket_id));
}

template<typename Weight>
CSR<Weight>::CSR(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_) {
    Compressed_Format<Weight>::compression_type = COMPRESSED_FORMAT::_CSR_;
    Compressed_Format<Weight>::nnz = nnz_;
    Compressed_Format<Weight>::nnz_i = nnz_;
    Compressed_Format<Weight>::nrows = nrows_; 
    Compressed_Format<Weight>::ncols = ncols_;
    
    CSR::compression_type = COMPRESSED_FORMAT::_CSR_;
    CSR::nnz = nnz_;
    CSR::nnz_i = nnz_;
    CSR::nrows = nrows_; 
    CSR::ncols = ncols_;
    
    CSR::IA_blk = std::move(std::make_shared<struct Data_Block<uint32_t>>((CSR::nrows + 1), Env::rank_socket_id));
    CSR::JA_blk = std::move(std::make_shared<struct Data_Block<uint32_t>>(CSR::nnz, Env::rank_socket_id));
    CSR::A_blk = std::move(std::make_shared<struct Data_Block<Weight>>(CSR::nnz, Env::rank_socket_id));
}

/*
template<typename Weight>
CSR<Weight>::CSR(const std::shared_ptr<struct Compressed_Format<Weight>> other_spmat, const uint32_t start_row, const uint32_t tile_height, const uint32_t start_col, const uint32_t tile_width, const int32_t socket_id) {
    std::shared_ptr<struct CSC<Weight>> other_csc = std::static_pointer_cast<struct CSC<Weight>>(other_spmat);
    uint32_t  o_ncols = other_csc->ncols;
    uint32_t  o_nrows = other_csc->nrows;
    uint64_t  o_nnz   = other_csc->nnz;
    uint64_t  o_nnz_i = other_csc->nnz_i;
    uint32_t* o_JA    = other_csc->JA_blk->ptr;
    uint32_t* o_IA    = other_csc->IA_blk->ptr;
    Weight*   o_A     = other_csc->A_blk->ptr;
    
    Compressed_Format<Weight>::compression_type = COMPRESSED_FORMAT::_CSR_;
    Compressed_Format<Weight>::nnz = o_nnz;
    Compressed_Format<Weight>::nnz_i = o_nnz;
    Compressed_Format<Weight>::nrows = o_nrows; 
    Compressed_Format<Weight>::ncols = o_ncols;
    
    CSR::compression_type = COMPRESSED_FORMAT::_CSR_;
    CSR::nnz = o_nnz;
    CSR::nnz_i = o_nnz;
    CSR::nrows = o_nrows; 
    CSR::ncols = o_ncols;
    
    CSR::IA_blk = std::move(std::make_shared<struct Data_Block<uint32_t>>((CSR::nrows + 1), socket_id));
    CSR::JA_blk = std::move(std::make_shared<struct Data_Block<uint32_t>>(CSR::nnz, socket_id));
    CSR::A_blk = std::move(std::make_shared<struct Data_Block<Weight>>(CSR::nnz, socket_id));
    
    uint32_t* IA = CSR::IA_blk->ptr;
    uint32_t* JA = CSR::JA_blk->ptr;
    Weight* A = CSR::A_blk->ptr;
    
    std::vector<struct Triple<Weight>> triples;
    for(uint32_t j = 0; j < o_ncols; j++) {
        for(uint32_t i = o_JA[j]; i < o_JA[j + 1]; i++) {
            struct Triple<Weight> triple = {o_IA[i], j, o_A[i]};
            triples.push_back(triple);
        }
    }
    CSR::populate(triples);   
    triples.clear();
    triples.shrink_to_fit();
}

template<typename Weight>
void CSR<Weight>::populate(std::vector<struct Triple<Weight>>& triples) {    
    const RowSort<Weight> f_row;
    std::sort(triples.begin(), triples.end(), f_row);      

    uint32_t* IA = CSR::IA_blk->ptr;
    uint32_t* JA = CSR::JA_blk->ptr;
    Weight* A = CSR::A_blk->ptr;
    
    uint32_t i = 1;
    uint32_t j = 0; 
    IA[0] = 0;
    for(auto &triple: triples) {
        while((i - 1) != triple.row) {
            i++;
            IA[i] = IA[i - 1];
        }                  
        IA[i]++;
        JA[j] = triple.col;
        A[j] = triple.weight;
        j++;
    }
    
    while(i < CSR::nrows) {
        i++;
        IA[i] = IA[i - 1];
    }
    CSR::nnz_i = CSR::nnz; 
}


template<typename Weight>
void CSR<Weight>::populate(std::vector<struct Triple<Weight>>& triples, const uint32_t start_row, const uint32_t tile_height, 
                                                                             const uint32_t start_col, const uint32_t tile_width) {    
    const RowSort<Weight> f_row;
    std::sort(triples.begin(), triples.end(), f_row);      

    uint32_t* IA = CSR::IA_blk->ptr;
    uint32_t* JA = CSR::JA_blk->ptr;
    Weight* A = CSR::A_blk->ptr;
    
    uint32_t i = 1;
    uint32_t j = 0; 
    IA[0] = 0;
    for(auto &triple: triples) {
        std::pair pair = std::make_pair(((triple.row - start_row) % tile_height), ((triple.col - start_col) % tile_width));
        while((i - 1) != pair.first) {
            i++;
            IA[i] = IA[i - 1];
        }                  
        IA[i]++;
        JA[j] = pair.second;
        A[j] = triple.weight;
        j++;
    }
    
    while(i < CSR::nrows) {
        i++;
        IA[i] = IA[i - 1];
    }
    CSR::nnz_i = CSR::nnz; 
}
*/

template<typename Weight>
void CSR<Weight>::populate(std::vector<struct Triple<Weight>>& triples, const uint32_t tile_height, const uint32_t tile_width) {
    const RowSort<Weight> f_row;
    std::sort(triples.begin(), triples.end(), f_row);    
    
    uint32_t* IA = CSR::IA_blk->ptr;
    uint32_t* JA = CSR::JA_blk->ptr;
    Weight* A = CSR::A_blk->ptr;
        
    uint32_t i = 1;
    uint32_t j = 0; 
    IA[0] = 0;
    for(auto &triple: triples) {
		std::pair pair = std::make_pair((triple.row % tile_height), (triple.col % tile_width));
        while((i - 1) != pair.first) {
            i++;
            IA[i] = IA[i - 1];
        }                  
        IA[i]++;
        JA[j] = pair.second;
        A[j] = triple.weight;
        j++;
    }
    
    while(i < CSR::nrows) {
        i++;
        IA[i] = IA[i - 1];
    }

    CSR::nnz_i = CSR::nnz; 	
}
/*
template<typename Weight>
void CSR<Weight>::populate_spa(Weight** spa, const Weight* bias, const uint32_t row, uint64_t& index, const int32_t tid) {
    uint64_t&  k = index;
    uint32_t   r = row + 1;
    uint32_t* IA = CSR::IA_blk->ptr;
    uint32_t* JA = CSR::JA_blk->ptr;
    Weight*    A = CSR::A_blk->ptr;
    Weight*    s = *spa;
    const Weight* b = bias;
    
    // ReLU activation function thresholds 
    const Weight YMIN = 0; 
    const Weight YMAX = 32;
    
    for(uint32_t j = 0; j < CSR::ncols; j++) {
        if(s[j]) {
            s[j] += b[j];
            s[j] = (s[j] < YMIN) ? YMIN : (s[j] > YMAX) ? YMAX : s[j];
            if(s[j]) {
                JA[k] = j;
                A[k] = s[j];
                k++;
                s[j] = 0;
            }
        }
    }
    IA[r] = k;
}
*/
template<typename Weight>
void CSR<Weight>::populate_spa(Weight** spa, const Weight* bias, const uint32_t row, uint64_t& index, Weight(*activation_function)(Weight), const int32_t tid) {
    uint64_t&  k = index;
    uint32_t   r = row + 1;
    uint32_t* IA = CSR::IA_blk->ptr;
    uint32_t* JA = CSR::JA_blk->ptr;
    Weight*    A = CSR::A_blk->ptr;
    Weight*    s = *spa;
    const Weight* b = bias;
    
    /* ReLU activation function thresholds */
    const Weight YMIN = 0; 
    const Weight YMAX = 32;
    
    for(uint32_t j = 0; j < CSR::ncols; j++) {
        if(s[j]) {
            s[j] += b[j];
            //s[j] = (s[j] < YMIN) ? YMIN : (s[j] > YMAX) ? YMAX : s[j];
			s[j]=activation_function(s[j]);
            if(s[j]) {
                JA[k] = j;
                A[k] = s[j];
                k++;
                s[j] = 0;
            }
        }
    }
    IA[r] = k;
}

template<typename Weight>
void CSR<Weight>::walk_dxm(const bool one_rank, const int32_t leader_tid, const int32_t tid) {  
    if(tid == leader_tid) {
        uint32_t* IA = CSR::IA_blk->ptr;
        uint32_t* JA = CSR::JA_blk->ptr;
        Weight*    A = CSR::A_blk->ptr;
        
        double checksum = 0;
        uint64_t checkcount = 0;
        for(uint32_t i = 0; i < CSR::nrows; i++) {
            //if(!Env::rank)
            //    std::cout << "i=" << i << "," << i << ": " << IA[i] << "--" << IA[i + 1] << ": " <<  IA[i + 1] - IA[i] << std::endl;
            for(uint32_t j = IA[i]; j < IA[i + 1]; j++) {
                (void) JA[j];
                (void) A[j];
                checksum += A[j];
                checkcount++;
                //if(!Env::rank)
                //    std::cout << "    j=" << j << ",j=" << JA[j] <<  ",value=" << A[j] << std::endl;
            }
        }

        Env::barrier();
        if(one_rank) {
            Logging::print(Logging::LOG_LEVEL::INFO, "CSR: Iteration=%d, Total checksum=%f, Total count=%d\n", Env::iteration, checksum, checkcount);
        }
        else {
            uint64_t nnz_ = CSR::nnz_i;
            uint64_t nnz_ranks = 0;
            double sum_ranks = 0;
            uint64_t count_ranks = 0;
            MPI_Allreduce(&nnz_, &nnz_ranks, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&checksum, &sum_ranks, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&checkcount, &count_ranks, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);

            if(count_ranks != nnz_ranks) {
                Logging::print(Logging::LOG_LEVEL::WARN, "Compression checksum warning!!\n");
            }
            Logging::print(Logging::LOG_LEVEL::INFO, "tid=%d CSR: Iteration=%d, Total checksum=%f, Total count=%d\n", tid, Env::iteration, sum_ranks, count_ranks);
        } 
    }    
}

template<typename Weight>
void CSR<Weight>::walk_dxd(const bool one_rank, const int32_t leader_tid, const int32_t tid) {
    uint32_t* IA = CSR::IA_blk->ptr;
    uint32_t* JA = CSR::JA_blk->ptr;
    Weight*    A = CSR::A_blk->ptr;
    
    double&   checksum   = Env::counters[tid].checksum;
    uint64_t& checkcount = Env::counters[tid].checkcount;
    uint64_t& checknnz   = Env::counters[tid].checknnz;
    
    checksum   = 0;
    checkcount = 0;    
    checknnz   = CSR::nnz_i;

    for(uint32_t i = 0; i < CSR::nrows; i++) { 
		//printf("",);
        //if(!Env::rank and !tid)
            //std::cout << "i=" << i << "," << i << ": " << IA[i] << "--" << IA[i + 1] << ": " <<  IA[i + 1] - IA[i] << std::endl;    
        for(uint32_t j = IA[i]; j < IA[i + 1]; j++) {
            (void) JA[j];
            (void) A[j];
            checksum += A[j];
            checkcount++;
            //if(!Env::rank and !tid)
            //std::cout << "    j=" << j << ",j=" << JA[j] <<  ",value=" << A[j] << std::endl;
        }
    }   
    
    Env::barrier();
    pthread_barrier_wait(&Env::thread_barrier);
    if(tid == leader_tid) {
        double     sum_threads = 0;
        uint64_t count_threads = 0;
        uint64_t   nnz_threads = 0;
        
        for(auto it = Env::counters.begin(); it != Env::counters.end(); it++) {
            sum_threads   += (*it).checksum;
            count_threads += (*it).checkcount;
            nnz_threads   += (*it).checknnz;
        }
        
        if(one_rank) {
            Logging::print(Logging::LOG_LEVEL::INFO, "CSR: Iteration=%d, Total checksum=%f, Total count=%d\n", Env::iteration, sum_threads, count_threads);
        }
        else {
            double     sum_ranks = 0;
            uint64_t count_ranks = 0;
            
            MPI_Allreduce(&sum_threads, &sum_ranks, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&count_threads, &count_ranks, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
            
            if(count_threads != nnz_threads) {
                Logging::print(Logging::LOG_LEVEL::WARN, "Compression checksum warning!! (%lu != %lu)\n", count_threads, nnz_threads);
            }
            Logging::print(Logging::LOG_LEVEL::INFO, "CSR: Iteration=%d, Total checksum=%f, Total count=%d\n", Env::iteration, sum_ranks, count_ranks);
        }
    }
}

template<typename Weight>
void CSR<Weight>::reallocate(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t leader_tid, const int32_t tid) {
    //if(CSR::ncols != ncols_) {
      //  Logging::print(Logging::LOG_LEVEL::ERROR, "Cannot reallocate.\n");
        //std::exit(Env::finalize());     
    //}
    if((leader_tid == -1) or (tid == leader_tid)) {
        CSR::nnz = nnz_;
        CSR::nnz_i = 0;
        CSR::nrows = nrows_; 
        CSR::ncols = ncols_;
		CSR::IA_blk->reallocate(CSR::nrows+1);
		CSR::IA_blk->clear();
        CSR::JA_blk->reallocate(CSR::nnz);
		CSR::JA_blk->clear();
        CSR::A_blk->reallocate(CSR::nnz);
        CSR::A_blk->clear();    
        Compressed_Format<Weight>::nnz = nnz_;
        Compressed_Format<Weight>::nnz_i = 0;
        Compressed_Format<Weight>::nrows = nrows_; 
        Compressed_Format<Weight>::ncols = ncols_;
    }

}

template<typename Weight>
void CSR<Weight>::adjust(const int32_t tid){
    CSR::nnz_i = Env::threads[tid].idx_nnz;
    Env::nnzs[tid].push_back(Env::threads[tid].idx_nnz);
}

template<typename Weight>
void CSR<Weight>::adjust(const int32_t leader_tid, const int32_t tid){
    if((leader_tid == -1) or (tid == leader_tid)) {
        CSR::nnz_i = 0;
        for(uint32_t i = 0; i < Env::threads.size(); i++) {    
            CSR::nnz_i += (Env::threads[i].idx_nnz - Env::threads[i].off_nnz);
            Env::nnzs[i].push_back(Env::threads[i].idx_nnz - Env::threads[i].off_nnz);
        }
    }
    pthread_barrier_wait(&Env::thread_barrier);
}

template<typename Weight>
void CSR<Weight>::adjust(const std::deque<int32_t> my_threads, const int32_t leader_tid, const int32_t tid){    
    if((leader_tid == -1) or (tid == leader_tid)) {
        CSR::nnz_i = 0;
        for(uint32_t i = 0; i < my_threads.size(); i++) {    
            int32_t t = my_threads[i];
            CSR::nnz_i += (Env::threads[t].idx_nnz - Env::threads[t].off_nnz);
            Env::nnzs[t].push_back(Env::threads[t].idx_nnz - Env::threads[t].off_nnz);
        }
    }
    pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
}

template<typename Weight>
void CSR<Weight>::repopulate(const std::shared_ptr<struct Compressed_Format<Weight>> other_spmat, const uint32_t dis_nnz, const int32_t leader_tid, const int32_t tid) {
    std::shared_ptr<struct CSR<Weight>> other_csr = std::static_pointer_cast<struct CSR<Weight>>(other_spmat);
    
    uint32_t  o_ncols = other_csr->ncols;
    uint32_t  o_nrows = other_csr->nrows;
    uint64_t  o_nnz   = other_csr->nnz;
    uint64_t  o_nnz_i = other_csr->nnz_i;
    uint32_t* o_JA    = other_csr->JA_blk->ptr;
    uint32_t* o_IA    = other_csr->IA_blk->ptr;
    Weight*   o_A     = other_csr->A_blk->ptr;

    //if(CSR::nrows != o_nrows){
    //    fprintf(stderr, "Error: Cannot repopulate CSR\n");
     //   exit(1);
    //}
    
    if(tid == leader_tid) {
        CSR::nnz = o_nnz_i;
        CSR::nnz_i = o_nnz_i;
		CSR::nrows = o_nrows;
		CSR::ncols = o_ncols;
		CSR::IA_blk->reallocate(CSR::nrows+1);
        CSR::IA_blk->clear();
        CSR::JA_blk->reallocate(CSR::nnz_i);
        CSR::JA_blk->clear();
        CSR::A_blk->reallocate(CSR::nnz_i);
        CSR::A_blk->clear();
        
        Compressed_Format<Weight>::nnz = CSR::nnz_i;
        Compressed_Format<Weight>::nnz_i = CSR::nnz_i;
		Compressed_Format<Weight>::nrows = CSR::nrows;
        Compressed_Format<Weight>::ncols = CSR::ncols;
    }    
    pthread_barrier_wait(&Env::thread_barrier);
    
    uint32_t* IA = CSR::IA_blk->ptr;
    uint32_t* JA = CSR::JA_blk->ptr;
    Weight*    A = CSR::A_blk->ptr;

	const uint32_t start_row = Env::threads[tid].start_row;
    const uint32_t end_row   = Env::threads[tid].end_row;

    for(int32_t i = 0; i < tid; i++) {
        IA[start_row+1] += (Env::threads[i].idx_nnz - Env::threads[i].off_nnz);
    }

    for(uint32_t i = start_row; i < end_row; i++) {
        IA[i+1] = (i == start_row) ? IA[i+1] : IA[i];
        uint32_t& k = IA[i+1];
        uint32_t m = (i == start_row) ? dis_nnz : 0;
        for(uint32_t j = o_IA[i] + m; j < o_IA[i + 1]; j++) {
            JA[k] = o_JA[j];
            A[k]  = o_A[j];
            k++;
        }
    }
    
    pthread_barrier_wait(&Env::thread_barrier);
}

template<typename Weight>
void CSR<Weight>::repopulate(const std::shared_ptr<struct Compressed_Format<Weight>> other_spmat, const std::deque<int32_t> my_threads, const int32_t leader_tid,  const int32_t tid) {
    std::shared_ptr<struct CSR<Weight>> other_csr = std::static_pointer_cast<struct CSR<Weight>>(other_spmat);
    
    uint32_t  o_ncols = other_csr->ncols;
    uint32_t  o_nrows = other_csr->nrows;
    uint64_t  o_nnz   = other_csr->nnz;
    uint64_t  o_nnz_i = other_csr->nnz_i;
    uint32_t* o_JA    = other_csr->JA_blk->ptr;
    uint32_t* o_IA    = other_csr->IA_blk->ptr;
    Weight*   o_A     = other_csr->A_blk->ptr;

    //if(CSR::nrows != o_nrows){
    //    fprintf(stderr, "Error: Cannot repopulate CSR\n");
     //   exit(1);
    //}
    //printf("%d %d %lu %lu\n", o_nrows, o_ncols, o_nnz, o_nnz_i);
    if(tid == leader_tid) {
        CSR::nnz = o_nnz_i;
        CSR::nnz_i = o_nnz_i;
		CSR::nrows = o_nrows;
		CSR::ncols = o_ncols;
		CSR::IA_blk->reallocate(CSR::nrows+1);
        CSR::IA_blk->clear();
        CSR::JA_blk->reallocate(CSR::nnz_i);
        CSR::JA_blk->clear();
        CSR::A_blk->reallocate(CSR::nnz_i);
        CSR::A_blk->clear();
        
        Compressed_Format<Weight>::nnz = CSR::nnz_i;
        Compressed_Format<Weight>::nnz_i = CSR::nnz_i;
		Compressed_Format<Weight>::nrows = CSR::nrows;
        Compressed_Format<Weight>::ncols = CSR::ncols;
    }    
    pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
    
    uint32_t* JA = CSR::JA_blk->ptr;
    uint32_t* IA = CSR::IA_blk->ptr;
    Weight*    A = CSR::A_blk->ptr;
	
	/*
	if(tid==leader_tid) {
		for(int t: Env::my_threads[leader_tid]) {
			printf("tid=%d: off=%lu idx=%lu dis=%lu\n", t, Env::threads[t].off_nnz, Env::threads[t].idx_nnz, Env::threads[t].dis_nnz);
		}
		
	}
	pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
	std::exit(0);
	*/
	
	
	
	
	
	
	
	
    const uint32_t start_row = Env::threads[tid].start_row;
    const uint32_t end_row   = Env::threads[tid].end_row;	
   
    for(uint32_t j = 0; j < Env::threads[tid].index; j++) {
        int32_t tt = Env::my_threads[leader_tid][j];
        IA[start_row+1] += (Env::threads[tt].idx_nnz - Env::threads[tt].off_nnz);
    }
    

	//printf("tid=%d start=%d end=%d off=%lu idx=%lu dis=%lu\n", tid, start_row, end_row, Env::threads[tid].off_nnz, Env::threads[tid].idx_nnz, Env::threads[tid].dis_nnz);
    for(uint32_t i = start_row; i < end_row; i++) {
        IA[i+1] = (i == start_row) ? IA[i+1] : IA[i];
        uint32_t& k = IA[i+1];
        uint32_t m = (i == start_row) ? Env::threads[tid].dis_nnz : 0;
        for(uint32_t j = o_IA[i] + m; j < o_IA[i + 1]; j++) {
            JA[k] = o_JA[j];
            A[k]  = o_A[j];
            k++;
        }
    }
    pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
	
	/*
	if(leader_tid == tid) {
		
		
		
		double checksum = 0;
		uint64_t checkcount = 0;
		uint32_t displacement = 0;
		int t = 0;
		for(uint32_t i = 0; i < o_nrows; i++) { 
			int m = 0;
			int32_t tt = Env::my_threads[leader_tid][t];
			if(i == Env::threads[tt].start_row) {
				displacement = Env::threads[tt].dis_nnz; 
				t++;
			}
			else {
				displacement = 0;        
			}
		   //if(!Env::rank)
			 //   std::cout << "j=" << j << ": " << JA[j] << "--" << JA[j + 1] << ": " <<  JA[j + 1] - JA[j] << std::endl;
			//printf("i=%d d=%d\n", i, o_IA[i + 1]-o_IA[i]);
			for(uint32_t j = o_IA[i] + displacement; j < o_IA[i + 1]; j++) {
				(void) o_JA[j];
				(void) o_A[j];
				checksum += o_A[j];
				checkcount++;
			}
		}
		printf("1.checksum=%f checkcount=%lu\n", checksum, checkcount);
		checksum=0;
		checkcount=0;
		for(uint32_t i = 0; i < CSR::nrows; i++) { 
		//printf("i=%d d=%d\n", i, IA[i + 1]-IA[i]);
			for(uint32_t j = IA[i]; j < IA[i + 1]; j++) {
				(void) JA[j];
				(void) A[j];
				checksum += A[j];
				checkcount++;
			}
		}
		printf("2.checksum=%f checkcount=%lu\n", checksum, checkcount);
	}
	
	
	pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
	*/
    //std::exit(0);
    /*
    // It's ugly but I have to :-/
    for(uint32_t j = 0; j < Env::threads[tid].index; j++) {
        int32_t tt = Env::my_threads[leader_tid][j];
        IA[Env::threads[tid].start_row+1] += (Env::threads[tt].idx_nnz - Env::threads[tt].off_nnz);
    }

    const uint32_t start_row = Env::threads[tid].start_row;
    const uint32_t end_row   = Env::threads[tid].end_row;

    for(uint32_t j = o_IA[start_row] + Env::threads[tid].dis_nnz; j < o_IA[start_row + 1]; j++) {
        JA[IA[start_row+1]] = o_JA[j];
        A[IA[start_row+1]]  = o_A[j];
        IA[start_row+1]++;
    }
    
    for(uint32_t i = start_row+1; i < end_row; i++) {
        IA[i+1] = IA[i];
        for(uint32_t j = o_IA[i]; j < o_IA[i + 1]; j++) {
            JA[IA[i+1]] = o_JA[j];
            A[IA[i+1]]  = o_A[j];
            IA[i+1]++;
        }
    }
    pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
    */
}

template<typename Weight>
CSC<Weight>::CSC(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t socket_id) {
    Compressed_Format<Weight>::compression_type = COMPRESSED_FORMAT::_CSC_;
    Compressed_Format<Weight>::nnz = nnz_;
    Compressed_Format<Weight>::nnz_i = nnz_;
    Compressed_Format<Weight>::nrows = nrows_; 
    Compressed_Format<Weight>::ncols = ncols_;
    
    CSC::compression_type = COMPRESSED_FORMAT::_CSC_;
    CSC::nnz = nnz_;
    CSC::nnz_i = nnz_;
    CSC::nrows = nrows_; 
    CSC::ncols = ncols_;
    
    CSC::JA_blk = std::move(std::make_shared<struct Data_Block<uint32_t>>((CSC::ncols + 1), socket_id));
    CSC::IA_blk = std::move(std::make_shared<struct Data_Block<uint32_t>>(CSC::nnz, socket_id));
    CSC::A_blk = std::move(std::make_shared<struct Data_Block<Weight>>(CSC::nnz, socket_id));
}

template<typename Weight>
CSC<Weight>::CSC(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_) {
    Compressed_Format<Weight>::compression_type = COMPRESSED_FORMAT::_CSC_;
    Compressed_Format<Weight>::nnz = nnz_;
    Compressed_Format<Weight>::nnz_i = nnz_;
    Compressed_Format<Weight>::nrows = nrows_; 
    Compressed_Format<Weight>::ncols = ncols_;
    
    CSC::compression_type = COMPRESSED_FORMAT::_CSC_;
    CSC::nnz = nnz_;
    CSC::nnz_i = nnz_;
    CSC::nrows = nrows_; 
    CSC::ncols = ncols_;
    
    CSC::JA_blk = std::move(std::make_shared<struct Data_Block<uint32_t>>((CSC::ncols + 1), Env::rank_socket_id));
    CSC::IA_blk = std::move(std::make_shared<struct Data_Block<uint32_t>>(CSC::nnz, Env::rank_socket_id));
    CSC::A_blk = std::move(std::make_shared<struct Data_Block<Weight>>(CSC::nnz, Env::rank_socket_id));
}
/*
template<typename Weight>
CSC<Weight>::CSC(const std::shared_ptr<struct Compressed_Format<Weight>> other_spmat, const uint32_t start_row, const uint32_t tile_height, const uint32_t start_col, const uint32_t tile_width, const int32_t socket_id) {
    std::shared_ptr<struct CSR<Weight>> other_csr = std::static_pointer_cast<struct CSR<Weight>>(other_spmat);
    uint32_t  o_ncols = other_csr->ncols;
    uint32_t  o_nrows = other_csr->nrows;
    uint64_t  o_nnz   = other_csr->nnz;
    uint64_t  o_nnz_i = other_csr->nnz_i;
    uint32_t* o_JA    = other_csr->JA_blk->ptr;
    uint32_t* o_IA    = other_csr->IA_blk->ptr;
    Weight*   o_A     = other_csr->A_blk->ptr;
    
    Compressed_Format<Weight>::compression_type = COMPRESSED_FORMAT::_CSC_;
    Compressed_Format<Weight>::nnz = o_nnz;
    Compressed_Format<Weight>::nnz_i = o_nnz;
    Compressed_Format<Weight>::nrows = o_nrows; 
    Compressed_Format<Weight>::ncols = o_ncols;
    
    CSC::compression_type = COMPRESSED_FORMAT::_CSC_;
    CSC::nnz = o_nnz;
    CSC::nnz_i = o_nnz;
    CSC::nrows = o_nrows; 
    CSC::ncols = o_ncols;
    
    CSC::JA_blk = std::move(std::make_shared<struct Data_Block<uint32_t>>((CSC::ncols + 1), socket_id));
    CSC::IA_blk = std::move(std::make_shared<struct Data_Block<uint32_t>>(CSC::nnz, socket_id));
    CSC::A_blk = std::move(std::make_shared<struct Data_Block<Weight>>(CSC::nnz, socket_id));
    
    uint32_t* IA = CSC::IA_blk->ptr;
    uint32_t* JA = CSC::JA_blk->ptr;
    Weight* A = CSC::A_blk->ptr;
    
    std::vector<struct Triple<Weight>> triples;
    for(uint32_t i = 0; i < o_nrows; i++) {
        for(uint32_t j = o_IA[i]; j < o_IA[i + 1]; j++) {
            struct Triple<Weight> triple = {i, o_JA[j], o_A[j]};
            triples.push_back(triple);
        }
    }
    
    CSC::populate(triples);
    triples.clear();
    triples.shrink_to_fit();    
}
    
template<typename Weight>
void CSC<Weight>::populate(std::vector<struct Triple<Weight>>& triples) {
    const ColSort<Weight> f_col;
    std::sort(triples.begin(), triples.end(), f_col);  
    
    uint32_t* IA = CSC::IA_blk->ptr;
    uint32_t* JA = CSC::JA_blk->ptr;
    Weight* A = CSC::A_blk->ptr;
    
    uint32_t i = 0;
    uint32_t j = 1; 
    JA[0] = 0;
    for(auto &triple: triples) {
        while((j - 1) != triple.col) {
            j++;
            JA[j] = JA[j - 1];
        }                  
        JA[j]++;
        IA[i] = triple.row;
        A[i] = triple.weight;
        i++;
    }
    
    while(j < CSC::ncols) {
        j++;
        JA[j] = JA[j - 1];
    }
    CSC::nnz_i = CSC::nnz;
}

template<typename Weight>
void CSC<Weight>::populate(std::vector<struct Triple<Weight>>& triples, const uint32_t start_row, const uint32_t tile_height, 
                                                                             const uint32_t start_col, const uint32_t tile_width) {    
    const ColSort<Weight> f_col;
    std::sort(triples.begin(), triples.end(), f_col);   
	
    uint32_t* IA = CSC::IA_blk->ptr;
    uint32_t* JA = CSC::JA_blk->ptr;
    Weight* A = CSC::A_blk->ptr;
    
    uint32_t i = 0;
    uint32_t j = 1; 
    JA[0] = 0;
    for(auto &triple: triples) {
        std::pair pair = std::make_pair(((triple.row - start_row) % tile_height), ((triple.col - start_col) % tile_width));
        while((j - 1) != pair.second) {
            j++;
            JA[j] = JA[j - 1];
        }                  
        JA[j]++;
        IA[i] = pair.first;
        A[i] = triple.weight;
        i++;
		
    }
    
    while(j < CSC::ncols) {
        j++;
        JA[j] = JA[j - 1];
    }
    CSC::nnz_i = CSC::nnz;   
}
*/

template<typename Weight>
void CSC<Weight>::populate(std::vector<struct Triple<Weight>>& triples, const uint32_t tile_height, const uint32_t tile_width) {
    const ColSort<Weight> f_col;
    std::sort(triples.begin(), triples.end(), f_col);  
    
    uint32_t* IA = CSC::IA_blk->ptr;
    uint32_t* JA = CSC::JA_blk->ptr;
    Weight* A = CSC::A_blk->ptr;
    
    uint32_t i = 0;
    uint32_t j = 1; 
    JA[0] = 0;
    for(auto &triple: triples) {
        std::pair pair = std::make_pair((triple.row % tile_height), (triple.col % tile_width));
        while((j - 1) != pair.second) {
            j++;
            JA[j] = JA[j - 1];
        }                  
        JA[j]++;
        IA[i] = pair.first;
        A[i] = triple.weight;
        i++;
    }

    while(j < CSC::ncols) {
        j++;
        JA[j] = JA[j - 1];
    }
    CSC::nnz_i = CSC::nnz;
}
/*
template<typename Weight>
void CSC<Weight>::populate_spa(Weight** spa, const Weight* bias, const uint32_t col, uint64_t& index, const int32_t tid) {
    uint64_t&  k = index;
    uint32_t   c = col + 1;
    uint32_t* IA = CSC::IA_blk->ptr;
    uint32_t* JA = CSC::JA_blk->ptr;
    Weight*    A = CSC::A_blk->ptr;
    Weight*    s = *spa;
    const Weight* b = bias;
    
    // ReLU activation function thresholds 
    const Weight YMIN = 0; 
    const Weight YMAX = 32;
    
    for(uint32_t i = 0; i < CSC::nrows; i++) {
        if(s[i]) {
            s[i] += b[c];
            s[i] = (s[i] < YMIN) ? YMIN : (s[i] > YMAX) ? YMAX : s[i];
			//s[i]=relu(s[i]);
            if(s[i]) {
                IA[k] = i;
                A[k] = s[i];
                k++;
                s[i] = 0;
            }
        }
    }
    JA[c] = k;
	if(tid == Env::nthreads-1) {
		//printf("col=%d c=%d index=%lu\n", col, c, index);
	}
}
*/
template<typename Weight>
void CSC<Weight>::populate_spa(Weight** spa, const Weight* bias, const uint32_t col, uint64_t& index, Weight(*activation_function)(Weight), const int32_t tid) {
    uint64_t&  k = index;
    uint32_t   c = col + 1;
    uint32_t* IA = CSC::IA_blk->ptr;
    uint32_t* JA = CSC::JA_blk->ptr;
    Weight*    A = CSC::A_blk->ptr;
    Weight*    s = *spa;
    const Weight* b = bias;
    
    /* ReLU activation function thresholds */
    //const Weight YMIN = 0; 
    //const Weight YMAX = 32;
    
    for(uint32_t i = 0; i < CSC::nrows; i++) {
        if(s[i]) {
            s[i] += b[c-1];
            //s[i] = (s[i] < YMIN) ? YMIN : (s[i] > YMAX) ? YMAX : s[i];
			s[i]=activation_function(s[i]);
            if(s[i]) {
                IA[k] = i;
                A[k] = s[i];
                k++;
                s[i] = 0;
            }
        }
    }
    JA[c] = k;
	//if(tid == Env::nthreads-1) {
		//printf("col=%d c=%d index=%lu\n", col, c, index);
	//}
}

template<typename Weight>
void CSC<Weight>::walk_dxm(const bool one_rank, const int32_t leader_tid, const int32_t tid) {  
    if(tid == leader_tid) {
        uint32_t* IA = CSC::IA_blk->ptr;
        uint32_t* JA = CSC::JA_blk->ptr;
        Weight*    A = CSC::A_blk->ptr;
        
        double checksum = 0;
        uint64_t checkcount = 0;
        for(uint32_t j = 0; j < CSC::ncols; j++) {
            //if(!Env::rank)
            //    std::cout << "j=" << j << "," << j << ": " << JA[j] << "--" << JA[j + 1] << ": " <<  JA[j + 1] - JA[j] << std::endl;
            for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                (void) IA[i];
                (void) A[i];
                checksum += A[i];
                checkcount++;
                //if(!Env::rank)
                //    std::cout << "    i=" << i << ",i=" << IA[i] <<  ",value=" << A[i] << std::endl;
            }
        }

        Env::barrier();
        if(one_rank) {
            Logging::print(Logging::LOG_LEVEL::INFO, "Iteration=%d, Total checksum=%f, Total count=%d\n", Env::iteration, checksum, checkcount);
        }
        else {
            uint64_t nnz_ = CSC::nnz_i;
            uint64_t nnz_ranks = 0;
            double sum_ranks = 0;
            uint64_t count_ranks = 0;
            MPI_Allreduce(&nnz_, &nnz_ranks, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&checksum, &sum_ranks, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&checkcount, &count_ranks, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);

            if(count_ranks != nnz_ranks) {
                Logging::print(Logging::LOG_LEVEL::WARN, "Compression checksum warning!!\n");
				Logging::print(Logging::LOG_LEVEL::WARN, "Compression checksum warning!! (%lu != %lu)\n", count_ranks, nnz_ranks);
            }
            Logging::print(Logging::LOG_LEVEL::INFO, "tid=%d CSC: Iteration=%d, Total checksum=%f, Total count=%d\n", tid, Env::iteration, sum_ranks, count_ranks);
        } 
    }    
}

template<typename Weight>
void CSC<Weight>::walk_dxd(const bool one_rank, const int32_t leader_tid, const int32_t tid) {
    uint32_t* IA = CSC::IA_blk->ptr;
    uint32_t* JA = CSC::JA_blk->ptr;
    Weight*    A = CSC::A_blk->ptr;
    
    double&   checksum   = Env::counters[tid].checksum;
    uint64_t& checkcount = Env::counters[tid].checkcount;
    uint64_t& checknnz   = Env::counters[tid].checknnz;
    
    checksum   = 0;
    checkcount = 0;    
    checknnz   = CSC::nnz_i;

    for(uint32_t j = 0; j < CSC::ncols; j++) { 
        //if(!Env::rank and !tid)
           // std::cout << "j=" << j << "," << j+1 << ": " << JA[j] << "--" << JA[j + 1] << ": " <<  JA[j + 1] - JA[j] << std::endl;    
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            (void) IA[i];
            (void) A[i];
            checksum += A[i];
            checkcount++;
            //if(!Env::rank and !tid)
            //std::cout << "    i=" << i << ",i=" << IA[i] <<  ",value=" << A[i] << std::endl;
        }
    }   
 
	/*
	std::vector<std::vector<std::pair<int,Weight>>> rows(CSC::nrows);
	for(uint32_t j = 0; j < CSC::ncols; j++) { 
        //if(!Env::rank and !tid)
           // std::cout << "j=" << j << "," << j+1 << ": " << JA[j] << "--" << JA[j + 1] << ": " <<  JA[j + 1] - JA[j] << std::endl;    
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
			rows[IA[i]].push_back({j,A[i]});
            //(void) IA[i];
            //(void) A[i];
            //checksum += A[i];
            //checkcount++;
            //if(!Env::rank and !tid)
            //std::cout << "    i=" << i << ",i=" << IA[i] <<  ",value=" << A[i] << std::endl;
        }
    }  
	 
	
	for(uint32_t i = 0; i < rows.size(); i++){
		printf("i=%d: ", i);
		for(auto c: rows[i]) {
			printf("j=%d v=%f\n", c.first, c.second);
			if(c.first>50) break;
		}
		printf("\n");
	}
	*/
	
	
	
    Env::barrier();
    pthread_barrier_wait(&Env::thread_barrier);
    if(tid == leader_tid) {
        double     sum_threads = 0;
        uint64_t count_threads = 0;
        uint64_t   nnz_threads = 0;
        
        for(auto it = Env::counters.begin(); it != Env::counters.end(); it++) {
            sum_threads   += (*it).checksum;
            count_threads += (*it).checkcount;
            nnz_threads   += (*it).checknnz;
        }
        
        if(one_rank) {
            Logging::print(Logging::LOG_LEVEL::INFO, "CSC: Iteration=%d, Total checksum=%f, Total count=%d\n", Env::iteration, sum_threads, count_threads);
        }
        else {
            double     sum_ranks = 0;
            uint64_t count_ranks = 0;
            
            MPI_Allreduce(&sum_threads, &sum_ranks, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&count_threads, &count_ranks, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
            
            if(count_threads != nnz_threads) {
                Logging::print(Logging::LOG_LEVEL::WARN, "Compression checksum warning!! (%lu != %lu)\n", count_threads, nnz_threads);
            }
            Logging::print(Logging::LOG_LEVEL::INFO, "CSC: Iteration=%d, Total checksum=%f, Total count=%d\n", Env::iteration, sum_ranks, count_ranks);
        }
    }


	
}

template<typename Weight>
void CSC<Weight>::reallocate(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t leader_tid, const int32_t tid) {
    //if(CSC::ncols != ncols_) {
    //    Logging::print(Logging::LOG_LEVEL::ERROR, "Cannot reallocate CSC A[%d %d] -> A[%d %d]\n", CSC::nrows, CSC::ncols, nrows_, ncols_);
    //    std::exit(Env::finalize());     
    //}
    if((leader_tid == -1) or (tid == leader_tid)) {
		//Logging::print(Logging::LOG_LEVEL::ERROR, " reallocate CSC A[%d %d] -> A[%d %d] %lu\n", CSC::nrows, CSC::ncols, nrows_, ncols_, nnz_);
        CSC::nnz = nnz_;
        CSC::nnz_i = 0;
        CSC::nrows = nrows_; 
        CSC::ncols = ncols_;
        CSC::JA_blk->reallocate(CSC::ncols+1);
		CSC::JA_blk->clear();
        CSC::IA_blk->reallocate(CSC::nnz);
		CSC::IA_blk->clear();
        CSC::A_blk->reallocate(CSC::nnz);
        CSC::A_blk->clear();
        
        Compressed_Format<Weight>::nnz = nnz_;
        Compressed_Format<Weight>::nnz_i = 0;
        Compressed_Format<Weight>::nrows = nrows_; 
        Compressed_Format<Weight>::ncols = ncols_;
    }
}

template<typename Weight>
void CSC<Weight>::adjust(const int32_t tid){
    CSC::nnz_i = Env::threads[tid].idx_nnz;
    //Env::nnzs[tid].push_back(Env::threads[tid].idx_nnz);
}

template<typename Weight>
void CSC<Weight>::adjust(const int32_t leader_tid, const int32_t tid){
    if((leader_tid == -1) or (tid == leader_tid)) {
        CSC::nnz_i = 0;
        for(uint32_t i = 0; i < Env::threads.size(); i++) {    
            CSC::nnz_i += (Env::threads[i].idx_nnz - Env::threads[i].off_nnz);
            //Env::nnzs[i].push_back(Env::threads[i].idx_nnz - Env::threads[i].off_nnz);
        }
    }
    pthread_barrier_wait(&Env::thread_barrier);
}

template<typename Weight>
void CSC<Weight>::adjust(const std::deque<int32_t> my_threads, const int32_t leader_tid, const int32_t tid){    
    if((leader_tid == -1) or (tid == leader_tid)) {
        CSC::nnz_i = 0;
        for(uint32_t i = 0; i < my_threads.size(); i++) {    
            int32_t t = my_threads[i];
            CSC::nnz_i += (Env::threads[t].idx_nnz - Env::threads[t].off_nnz);
           // Env::nnzs[t].push_back(Env::threads[t].idx_nnz - Env::threads[t].off_nnz);
        }
    }
    pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
}

template<typename Weight>
void CSC<Weight>::repopulate(const std::shared_ptr<struct Compressed_Format<Weight>> other_spmat, const uint32_t dis_nnz, const int32_t leader_tid, const int32_t tid) {
    std::shared_ptr<struct CSC<Weight>> other_csc = std::static_pointer_cast<struct CSC<Weight>>(other_spmat);
    
    uint32_t  o_ncols = other_csc->ncols;
    uint32_t  o_nrows = other_csc->nrows;
    uint64_t  o_nnz   = other_csc->nnz;
    uint64_t  o_nnz_i = other_csc->nnz_i;
    uint32_t* o_JA    = other_csc->JA_blk->ptr;
    uint32_t* o_IA    = other_csc->IA_blk->ptr;
    Weight*   o_A     = other_csc->A_blk->ptr;
	
	
	/*
	pthread_barrier_wait(&Env::thread_barrier);
	if(tid == leader_tid) {
		printf("tid=%d %d %lu\n", tid, CSC::ncols, JA_blk->nitems);
		 double checksum = 0;
        uint64_t checkcount = 0;
		
		int t = 0;
		for(uint32_t j = 0; j < o_ncols; j++) {
			int m = 0;
			if(j == Env::threads[t].start_col) {m+=Env::threads[t].dis_nnz; t++;}
			//if(j==o_ncols-1)
				//printf("j=%d %d <-> %d: %d\n", j, o_JA[j], o_JA[j + 1], o_JA[j + 1]-o_JA[j]);
			for(uint32_t i = o_JA[j]+m; i < o_JA[j + 1]; i++) {
                (void) o_IA[i];
                (void) o_A[i];
                checksum += o_A[i];
                checkcount++;
                //if(!Env::rank)
                //    std::cout << "    i=" << i << ",i=" << IA[i] <<  ",value=" << A[i] << std::endl;
            }
		}
		
		printf("0.%f %lu\n", checksum, checkcount);
		printf("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
	}
	*/
	
	

    //if(CSC::ncols != o_ncols){
	//	Logging::print(Logging::LOG_LEVEL::ERROR, "Error: Cannot repopulate CSC A[%d %d] != C[%d %d]\n", CSC::nrows, CSC::ncols, o_nrows, o_ncols);
	//	std::exit(Env::finalize());
    //}
    //printf("%d %d\n", CSC::ncols, o_ncols);
    if(tid == leader_tid) {
		//Logging::print(Logging::LOG_LEVEL::ERROR, "Error: Cannot repopulate CSC A[%d %d] != C[%d %d]\n", CSC::nrows, CSC::ncols, o_nrows, o_ncols);
		CSC::nnz = o_nnz_i;
        CSC::nnz_i = o_nnz_i;
		CSC::ncols = o_ncols;
		CSC::nrows = o_nrows;
		//printf("%lu %lu %lu\n", CSC::JA_blk->nbytes, CSC::IA_blk->nbytes, CSC::A_blk->nbytes, CSC::JA_blk );
		CSC::JA_blk->reallocate(CSC::ncols+1);
        CSC::JA_blk->clear();
        CSC::IA_blk->reallocate(CSC::nnz_i);
        CSC::IA_blk->clear();
        CSC::A_blk->reallocate(CSC::nnz_i);
        CSC::A_blk->clear();
		//printf("%lu %lu %lu\n", CSC::JA_blk->nbytes, CSC::IA_blk->nbytes, CSC::A_blk->nbytes );
        Compressed_Format<Weight>::nnz = CSC::nnz_i;
        Compressed_Format<Weight>::nnz_i = CSC::nnz_i;
		Compressed_Format<Weight>::nrows = CSC::nrows;
		Compressed_Format<Weight>::ncols = CSC::ncols;
    }
    pthread_barrier_wait(&Env::thread_barrier);
    
    uint32_t* JA = CSC::JA_blk->ptr;
    uint32_t* IA = CSC::IA_blk->ptr;
    Weight*    A = CSC::A_blk->ptr;
	
	const uint32_t start_col = Env::threads[tid].start_col;
    const uint32_t end_col   = Env::threads[tid].end_col;
	//printf("%d [%d %d] [%d %d]\n", tid, start_col, end_col, CSC::nrows, CSC::ncols);
    for(int32_t i = 0; i < tid; i++) {
        JA[start_col+1] += (Env::threads[i].idx_nnz - Env::threads[i].off_nnz);
    }
	//if(tid==Env::nthreads-1) printf("%d\n", JA[start_col+1]); //13057400
	//printf(">>>%d %d %d\n", tid, JA[end_col], JA[end_col]);
    for(uint32_t j = start_col; j < end_col; j++) {
        JA[j+1] = (j == start_col) ? JA[j+1] : JA[j];
        uint32_t& k = JA[j+1];
        uint32_t m = (j == start_col) ? dis_nnz : 0;
		//if(tid == Env::nthreads-1) printf("%d %d\n", j, k);
        for(uint32_t i = o_JA[j] + m; i < o_JA[j + 1]; i++) {
            IA[k] = o_IA[i];
            A[k]  = o_A[i];
            k++;
        }
    }
	

	
	
	/*
    pthread_barrier_wait(&Env::thread_barrier);
	if(tid == leader_tid) {
		printf("tid=%d %d %lu\n", tid, CSC::ncols, JA_blk->nitems);
		  double checksum = 0;
        uint64_t checkcount = 0;
		
	
		
		printf("0.%f %lu\n", checksum, checkcount);
		checksum = 0;
		checkcount=0;
        for(uint32_t j = 0; j < CSC::ncols; j++) {
            //if(!Env::rank)
            //    std::cout << "j=" << j << "," << j << ": " << JA[j] << "--" << JA[j + 1] << ": " <<  JA[j + 1] - JA[j] << std::endl;
            for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                (void) IA[i];
                (void) A[i];
                checksum += A[i];
                checkcount++;
                //if(!Env::rank)
                //    std::cout << "    i=" << i << ",i=" << IA[i] <<  ",value=" << A[i] << std::endl;
            }
        }
		printf("1.%f %lu %d\n", checksum, checkcount, o_ncols);
	}
	*/
}

template<typename Weight>
void CSC<Weight>::repopulate(const std::shared_ptr<struct Compressed_Format<Weight>> other_spmat, const std::deque<int32_t> my_threads, const int32_t leader_tid,  const int32_t tid) {
    std::shared_ptr<struct CSC<Weight>> other_csc = std::static_pointer_cast<struct CSC<Weight>>(other_spmat);
    
    uint32_t  o_ncols = other_csc->ncols;
    uint32_t  o_nrows = other_csc->nrows;
    uint64_t  o_nnz   = other_csc->nnz;
    uint64_t  o_nnz_i = other_csc->nnz_i;
    uint32_t* o_JA    = other_csc->JA_blk->ptr;
    uint32_t* o_IA    = other_csc->IA_blk->ptr;
    Weight*   o_A     = other_csc->A_blk->ptr;

    //if(CSC::ncols != o_ncols){
    //    Logging::print(Logging::LOG_LEVEL::ERROR, "Error: Cannot repopulate CSC A[%d %d] != C[%d %d]\n", CSC::nrows, CSC::ncols, o_nrows, o_ncols);
	//	std::exit(Env::finalize());
    //}

    if(tid == leader_tid) {
		CSC::nnz = o_nnz_i;
        CSC::nnz_i = o_nnz_i;
		CSC::ncols = o_ncols;
		CSC::nrows = o_nrows;
		CSC::JA_blk->reallocate(CSC::ncols+1);
        CSC::JA_blk->clear();
        CSC::IA_blk->reallocate(CSC::nnz_i);
        CSC::IA_blk->clear();
        CSC::A_blk->reallocate(CSC::nnz_i);
        CSC::A_blk->clear();
        Compressed_Format<Weight>::nnz = CSC::nnz_i;
        Compressed_Format<Weight>::nnz_i = CSC::nnz_i;
		Compressed_Format<Weight>::nrows = CSC::nrows;
		Compressed_Format<Weight>::ncols = CSC::ncols;
    }
    pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
    
    uint32_t* JA = CSC::JA_blk->ptr;
    uint32_t* IA = CSC::IA_blk->ptr;
    Weight*    A = CSC::A_blk->ptr;

    /*
    for(uint32_t j = 0; j < Env::threads[tid].index; j++) {
        int32_t tt = Env::my_threads[leader_tid][j];
        JA[Env::threads[tid].start_col+1] += (Env::threads[tt].idx_nnz - Env::threads[tt].off_nnz);
    }
    
    const uint32_t start_col = Env::threads[tid].start_col;
    const uint32_t end_col   = Env::threads[tid].end_col;

    for(uint32_t j = start_col; j < end_col; j++) {
        JA[j+1] = (j == start_col) ? JA[j+1] : JA[j];
        uint32_t& k = JA[j+1];
        uint32_t m = (j == start_col) ? Env::threads[tid].dis_nnz : 0;
        for(uint32_t i = o_JA[j] + m; i < o_JA[j + 1]; i++) {
            IA[k] = o_IA[i];
            A[k]  = o_A[i];
            k++;
        }
    }
    pthread_barrier_wait(&Env::thread_barrier);
    */
	/*
    if(tid==leader_tid) {
		for(int t: Env::my_threads[leader_tid]) {
			printf("tid=%d: off=%lu idx=%lu dis=%lu\n", t, Env::threads[t].off_nnz, Env::threads[t].idx_nnz, Env::threads[t].dis_nnz);
		}
		
	}
	pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
	std::exit(0);
*/
    
    // It's ugly but I have to :-/
	const uint32_t start_col = Env::threads[tid].start_col;
    const uint32_t end_col   = Env::threads[tid].end_col;
	
    for(uint32_t j = 0; j < Env::threads[tid].index; j++) {
        int32_t tt = Env::my_threads[leader_tid][j];
        JA[start_col+1] += (Env::threads[tt].idx_nnz - Env::threads[tt].off_nnz);
    }
    
	printf("#####tid=%d, JA[start_col+1]=%d\n", tid, JA[start_col+1]);
    for(uint32_t j = start_col; j < end_col; j++) {
        JA[j+1] = (j == start_col) ? JA[j+1] : JA[j];
        uint32_t& k = JA[j+1];
        uint32_t m = (j == start_col) ? Env::threads[tid].dis_nnz : 0;
		//if(tid == Env::nthreads-1) printf("%d %d\n", j, k);
        for(uint32_t i = o_JA[j] + m; i < o_JA[j+1]; i++) {
            IA[k] = o_IA[i];
            A[k]  = o_A[i];
            k++;
        }
    }
	
    pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
	printf("#####tid=%d start=%d end=%d index=%lu offset=%lu dis=%lu JA=%d\n", tid, start_col, end_col, Env::threads[tid].idx_nnz, Env::threads[tid].off_nnz, Env::threads[tid].dis_nnz, JA[end_col]);
	
		//const uint32_t start_col = Env::threads[tid].start_col;
    //const uint32_t end_col   = Env::threads[tid].end_col;
	//printf("%d [%d %d] [%d %d]\n", tid, start_col, end_col, CSC::nrows, CSC::ncols);
    //for(int32_t i = 0; i < tid; i++) {
      //  JA[start_col+1] += (Env::threads[i].idx_nnz - Env::threads[i].off_nnz);
    //}
	//if(tid==Env::nthreads-1) printf("%d\n", JA[start_col+1]); //13057400
	//printf(">>>%d %d %d\n", tid, JA[end_col], JA[end_col]);

	
	

	/*
    for(uint32_t i = o_JA[start_col] + Env::threads[tid].dis_nnz; i < o_JA[start_col + 1]; i++) {
        IA[JA[start_col+1]] = o_IA[i];
        A[JA[start_col+1]]  = o_A[i];
        JA[start_col+1]++;
    }
    
    for(uint32_t j = start_col+1; j < end_col; j++) {
        JA[j+1] = JA[j];
        for(uint32_t i = o_JA[j]; i < o_JA[j + 1]; i++) {
            IA[JA[j+1]] = o_IA[i];
            A[JA[j+1]]  = o_A[i];
            JA[j+1]++;
        }
    }
	pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
	*/
    
}

template<typename Weight>
void CSC<Weight>::walk_dxm1(const bool one_rank, const int32_t leader_tid, const int32_t tid) {  
    if(tid == leader_tid) {
        uint32_t* IA = CSC::IA_blk->ptr;
        uint32_t* JA = CSC::JA_blk->ptr;
        Weight*    A = CSC::A_blk->ptr;
        
        double checksum = 0;
        uint64_t checkcount = 0;
         uint32_t displacement = 0;
        int t = 0;
        for(uint32_t j = 0; j < CSC::ncols; j++) { 
            int32_t tt = Env::my_threads[leader_tid][t];
            if(j == Env::threads[tt].start_col) {
                displacement = Env::threads[tt].dis_nnz; 
                tt++;
            }
            else {
                displacement = 0;        
            }
           //if(!Env::rank)
             //   std::cout << "j=" << j << ": " << JA[j] << "--" << JA[j + 1] << ": " <<  JA[j + 1] - JA[j] << std::endl;

            for(uint32_t i = JA[j] + displacement; i < JA[j + 1]; i++) {
                (void) IA[i];
                (void) A[i];
                checksum += A[i];
                checkcount++;
            }
        }

        Env::barrier();
        if(one_rank) {
            Logging::print(Logging::LOG_LEVEL::INFO, "Iteration=%d, Total checksum=%f, Total count=%d 1\n", Env::iteration, checksum, checkcount);
        }
        else {
            uint64_t nnz_ = CSC::nnz_i;
            uint64_t nnz_ranks = 0;
            double sum_ranks = 0;
            uint64_t count_ranks = 0;
            MPI_Allreduce(&nnz_, &nnz_ranks, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&checksum, &sum_ranks, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&checkcount, &count_ranks, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);

            if(count_ranks != nnz_ranks) {
                Logging::print(Logging::LOG_LEVEL::WARN, "Compression checksum warning!! 1\n");
            }
            Logging::print(Logging::LOG_LEVEL::INFO, "Iteration=%d, Total checksum=%f, Total count=%d 1\n", Env::iteration, sum_ranks, count_ranks);
        } 
    }    
}

#endif