/*
 * spmat.hpp: Sparse Matrix implementation 
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com
 */
 
#ifndef SPMAT_HPP
#define SPMAT_HPP

#include <numeric>

#include "allocator.hpp"
#include "triple.hpp"
#include "env.hpp"

template<typename Weight>
struct CSC {
    public:
        CSC(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_);
        ~CSC(){};
        
        void populate(const std::vector<struct Triple<Weight>> triples, const uint32_t tile_height, const uint32_t tile_width);
        void populate(const std::vector<struct Triple<Weight>> triples, const uint32_t start_row, const uint32_t tile_height, const uint32_t start_col, const uint32_t tile_width);
        void populate_spa(Weight** spa, const Weight* bias, const uint32_t col,  uint64_t& index, const int32_t tid);
        void walk_dxm(const bool one_rank, const int32_t leader_tid, const int32_t tid);
        void walk_dxd(const bool one_rank, const int32_t leader_tid, const int32_t tid);
        void reallocate(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t leader_tid, const int32_t tid);
        void adjust(const int32_t tid);
        void adjust(const int32_t leader_tid, const int32_t tid);
        void repopulate(const std::shared_ptr<struct CSC<Weight>> other, const uint32_t start_col, const uint32_t end_col, const uint32_t dis_nnz, const int32_t tid);
        void repopulate(const std::shared_ptr<struct CSC<Weight>> other, const int32_t tid, const int32_t leader, const std::vector<int32_t> my_follower_threads);

        uint64_t nnz   = 0;
        uint64_t nnz_i = 0;
        uint32_t nrows = 0;
        uint32_t ncols = 0;
        
        std::shared_ptr<struct Data_Block<uint32_t>> IA_blk;
        std::shared_ptr<struct Data_Block<uint32_t>> JA_blk;
        std::shared_ptr<struct Data_Block<Weight>>   A_blk;
};


template<typename Weight>
CSC<Weight>::CSC(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_) {
    CSC::nnz = nnz_;
    CSC::nrows = nrows_; 
    CSC::ncols = ncols_;
    
    CSC::JA_blk = std::move(std::make_shared<struct Data_Block<uint32_t>>((CSC::ncols + 1), Env::rank_socket_id));
    CSC::IA_blk = std::move(std::make_shared<struct Data_Block<uint32_t>>(CSC::nnz, Env::rank_socket_id));
    CSC::A_blk = std::move(std::make_shared<struct Data_Block<Weight>>(CSC::nnz, Env::rank_socket_id));
}

/* For fixed height and width s.t. 
   tile height = matrix nrows / nranks and 
   tile height = matrix ncols / nranks  */
template<typename Weight>
void CSC<Weight>::populate(const std::vector<struct Triple<Weight>> triples, const uint32_t tile_height, const uint32_t tile_width) {
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

/* Tile height and width are not necessarily 
   a multiple matrix nrows and ncols */
template<typename Weight>
void CSC<Weight>::populate(const std::vector<struct Triple<Weight>> triples, const uint32_t start_row, const uint32_t tile_height, 
                                                                             const uint32_t start_col, const uint32_t tile_width) {
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

template<typename Weight>
void CSC<Weight>::populate_spa(Weight** spa, const Weight* bias, const uint32_t col, uint64_t& index, const int32_t tid) {
    uint64_t&  k = index;
    uint32_t   c = col + 1;
    uint32_t* IA = CSC::IA_blk->ptr;
    uint32_t* JA = CSC::JA_blk->ptr;
    Weight*       A = CSC::A_blk->ptr;
    Weight*       s = *spa;
    const Weight* b = bias;
    
    /* ReLU activation function thresholds */
    Weight YMIN = 0; 
    Weight YMAX = 32;
    
    JA[c] = k;
    for(uint32_t i = 0; i < CSC::nrows; i++) {
        if(s[i]) {
            s[i] += b[c];
            if(s[i] < YMIN) {
                s[i] = YMIN;
            }
            else if(s[i] > YMAX) {
                s[i] = YMAX;
            }
            if(s[i]) {
                JA[c]++;
                IA[k] = i;
                A[k] = s[i];
                k++;
                s[i] = 0;
            }
        }
    }
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
            //    std::cout << "j=" << j << "," << j << ": " << JA[j] << "--" << JA[j + 1] << ": " <<  JA[j + 1] - JA[j] << std::endl
            for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                (void) IA[i];
                (void) A[i];
                checksum += A[i];
                checkcount++;
                //std::cout << "    i=" << i << ",i=" << IA[i] <<  ",value=" << A[i] << std::endl;
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
            }
            Logging::print(Logging::LOG_LEVEL::INFO, "Iteration=%d, Total checksum=%f, Total count=%d\n", Env::iteration, sum_ranks, count_ranks);
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
        //    std::cout << "j=" << j << "," << j << ": " << JA[j] << "--" << JA[j + 1] << ": " <<  JA[j + 1] - JA[j] << std::endl;    
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            (void) IA[i];
            (void) A[i];
            checksum += A[i];
            checkcount++;
            //if(!Env::rank and !tid)
            //std::cout << "    i=" << i << ",i=" << IA[i] <<  ",value=" << A[i] << std::endl;
        }
    }   
    
    Env::barrier();
    pthread_barrier_wait(&Env::thread_barrier);
    if(tid == leader_tid) {
        double     sum_threads = 0;//std::accumulate(checksum.begin(),   checksum.end(),   0);
        uint64_t count_threads = 0;//std::accumulate(checkcount.begin(), checkcount.end(), 0);
        uint64_t   nnz_threads = 0;//std::accumulate(checknnz.begin(), checknnz.end(), 0);
        
        //std::vector<struct Env::thread_struct>::iterator it;
        for(auto it = Env::counters.begin(); it != Env::counters.end(); it++) {
            sum_threads   += (*it).checksum;
            count_threads += (*it).checkcount;
            nnz_threads   += (*it).checknnz;
        }
        
        
        if(one_rank) {
            Logging::print(Logging::LOG_LEVEL::INFO, "Iteration=%d, Total checksum=%f, Total count=%d\n", Env::iteration, sum_threads, count_threads);
        }
        else {
            double     sum_ranks = 0;
            uint64_t count_ranks = 0;
            
            MPI_Allreduce(&sum_threads, &sum_ranks, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&count_threads, &count_ranks, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
            
            if(count_threads != nnz_threads) {
                Logging::print(Logging::LOG_LEVEL::WARN, "Compression checksum warning!! (%lu != %lu)\n", count_threads, nnz_threads);
            }
            Logging::print(Logging::LOG_LEVEL::INFO, "Iteration=%d, Total checksum=%f, Total count=%d\n", Env::iteration, sum_ranks, count_ranks);
        }
    }
}

template<typename Weight>
void CSC<Weight>::reallocate(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t leader_tid, const int32_t tid) {
    if(CSC::ncols != ncols_) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Cannot reallocate.\n");
        std::exit(Env::finalize());     
    }
    if((leader_tid == -1) or (tid == leader_tid)) {
        CSC::nnz = nnz_;
        CSC::nnz_i = 0;
        CSC::nrows = nrows_; 
        CSC::ncols = ncols_;
        
        CSC::IA_blk->reallocate(CSC::nnz);
        CSC::A_blk->reallocate(CSC::nnz);
            
        CSC::JA_blk->clear();
        CSC::IA_blk->clear();
        CSC::A_blk->clear();
    }
}

template<typename Weight>
void CSC<Weight>::adjust(const int32_t tid){
    CSC::nnz_i = Env::threads[tid].idx_nnz;
}

template<typename Weight>
void CSC<Weight>::adjust(const int32_t leader_tid, const int32_t tid){    
    //uint32_t displacement = (tid == 0) ? 0 : Env::offset_nnz[tid] - Env::index_nnz[tid-1];
    //Env::displacement_nnz[tid] = displacement;
    Env::threads[tid].dis_nnz = (tid == 0) ? 0 : Env::threads[tid].off_nnz - Env::threads[tid-1].idx_nnz;
    pthread_barrier_wait(&Env::thread_barrier);
    if((leader_tid == -1) or (tid == leader_tid)) {
        //CSC::nnz_i = 0;
        //for(int32_t i = 0; i < Env::nthreads; i++) {    
        //    CSC::nnz_i += (Env::index_nnz[i] - Env::offset_nnz[i]);
        //}
        CSC::nnz_i = 0;
        //std::vector<struct Env::thread_struct>::iterator it;
        for(auto it = Env::threads.begin(); it != Env::threads.end(); it++) {
            CSC::nnz_i += ((*it).idx_nnz - (*it).off_nnz);
        }        
    }
    pthread_barrier_wait(&Env::thread_barrier);
}

template<typename Weight>
void CSC<Weight>::repopulate(const std::shared_ptr<struct CSC<Weight>> other, const uint32_t start_col, const uint32_t end_col, const uint32_t dis_nnz, const int32_t tid) {
    uint32_t  o_ncols = other->ncols;
    uint32_t  o_nrows = other->nrows;
    uint64_t  o_nnz   = other->nnz;
    uint64_t  o_nnz_i = other->nnz_i;
    uint32_t* o_JA    = other->JA_blk->ptr;
    uint32_t* o_IA    = other->IA_blk->ptr;
    Weight*   o_A     = other->A_blk->ptr;

    if(CSC::ncols != o_ncols){
        fprintf(stderr, "Error: Cannot repopulate CSC\n");
        exit(1);
    }
    
    if(!tid) {
        CSC::nnz = o_nnz_i;
        CSC::nnz_i = o_nnz_i;
        CSC::JA_blk->clear();
        CSC::IA_blk->reallocate(CSC::nnz_i);
        CSC::IA_blk->clear();
        CSC::A_blk->reallocate(CSC::nnz_i);
        CSC::A_blk->clear();
    }
    pthread_barrier_wait(&Env::thread_barrier);
    
    uint32_t* JA = CSC::JA_blk->ptr;
    uint32_t* IA = CSC::IA_blk->ptr;
    Weight*    A = CSC::A_blk->ptr;

    //uint32_t start_col = Env::start_col[tid];
    //uint32_t end_col = Env::end_col[tid];
    //for(int32_t i = 0; i < tid; i++) {
      //  JA[start_col+1] += (Env::index_nnz[i] - Env::offset_nnz[i]);
    //}
    
    for(int32_t i = 0; i < tid; i++) {
        JA[start_col+1] += (Env::threads[i].idx_nnz - Env::threads[i].off_nnz);
    }

    for(uint32_t j = start_col; j < end_col; j++) {
        JA[j+1] = (j == start_col) ? JA[j+1] : JA[j];
        uint32_t& k = JA[j+1];
        //uint32_t m = (j == start_col) ? Env::displacement_nnz[tid] : 0;
        uint32_t m = (j == start_col) ? dis_nnz : 0;
        for(uint32_t i = o_JA[j] + m; i < o_JA[j + 1]; i++) {
            IA[k] = o_IA[i];
            A[k]  = o_A[i];
            k++;
        }
    }
    pthread_barrier_wait(&Env::thread_barrier);
}

template<typename Weight>
void CSC<Weight>::repopulate(const std::shared_ptr<struct CSC<Weight>> other, const int32_t tid, const int32_t leader, const std::vector<int32_t> my_follower_threads) {
    uint32_t  o_ncols = other->ncols;
    uint32_t  o_nrows = other->nrows;
    uint64_t  o_nnz   = other->nnz;
    uint64_t  o_nnz_i = other->nnz_i;
    uint32_t* o_JA    = other->JA_blk->ptr;
    uint32_t* o_IA    = other->IA_blk->ptr;
    Weight*   o_A     = other->A_blk->ptr;

    if(CSC::ncols != o_ncols){// or (CSC::nrows != o_nrows)) {
        fprintf(stderr, "Error: Cannot repopulate CSC\n");
        exit(1);
    }
    

    
    if(tid == leader) {
        CSC::nnz = o_nnz_i;
        CSC::nnz_i = o_nnz_i;
        CSC::JA_blk->clear();
        CSC::IA_blk->reallocate(CSC::nnz_i);
        CSC::IA_blk->clear();
        CSC::A_blk->reallocate(CSC::nnz_i);
        CSC::A_blk->clear();
    }    
    
    
    pthread_barrier_wait(&Env::thread_barriers[leader]);
    

    
    uint32_t* JA = CSC::JA_blk->ptr;
    uint32_t* IA = CSC::IA_blk->ptr;
    Weight*    A = CSC::A_blk->ptr;
    
    if(tid == leader) {
        JA[0] = 0;
        JA[1] = 0;
        for(uint32_t i = 0; i < my_follower_threads.size(); i++) {
            int32_t t = my_follower_threads[i];
            uint32_t start_col = Env::follower_threads_info[leader][t].start_col;
            for(uint32_t j = 0; j < i; j++) {
                int32_t tt = my_follower_threads[j];
                JA[start_col+1] += (Env::index_nnz[tt] - Env::offset_nnz[tt]);
            }
        }
    }
    
        
        
        
    
    //else {
      //  pthread_mutex_lock(&Env::thread_mutexes[leader]);
        //pthread_cond_wait(&Env::thread_conds[leader], &Env::thread_mutexes[leader]);  
        //pthread_mutex_unlock(&Env::thread_mutexes[leader]);
    //}
    pthread_barrier_wait(&Env::thread_barriers[leader]);
    
    const uint32_t start_col = Env::follower_threads_info[leader][tid].start_col;
    const uint32_t end_col   = Env::follower_threads_info[leader][tid].end_col;

    for(uint32_t j = start_col; j < end_col; j++) {
        JA[j+1] = (j == start_col) ? JA[j+1] : JA[j];
        uint32_t& k = JA[j+1];
        uint32_t m = (j == start_col) ? Env::displacement_nnz[tid] : 0;
        for(uint32_t i = o_JA[j] + m; i < o_JA[j + 1]; i++) {
            IA[k] = o_IA[i];
            A[k]  = o_A[i];
            k++;
        }
    }
    pthread_barrier_wait(&Env::thread_barriers[leader]);
}


#endif
