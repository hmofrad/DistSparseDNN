/*
 * spmat.hpp: Sparse Matrix implementation 
 * (c) Mohammad Hasanzadeh Mofrad, 2020
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
        CSC(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t socket_id);
        ~CSC(){};
        
        void populate(const std::vector<struct Triple<Weight>> triples, const uint32_t tile_height, const uint32_t tile_width);
        void populate(const std::vector<struct Triple<Weight>> triples, const uint32_t start_row, const uint32_t tile_height, const uint32_t start_col, const uint32_t tile_width);
        void populate_spa(Weight** spa, const Weight* bias, const uint32_t col,  uint64_t& index, const int32_t tid);
        void walk_dxm(const bool one_rank, const int32_t leader_tid, const int32_t tid);
        void walk_dxd(const bool one_rank, const int32_t leader_tid, const int32_t tid);
        void reallocate(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t leader_tid, const int32_t tid);
        void adjust(const int32_t tid);
        void adjust(const int32_t leader_tid, const int32_t tid);
        void adjust(const std::deque<int32_t> my_threads, const int32_t leader_tid, const int32_t tid);
        void repopulate(const std::shared_ptr<struct CSC<Weight>> other, const uint32_t start_col, const uint32_t end_col, const uint32_t dis_nnz, const int32_t leader_tid, const int32_t tid);
        void repopulate(const std::shared_ptr<struct CSC<Weight>> other, const std::deque<int32_t> my_threads, const int32_t leader_tid, const int32_t tid);
        void split_and_overwrite(std::vector<std::shared_ptr<struct CSC<Weight>>>& CSCs, const uint32_t nparts_local, const uint32_t nparts_remote);
        
        void Isend(std::vector<MPI_Request>& requests, const int32_t destination_rank, const MPI_Comm MPI_COMM);
        void Irecv(std::vector<MPI_Request>& requests, const int32_t source_rank,      const MPI_Comm MPI_COMM);
        
        uint64_t nnz   = 0;
        uint64_t nnz_i = 0;
        uint32_t nrows = 0;
        uint32_t ncols = 0;
        
        std::shared_ptr<struct Data_Block<uint32_t>> IA_blk;
        std::shared_ptr<struct Data_Block<uint32_t>> JA_blk;
        std::shared_ptr<struct Data_Block<Weight>>   A_blk;
};

template<typename Weight>
CSC<Weight>::CSC(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t socket_id) {
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
    CSC::nnz = nnz_;
    CSC::nnz_i = nnz_;
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
    const Weight YMIN = 0; 
    const Weight YMAX = 32;
    
    for(uint32_t i = 0; i < CSC::nrows; i++) {
        if(s[i]) {
            s[i] += b[c];
            s[i] = (s[i] < YMIN) ? YMIN : (s[i] > YMAX) ? YMAX : s[i];
            if(s[i]) {
                IA[k] = i;
                A[k] = s[i];
                k++;
                s[i] = 0;
            }
        }
    }
    JA[c] = k;
    
    
    /*
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
    */
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
                std::cout << "j=" << j << "," << j << ": " << JA[j] << "--" << JA[j + 1] << ": " <<  JA[j + 1] - JA[j] << std::endl;
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
    //Env::threads[tid].dis_nnz = (tid == 0) ? 0 : Env::threads[tid].off_nnz - Env::threads[tid-1].idx_nnz;
    //pthread_barrier_wait(&Env::thread_barrier);
    if((leader_tid == -1) or (tid == leader_tid)) {
        CSC::nnz_i = 0;
        /*
        for(auto it = Env::threads.begin(); it != Env::threads.end(); it++) {
            CSC::nnz_i += ((*it).idx_nnz - (*it).off_nnz);
        } 
        */        
        for(uint32_t i = 0; i < Env::threads.size(); i++) {    
            CSC::nnz_i += (Env::threads[i].idx_nnz - Env::threads[i].off_nnz);
            Env::nnzs[i].push_back(Env::threads[i].idx_nnz - Env::threads[i].off_nnz);
            //printf("%lu ", (Env::threads[i].idx_nnz - Env::threads[i].off_nnz));
        }
        //printf("\n");
        
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
            Env::nnzs[t].push_back(Env::threads[t].idx_nnz - Env::threads[t].off_nnz);
        }
    }
    pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
}

template<typename Weight>
void CSC<Weight>::repopulate(const std::shared_ptr<struct CSC<Weight>> other, const uint32_t start_col, const uint32_t end_col, const uint32_t dis_nnz, const int32_t leader_tid, const int32_t tid) {
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
    
    if(tid == leader_tid) {
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

    for(int32_t i = 0; i < tid; i++) {
        JA[start_col+1] += (Env::threads[i].idx_nnz - Env::threads[i].off_nnz);
    }

    for(uint32_t j = start_col; j < end_col; j++) {
        JA[j+1] = (j == start_col) ? JA[j+1] : JA[j];
        uint32_t& k = JA[j+1];
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
void CSC<Weight>::repopulate(const std::shared_ptr<struct CSC<Weight>> other, const std::deque<int32_t> my_threads, const int32_t leader_tid,  const int32_t tid) {
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
    

    
    if(tid == leader_tid) {
        CSC::nnz = o_nnz_i;
        CSC::nnz_i = o_nnz_i;
        CSC::JA_blk->clear();
        CSC::IA_blk->reallocate(CSC::nnz_i);
        CSC::IA_blk->clear();
        CSC::A_blk->reallocate(CSC::nnz_i);
        CSC::A_blk->clear();
    }    
    
    
    pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
    

    
    uint32_t* JA = CSC::JA_blk->ptr;
    uint32_t* IA = CSC::IA_blk->ptr;
    Weight*    A = CSC::A_blk->ptr;
    
    /*
    if(tid == leader_tid) {
        JA[0] = 0;
        JA[1] = 0;
        for(uint32_t i = 0; i < my_threads.size(); i++) {
            int32_t t = my_threads[i];
            uint32_t start_col = Env::threads[t].start_col;
            //uint32_t start_col = Env::follower_threads_info[leader_tid][t].start_col;
            for(uint32_t j = 0; j < i; j++) {
                int32_t tt = my_threads[j];
                //JA[start_col+1] += (Env::index_nnz[tt] - Env::offset_nnz[tt]);
                JA[start_col+1] += (Env::threads[tt].idx_nnz - Env::threads[tt].off_nnz);
            }
        }
    }
    pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
    */

    /* It's ugly but I have to :-/ */
    for(uint32_t j = 0; j < Env::threads[tid].index; j++) {
        int32_t tt = Env::my_threads[leader_tid][j];
        JA[Env::threads[tid].start_col+1] += (Env::threads[tt].idx_nnz - Env::threads[tt].off_nnz);
    }
 
    
        
        
        
    
    //else {
      //  pthread_mutex_lock(&Env::thread_mutexes[leader_tid]);
        //pthread_cond_wait(&Env::thread_conds[leader_tid], &Env::thread_mutexes[leader_tid]);  
        //pthread_mutex_unlock(&Env::thread_mutexes[leader_tid]);
    //}
    //pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
    
    const uint32_t start_col = Env::threads[tid].start_col;// Env::follower_threads_info[leader_tid][tid].start_col;
    const uint32_t end_col   = Env::threads[tid].end_col; //Env::follower_threads_info[leader_tid][tid].end_col;


    for(uint32_t i = o_JA[start_col] + Env::threads[tid].dis_nnz; i < o_JA[start_col + 1]; i++) {
        IA[JA[start_col+1]] = o_IA[i];
        A[JA[start_col+1]]  = o_A[i];
        JA[start_col+1]++;
    }
    
    for(uint32_t j = start_col; j < end_col; j++) {
        JA[j+1] = JA[j];
        for(uint32_t i = o_JA[j]; i < o_JA[j + 1]; i++) {
            IA[JA[j+1]] = o_IA[i];
            A[JA[j+1]]  = o_A[i];
            JA[j+1]++;
        }
    }

    /*
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
    */
    pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
}

template<typename Weight>
void CSC<Weight>::split_and_overwrite(std::vector<std::shared_ptr<struct CSC<Weight>>>& CSCs, const uint32_t nparts_local, const uint32_t nparts_remote) {

    uint64_t&  nnz   = CSC::nnz;
    uint64_t&  nnz_i = CSC::nnz_i;
    uint32_t&  nrows = CSC::nrows;
    uint32_t   ncols = CSC::ncols;
    uint32_t*     IA = CSC::IA_blk->ptr;
    uint32_t*     JA = CSC::JA_blk->ptr;
    Weight*        A = CSC::A_blk->ptr;    
    
    std::vector<uint32_t> rows(nrows);
    for(uint32_t j = 0; j < ncols; j++) {
        for(uint32_t i = JA[j]; i < JA[j+1]; i++) {
            rows[IA[i]]++;
        }
    }
    uint32_t nparts = 1 + nparts_remote;
    uint64_t balanced_nnz = nnz_i/(nparts_local + nparts_remote);
    std::vector<uint64_t> bounds_nnz(nparts, balanced_nnz);
    bounds_nnz[0] = balanced_nnz * nparts_local;
    
    //printf("nnz=%lu nnz_i=%lu\n", nnz, nnz_i);
    //for(auto b: bounds_nnz) {printf("%lu ", b);} printf("\n");
    
    
    std::vector<uint64_t> new_nnzs(nparts);
    std::vector<uint32_t> new_rows(nparts);
    uint32_t i = 0;
    for(uint32_t k = 0; k < nparts; k++) {
        while((new_nnzs[k] < bounds_nnz[k]) and (i < nrows)) {
            new_nnzs[k] += rows[i];
            i++;
        }
        new_rows[k] = i;
    }
    
    /*
    uint64_t s = 0;
    uint32_t t = 0;
    for(uint32_t k = 0; k < bounds_nnz.size(); k++) {
        s += new_nnzs[k];
        t += new_rows[k];
        printf("%lu %d\n", new_nnzs[k], new_rows[k]);
    }
    printf("%lu %lu %d\n", s, nnz, t);
    */
    
    std::vector<uint32_t> new_heights(nparts);
    std::vector<uint32_t> new_offsets(nparts);
    new_heights[0] = new_rows[0];
    for(uint32_t k = 1; k < nparts; k++) {
        new_heights[k] = new_rows[k] - new_rows[k-1];
        new_offsets[k]= new_rows[k-1];
    }
    uint32_t new_width = ncols;
    /*
    std::vector<uint32_t> new_offsets(nparts);
    new_offsets[0]
    for(uint32_t k = 1; k < new_offsets.size(); k++) {
        
        new_heights[k] = new_rows[k] - new_rows[k-1];
    }
    */
    
    /*
    uint32_t s = 0;
    for(uint32_t k = 0; k < new_heights.size(); k++) {
        s += new_heights[k];
        printf("%d %d\n", k, new_heights[k]);
    }
    printf("%d\n", s);
    */
    //std::vector<std::shared_ptr<struct CSC<Weight>>> new_CSC = std::make_shared<struct CSC<Weight>>(new_nnzs[0], new_heights[0], new_width);
    for(uint32_t k = 0; k < nparts; k++) {
        //printf("%d %lu %d %d %d\n", k, new_nnzs[k], new_heights[k], new_rows[k], new_offsets[k]);
        CSCs.push_back(std::make_shared<struct CSC<Weight>>(new_nnzs[k], new_heights[k], new_width));
    }
//std::exit(0);
    for(uint32_t k = 0; k < nparts; k++) {
        uint32_t* JA_ = CSCs[k]->JA_blk->ptr;
        JA_[0] = 0;
    }

    for(uint32_t j = 0; j < ncols; j++) {
        for(uint32_t k = 0; k < nparts; k++) {
            uint32_t* JA_ = CSCs[k]->JA_blk->ptr;
            JA_[j+1] = JA_[j];
        }
        for(uint32_t i = JA[j]; i < JA[j+1]; i++) {
            uint32_t k = 0;
            for(k = 0; k < nparts; k++) {
                if(IA[i] < new_rows[k]) {
                    break;
                }
            }
            
            uint32_t* IA_ = CSCs[k]->IA_blk->ptr;
            uint32_t* JA_ = CSCs[k]->JA_blk->ptr;
            double*  A_ = CSCs[k]->A_blk->ptr;
            uint32_t& k_ = JA_[j+1];
            IA_[k_] = IA[i] - new_offsets[k];
            A_[k_] = A[i];
            k_++;
        }
    }
    
    nnz   = CSCs[0]->nnz;
    nnz_i = CSCs[0]->nnz_i;
    nrows = CSCs[0]->nrows;
    ncols = CSCs[0]->ncols;
    CSC::JA_blk->clear();
    CSC::JA_blk->copy(CSCs[0]->JA_blk->ptr);
    CSC::IA_blk->reallocate(CSC::nnz_i);
    CSC::IA_blk->clear();
    CSC::IA_blk->copy(CSCs[0]->IA_blk->ptr);
    CSC::A_blk->reallocate(CSC::nnz_i);
    CSC::A_blk->clear(); 
    CSC::A_blk->copy(CSCs[0]->A_blk->ptr);
    
    
    
    
    
    
/*
    double checksum   = 0;
    uint64_t checkcount = 0;  

    for(uint32_t j = 0; j < ncols; j++) {
        for(uint32_t i = JA[j]; i < JA[j+1]; i++) {
            checkcount++;
            checksum += A[i];
        }
    }
    printf(">> %f %lu\n", checksum, checkcount);
    
    //uint32_t*     IA = CSC::IA_blk->ptr;
    //uint32_t*     JA = CSC::JA_blk->ptr;
    //Weight*        A = CSC::A_blk->ptr;
    
 
    
    
  
     checksum   = 0;
     checkcount = 0;  

for(auto csc: CSCs) {
    uint64_t checkcount1 = 0;
    uint32_t ncols1 = csc->ncols;
    uint32_t*   IA1 = csc->IA_blk->ptr;
    uint32_t*   JA1 = csc->JA_blk->ptr;
    double*      A1 = csc->A_blk->ptr;
    for(uint32_t j = 0; j < ncols1; j++) {
        for(uint32_t i = JA1[j]; i < JA1[j+1]; i++) {
            checkcount1++;
            checksum += A[i];
        }
    }
    printf("%lu\n", checkcount1);
    checkcount += checkcount1;
}    

printf("%f %lu\nn", checksum, checkcount);
       */
        
        /*
        //JA1[j+1] = 
        uint32_t& k1 = JA1[j+1];
        k1 = JA1[j];

        uint32_t& k2 = JA2[j+1];
        k2 = JA2[j];
        //uint32_t j1_old = JA1[j+1];
        //uint32_t j2_old = JA2[j+1];
        //if(JA[j+1] - JA[j]) {
        for(uint32_t i = JA[j]; i < JA[j+1]; i++) {
            if(IA[i] < new_height1) {
                //JA1[j+1]++
                IA1[k1] = IA[i];
                A1[k1] = A[i];
                k1++;
            }
            else {
                //JA1[j+1]++;
                IA2[k2] = IA[i] - new_height1;
                A2[k2] = A[i];
                k2++;
            }
        }
    }
*/
    
    
    
    /*
    const uint64_t  nnz   = CSC->nnz;
    const uint32_t  nrows = CSC->nrows;
    const uint32_t  ncols = CSC->ncols;
    const uint32_t*    IA = CSC->IA_blk->ptr;
    const uint32_t*    JA = CSC->JA_blk->ptr;
    const Weight*       A = CSC->A_blk->ptr;
    
    uint64_t balanced_nnz = CSC->nnz/2;
    
    printf("%d %lu %d %lu\n", Env::rank, CSC->nnz, CSC->nrows, balanced_nnz);
    
    

    
    
    uint32_t i = 0;
    uint64_t new_nnz1 = 0;
    while(new_nnz1 < balanced_nnz) {
        new_nnz1 += rows[i];
        i++;
    }
    uint32_t nrows1 = i;
    
    //uint64_t s = 0;
    //for(i = 0; i < row; i++) {
      //  s += rows[i];
    //}
    //printf("%d <%lu %lu %lu>\n", row, max_nnz, balanced_nnz, s);
    
    
    uint32_t new_height1 = nrows1;
    uint32_t new_height2 = tile.height - nrows1;
    uint32_t new_width = tile.width;
    uint64_t new_nnz2 = nnz - new_nnz1; 
    
    printf("[%d %d %d] [%lu %lu %lu]\n", new_height1, new_height2, new_height1 + new_height2, new_nnz1, new_nnz2, new_nnz1 + new_nnz2);
    std::shared_ptr<struct CSC<Weight>> CSC1 = std::make_shared<struct CSC<Weight>>(new_nnz1, new_height1, new_width);
    std::shared_ptr<struct CSC<Weight>> CSC2 = std::make_unique<struct CSC<Weight>>(new_nnz2, new_height2, new_width);
    
    uint32_t* IA1 = CSC1->IA_blk->ptr;
    uint32_t* JA1 = CSC1->JA_blk->ptr;
    Weight*    A1 = CSC1->A_blk->ptr;
    
    uint32_t* IA2 = CSC2->IA_blk->ptr;
    uint32_t* JA2 = CSC2->JA_blk->ptr;
    Weight*    A2 = CSC2->A_blk->ptr;
    
    
    //uint32_t i1 = 0;
    //uint32_t i2 = 0;
    //uint32_t j1 = 1;
    //uint32_t j2 = 1;
    JA1[0] = 0;
    JA2[0] = 0;
    for(uint32_t j = 0; j < ncols; j++) {
        //JA1[j+1] = 
        uint32_t& k1 = JA1[j+1];
        k1 = JA1[j];

        uint32_t& k2 = JA2[j+1];
        k2 = JA2[j];
        //uint32_t j1_old = JA1[j+1];
        //uint32_t j2_old = JA2[j+1];
        //if(JA[j+1] - JA[j]) {
        for(uint32_t i = JA[j]; i < JA[j+1]; i++) {
            if(IA[i] < new_height1) {
                //JA1[j+1]++
                IA1[k1] = IA[i];
                A1[k1] = A[i];
                k1++;
            }
            else {
                //JA1[j+1]++;
                IA2[k2] = IA[i] - new_height1;
                A2[k2] = A[i];
                k2++;
            }
        }
    }
    
    double checksum1   = 0;
    uint64_t checkcount1 = 0;    
    double checksum2   = 0;
    uint64_t checkcount2 = 0;    
    for(uint32_t j = 0; j < ncols; j++) {
        for(uint32_t i = JA1[j]; i < JA1[j+1]; i++) {
            checksum1 += A1[i];
            checkcount1++;
        }
    }
    
    for(uint32_t j = 0; j < ncols; j++) {
        for(uint32_t i = JA2[j]; i < JA2[j+1]; i++) {
            checksum2 += A2[i];
            checkcount2++;
        }
    }
    
    printf("[%f %lu %lu] [%f %lu %lu]\n", checksum1, checkcount1, new_nnz1, checksum2, checkcount2, new_nnz2);
    
    */
    
}


template<typename Weight>
void CSC<Weight>::Isend(std::vector<MPI_Request>& requests, const int32_t destination_rank, const MPI_Comm MPI_COMM) {
    MPI_Request request;
    MPI_Datatype WEIGHT_TYPE = MPI_Types::get_mpi_data_type<Weight>();
    
    uint32_t* JA = CSC::JA_blk->ptr;
    uint32_t* IA = CSC::IA_blk->ptr;
    Weight*    A = CSC::A_blk->ptr;
    
    MPI_Isend(JA, CSC::JA_blk->nitems, MPI_UNSIGNED, destination_rank, (destination_rank*3)+0, MPI_COMM, &request);
    requests.push_back(request);
    MPI_Isend(IA, CSC::IA_blk->nitems, MPI_UNSIGNED, destination_rank, (destination_rank*3)+1, MPI_COMM, &request);
    requests.push_back(request);
    MPI_Isend(A, CSC::A_blk->nitems,    WEIGHT_TYPE, destination_rank, (destination_rank*3)+2, MPI_COMM, &request);
    requests.push_back(request);
}

template<typename Weight>
void CSC<Weight>::Irecv(std::vector<MPI_Request>& requests, const int32_t source_rank, const MPI_Comm MPI_COMM) {
    MPI_Request request;
    MPI_Datatype WEIGHT_TYPE = MPI_Types::get_mpi_data_type<Weight>();
    
    uint32_t* JA = CSC::JA_blk->ptr;
    uint32_t* IA = CSC::IA_blk->ptr;
    Weight*    A = CSC::A_blk->ptr;
    
    MPI_Irecv(JA, CSC::JA_blk->nitems, MPI_UNSIGNED, source_rank, (Env::rank*3)+0, MPI_COMM, &request);
    requests.push_back(request);
    MPI_Irecv(IA, CSC::IA_blk->nitems, MPI_UNSIGNED, source_rank, (Env::rank*3)+1, MPI_COMM, &request);
    requests.push_back(request);
    MPI_Irecv(A, CSC::A_blk->nitems,    WEIGHT_TYPE, source_rank, (Env::rank*3)+2, MPI_COMM, &request);
    requests.push_back(request);
}

#endif
