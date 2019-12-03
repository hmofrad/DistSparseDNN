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
#include "bitmap.hpp"

enum COMPRESSED_FORMAT {_CSR_, _DCSR_, _TCSR_, _CSC_, _DCSC_, _TCSC_};
const char* COMPRESSED_FORMATS[] = {"_CSR_", "_DCSR_", "_TCSR_", "_CSC_", "_DCSC_", "_TCSC_"};

template<typename Weight>
struct Compressed_Format {
    public:
        Compressed_Format() {};
        virtual ~Compressed_Format() {};
        virtual void populate(const std::vector<struct Triple<Weight>> triples, const uint32_t tile_height, const uint32_t tile_width) {};
        virtual void refine_both(const uint32_t nrows) {};
        virtual void refine_cols() {};
        virtual void refine_rows(const uint32_t nrows) {};
        //virtual void populate_spa(struct Bitmap spa_bitmap, std::vector<Weight>& spa, const std::vector<Weight> bias, const uint32_t col, const int32_t tid) {};
        virtual void populate_spa(Weight** spa, const Weight* bias, const uint32_t col,  uint64_t& index, const int32_t tid) {};
        //virtual void populate_spa(std::vector<Weight>& spa, const std::vector<Weight> bias, const uint32_t col, const int32_t tid) {};
        virtual void adjust(const int32_t tid) {};
        virtual void reallocate(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t tid) {};
        virtual void reallocate(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_) {};
        virtual void repopulate(const std::shared_ptr<struct Compressed_Format<Weight>> other, const int32_t tid) {};
        virtual void walk() {};
        virtual void walk(const int32_t tid) {};
        
        COMPRESSED_FORMAT compression_type;
        
        uint64_t nnz;
        uint32_t nrows;
        uint32_t ncols;
        
        bool one_rank;
        
        std::shared_ptr<struct Data_Block<uint32_t>> IA_blk;
        std::shared_ptr<struct Data_Block<uint32_t>> JA_blk;
        std::shared_ptr<struct Data_Block<Weight>>    A_blk;
};

template<typename Weight>
struct CSC: public Compressed_Format<Weight> {
    public:
        CSC(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const bool one_rank_);
        ~CSC(){};
        
        void populate(const std::vector<struct Triple<Weight>> triples, const uint32_t tile_height, const uint32_t tile_width);
        void refine_both(const uint32_t nrows);
        void refine_cols();
        void refine_rows(const uint32_t nrows);
        //void populate_spa(std::vector<Weight>& spa, const std::vector<Weight> bias, const uint32_t col, const int32_t tid);
        void populate_spa(Weight** spa, const Weight* bias, const uint32_t col,  uint64_t& index, const int32_t tid);
        //void populate_spa(struct Bitmap spa_bitmap, std::vector<Weight>& spa, const std::vector<Weight> bias, const uint32_t col, const int32_t tid);
        void walk();
        void walk(const int32_t tid);
        void reallocate(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t tid);
        void reallocate(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_);
        void adjust(const int32_t tid);
        void repopulate(const std::shared_ptr<struct CSC<Weight>> other, const int32_t tid);

        
        uint64_t nnz_i = 0;
};


template<typename Weight>
CSC<Weight>::CSC(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const bool one_rank_) {
    Compressed_Format<Weight>::compression_type = COMPRESSED_FORMAT::_CSC_;
    CSC::nnz = nnz_;
    CSC::nrows = nrows_; 
    CSC::ncols = ncols_;
    CSC::one_rank = one_rank_;
    
    CSC::JA_blk = std::move(std::make_shared<struct Data_Block<uint32_t>>((CSC::ncols + 1), Env::rank_socket_id));
    CSC::IA_blk = std::move(std::make_shared<struct Data_Block<uint32_t>>(CSC::nnz, Env::rank_socket_id));
    CSC::A_blk = std::move(std::make_shared<struct Data_Block<Weight>>(CSC::nnz, Env::rank_socket_id));
}

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

template<typename Weight>
void CSC<Weight>::refine_both(const uint32_t nrows) {    
    // Refine columns
    refine_cols();
    // Refine Rows
    refine_rows(nrows);
}

template<typename Weight>
void CSC<Weight>::refine_cols() { 
    uint32_t* IA = CSC::IA_blk->ptr;
    uint32_t* JA = CSC::JA_blk->ptr;
    Weight*    A = CSC::A_blk->ptr;

    uint32_t c1 = 0;
    uint32_t c2 = 0;
    for(int32_t t = Env::nthreads; t > 0; t--) {
        c1 = ((t - 1) * (CSC::ncols/Env::nthreads)) + 1;
        c2 =   t      * (CSC::ncols/Env::nthreads);
        
        for(uint32_t j = c2; j >= c1; j--) {
            JA[j] = JA[j - t];
        }
    }
}

template<typename Weight>
void CSC<Weight>::refine_rows(const uint32_t nrows) {
    uint32_t* IA = CSC::IA_blk->ptr;
    uint32_t* JA = CSC::JA_blk->ptr;
    Weight*    A = CSC::A_blk->ptr;
    
    for(uint32_t j = 0; j < CSC::ncols; j++) {
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            IA[i] += (IA[i] / ((nrows - Env::nthreads) / Env::nthreads)) + 1;
        }
    }
}

template<typename Weight>
//void CSC<Weight>::populate_spa(struct Bitmap spa_bitmap, std::vector<Weight>& spa, const std::vector<Weight> bias, const uint32_t col, const int32_t tid) {
    
//void CSC<Weight>::populate_spa(std::vector<Weight>& spa, const std::vector<Weight> bias, const uint32_t col, const int32_t tid) {
//void CSC<Weight>::populate_spa(std::shared_ptr<struct Data_Block<Weight>> spa, const std::shared_ptr<struct Data_Block<Weight>> bias, const uint32_t col, const int32_t tid) {
void CSC<Weight>::populate_spa(Weight** spa, const Weight* bias, const uint32_t col, uint64_t& index, const int32_t tid) {
    //uint64_t&  k = Env::index_nnz[tid];
    //uint64_t&  k = CSC::nnz_i;
    uint64_t&  k = index;

    uint32_t   c = col + 1;
    uint32_t* IA = CSC::IA_blk->ptr;
    uint32_t* JA = CSC::JA_blk->ptr;
    Weight*    A = CSC::A_blk->ptr;
    
    Weight* s = *spa;
    const Weight* b = bias;
    
    //Weight*    s   = spa->ptr;
    //Weight*    b   = bias->ptr;
    
    Weight YMIN = 0;
    Weight YMAX = 32;
    /*
    JA[c] = k;
    for(uint32_t i = 0; i < CSC::nrows; i++) {
        if(spa_bitmap.get_bit(i)) {
            spa_bitmap.clear_bit(i);
            spa[i] += bias[c];
            if(spa[i] < YMIN) {
                spa[i] = YMIN;
            }
            else if(spa[i] > YMAX) {
                spa[i] = YMAX;
            }
            if(spa[i]) {
                JA[c]++;
                IA[k] = i;
                A[k] = spa[i];
                k++;
                spa[i] = 0;
            }
        }
    }
    */
    
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
void CSC<Weight>::walk(const int32_t tid) {  
    //std::vector<bool> rows(CSC::nrows);
    //std::vector<bool> cols(CSC::ncols);

    uint32_t* IA = CSC::IA_blk->ptr;
    uint32_t* JA = CSC::JA_blk->ptr;
    Weight*    A = CSC::A_blk->ptr;
    
    uint32_t start_col = Env::start_col[tid];// + tid + 1;
    uint32_t end_col = Env::end_col[tid];
    
    Env::checksum[tid] = 0;
    Env::checkcount[tid] = 0;    
    Env::checknnz[tid] = CSC::nnz_i;
    //pthread_barrier_wait(&Env::thread_barrier);
    //#pragma omp barrier
    for(uint32_t j = 0; j < CSC::ncols; j++) {  
        // std::cout << "j=" << j << "," << j << ": " << JA[j] << "--" << JA[j + 1] << ": " <<  JA[j + 1] - JA[j] << std::endl;
        //if(JA[j+1] - JA[j])
        //    Env::cols[tid][j] = true;
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            (void) IA[i];
            (void) A[i];
            Env::checksum[tid] += A[i];
            Env::checkcount[tid]++;
            //std::cout << "    i=" << i << ",i=" << IA[i] <<  ",value=" << A[i] << std::endl;
            //Env::rows[tid][IA[i]] = true;
            
            //cols[j] = 1;
        }
    }   
    
    Env::barrier();
    pthread_barrier_wait(&Env::thread_barrier);
    //#pragma omp barrier  
    if(!tid) {
        double     sum_threads = std::accumulate(Env::checksum.begin(),   Env::checksum.end(),   0);
        uint64_t count_threads = std::accumulate(Env::checkcount.begin(), Env::checkcount.end(), 0);
        uint64_t nnz_threads = std::accumulate(Env::checknnz.begin(), Env::checknnz.end(), 0);
        if(CSC::one_rank) {
            Logging::print(Logging::LOG_LEVEL::INFO, "Iteration=%d, Total checksum=%f, Total count=%d\n", Env::iteration, sum_threads, count_threads);
        }
        else {
            double sum_ranks = 0;
            uint64_t count_ranks = 0;
            
            MPI_Allreduce(&sum_threads, &sum_ranks, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&count_threads, &count_ranks, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
            
            if(count_threads != nnz_threads) {
                Logging::print(Logging::LOG_LEVEL::WARN, "Compression checksum warning!!\n");
            }
            
            Logging::print(Logging::LOG_LEVEL::INFO, "Iteration=%d, Total checksum=%f, Total count=%d\n", Env::iteration, sum_ranks, count_ranks);
        }

        
        
        /*
        uint32_t nrows_ = 0;
        std::vector<bool> rows_(CSC::nrows);
        for(uint32_t i = 0; i < CSC::nrows; i++) {
            for(int32_t j = 0; j < Env::nthreads; j++) {
                if(Env::rows[j][i]) {
                    rows_[i] = true;
                }
            }
            if(rows_[i]) {
                nrows_++;
                rows_[i] = false;
            }
        }
        
        uint32_t ncols_ = 0;
        std::vector<bool> cols_(CSC::ncols);
        for(uint32_t i = 0; i < CSC::ncols; i++) {
            for(int32_t j = 0; j < Env::nthreads; j++) {
                if(Env::cols[j][i]) {
                    cols_[i] = true;
                }
            }
            if(cols_[i]) {
                ncols_++;
                cols_[i] = false;
            }
        }
        
        
        printf("%d: [%d %d] [%d %d]\n", Env::rank, CSC::nrows, nrows_, CSC::ncols, ncols_);
        */
        
    }
    //pthread_barrier_wait(&Env::thread_barrier);
    //#pragma omp barrier  

    //if(!tid) {
        
    //}
    
}

template<typename Weight>
void CSC<Weight>::walk() {  
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
        }
    }

    Env::barrier();
    if(CSC::one_rank) {
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

template<typename Weight>
void CSC<Weight>::reallocate(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_) {
    double start_time = Env::tic();
    
    if(CSC::ncols != ncols_) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Cannot reallocate.\n");
        std::exit(Env::finalize());     
    }
    
    //if(!tid) {
        CSC::nnz_i = 0;
        CSC::nnz = nnz_;
        CSC::nrows = nrows_; 
        CSC::ncols = ncols_;
        
        CSC::IA_blk->reallocate(CSC::nnz);
        CSC::A_blk->reallocate(CSC::nnz);
            
        CSC::JA_blk->clear();
        CSC::IA_blk->clear();
        CSC::A_blk->clear();
    //}    
    //if(!tid) {
      //  Env::memory_time += Env::toc(start_time);
    //}
    Env::memory_allocation_time[0] += Env::toc(start_time);
    /*
    pthread_barrier_wait(&Env::thread_barrier);
    
    uint32_t share_col = (CSC::ncols) / 3;
    uint32_t start_c = share_col * tid;
    uint32_t end_c = (tid == (Env::nthreads - 1)) ? CSC::ncols : share_col * (tid + 1);     
    uint64_t share_nnz = CSC::nnz/Env::nthreads;
    uint64_t start_n = share_nnz * tid;
    uint64_t end_n = (tid == (Env::nthreads - 1)) ? CSC::nnz : share_nnz * (tid + 1);
    
    CSC::IA_blk->clear(start_n, end_n);
    CSC::A_blk->clear(start_n, end_n);
    
    pthread_barrier_wait(&Env::thread_barrier);
    */
}

template<typename Weight>
void CSC<Weight>::reallocate(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t tid) {
    double start_time = Env::tic();
    
    if(CSC::ncols != ncols_) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Cannot reallocate.\n");
        std::exit(Env::finalize());     
    }
    
    //if(!tid) {
        CSC::nnz_i = 0;
        CSC::nnz = nnz_;
        CSC::nrows = nrows_; 
        CSC::ncols = ncols_;
        
        CSC::IA_blk->reallocate(CSC::nnz);
        CSC::A_blk->reallocate(CSC::nnz);
            
        CSC::JA_blk->clear();
        CSC::IA_blk->clear();
        CSC::A_blk->clear();
        
    //if(!tid) {
      //  Env::memory_time += Env::toc(start_time);
    //}
    Env::memory_allocation_time[tid] += Env::toc(start_time);
    /*
    pthread_barrier_wait(&Env::thread_barrier);
    
    uint32_t share_col = (CSC::ncols) / 3;
    uint32_t start_c = share_col * tid;
    uint32_t end_c = (tid == (Env::nthreads - 1)) ? CSC::ncols : share_col * (tid + 1);     
    uint64_t share_nnz = CSC::nnz/Env::nthreads;
    uint64_t start_n = share_nnz * tid;
    uint64_t end_n = (tid == (Env::nthreads - 1)) ? CSC::nnz : share_nnz * (tid + 1);
    
    CSC::IA_blk->clear(start_n, end_n);
    CSC::A_blk->clear(start_n, end_n);
    
    pthread_barrier_wait(&Env::thread_barrier);
    */
}

template<typename Weight>
void CSC<Weight>::adjust(const int32_t tid){
    
    
    uint32_t displacement = (tid == 0) ? 0 : Env::offset_nnz[tid] - Env::index_nnz[tid-1];
    Env::displacement_nnz[tid] = displacement;
    ///printf(">>> %d %d | %d %d | off=%lu idx=%lu disp=%d\n",Env::rank, tid, Env::start_col[tid], Env::end_col[tid], Env::offset_nnz[tid], Env::index_nnz[tid], Env::displacement_nnz[tid]);
    
    //uint32_t* JA = CSC::JA_blk->ptr;
    //JA[Env::start_col[tid]] = Env::offset_nnz[tid];
    pthread_barrier_wait(&Env::thread_barrier);
    //#pragma omp barrier
    if(!tid) {
        CSC::nnz_i = 0;
        for(int32_t i = 0; i < Env::nthreads; i++) {    
            CSC::nnz_i += (Env::index_nnz[i] - Env::offset_nnz[i]);
        }
        //printf(">>>. %d %lu\n", Env::rank, CSC::nnz_i);
    }
    pthread_barrier_wait(&Env::thread_barrier);
    
    //#pragma omp barrier  
    /*
    if(!tid) {
        Env::nnz_ranks.push_back(CSC::nnz);
        Env::nnz_i_ranks.push_back(CSC::nnz_i);
        
        uint64_t sum = 0.0, mean = 0.0, std_dev = 0.0, min = 0.0, max = 0.0;
        
        Env::stats(Env::count_nnz, sum, mean, std_dev, min, max);
        Env::nnz_mean_thread_ranks.push_back(mean);
        Env::nnz_std_dev_thread_ranks.push_back(std_dev);
        Env::nnz_min_thread_ranks.push_back(min);
        Env::nnz_max_thread_ranks.push_back(max);
        
        for(int i = 0; i < Env::nthreads; i++) {
            Env::count_nnz_i[i] = Env::index_nnz[i] - Env::offset_nnz[i];
        }
        Env::stats(Env::count_nnz_i, sum, mean, std_dev, min, max);
        Env::nnz_i_mean_thread_ranks.push_back(mean);
        Env::nnz_i_std_dev_thread_ranks.push_back(std_dev);
        Env::nnz_i_min_thread_ranks.push_back(min);
        Env::nnz_i_max_thread_ranks.push_back(max);
    }
    */
}

template<typename Weight>
void CSC<Weight>::repopulate(const std::shared_ptr<struct CSC<Weight>> other, const int32_t tid) {
    //printf("%d/%d: repopulate\n", Env::rank, tid);
    
    

    uint32_t  o_ncols = other->ncols;
    uint32_t  o_nrows = other->nrows;
    uint32_t  o_nnz   = other->nnz;
    uint32_t  o_nnz_i = other->nnz_i;
    uint32_t* o_JA    = other->JA_blk->ptr;
    uint32_t* o_IA    = other->IA_blk->ptr;
    Weight*   o_A     = other->A_blk->ptr;

    if(CSC::ncols != o_ncols){// or (CSC::nrows != o_nrows)) {
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
    //#pragma omp barrier
    
    //printf("MOVE\n");
    uint32_t* JA = CSC::JA_blk->ptr;
    uint32_t* IA = CSC::IA_blk->ptr;
    Weight*    A = CSC::A_blk->ptr;
    
    //printf(">>> %lu %lu\n", CSC::IA_blk->nitems, other->IA_blk->nitems);

    uint32_t start_col = Env::start_col[tid];
    uint32_t end_col = Env::end_col[tid];
    //JA[start_col] = Env::offset_nnz[tid] - Env::displacement_nnz[tid];
    
    for(int32_t i = 0; i < tid; i++) {
        JA[start_col+1] += (Env::index_nnz[i] - Env::offset_nnz[i]);
    }
    
    //printf("%d %d %lu %lu %d\n", tid, JA[start_col], Env::index_nnz[0], Env::offset_nnz[0], (Env::index_nnz[i-1] - Env::offset_nnz[i-1]));
//#pragma omp barrier
    //std::exit(0);

    
  //  end_col -= (tid == (Env::nthreads - 1)) ? 0 : 1;
    //printf("%d %d %d\n", tid, JA[start_col], JA[start_col+1]);
    //printf("TID=%d\n", tid);
    //pthread_barrier_wait(&Env::thread_barrier);
    
    //if(tid){
    //for(uint32_t j = 0; j < o_ncols; j++) {
        //printf("%d %d %d %d\n", j, o_JA[j], o_JA[j+1], o_JA[j+1]-o_JA[j]);
    //}
        
    //}
    
    
    //if(tid){
    for(uint32_t j = start_col; j < end_col; j++) {
        JA[j+1] = (j == start_col) ? JA[j+1] : JA[j];
        
        uint32_t& k = JA[j+1];
        uint32_t m = (j == start_col) ? Env::displacement_nnz[tid] : 0;
        
        //printf("1. j=%d tid=%d JAj=%d JAj+1=%d m=%d i=%d\n", j, tid, JA[j], JA[j+1], m, o_JA[j] + m);
        //if(tid)
        //printf("%d %d %d %d %d\n", tid, j, k, JA[j+1], m);
        for(uint32_t i = o_JA[j] + m; i < o_JA[j + 1]; i++) {
            //if(tid)
            //printf("%d %d\n",k, i);
            //JA[j+1]++;
            IA[k] = o_IA[i];
            A[k]  = o_A[i];
            k++;
        }
        //if(tid)
      //  printf("2. j=%d tid=%d JAj=%d JAj+1=%d k=%d\n", j, tid, JA[j], JA[j+1], k);
    //    if(j == start_col + 10) break;
    }
    //}
    
    
    pthread_barrier_wait(&Env::thread_barrier);
    //printf("????????\n");
    //#pragma omp barrier
//    end_col += (tid == (Env::nthreads - 1)) ? 0 : 1;
//    printf("%d %d %d - %d %d %d %d\n", tid, start_col, end_col, JA[start_col], JA[start_col+1], JA[end_col-1], JA[end_col]);
  //  printf("%lu %lu\n", CSC::nnz, CSC::nnz_i);
    //#pragma omp barrier
/*    
    uint32_t start_col = Env::start_col[tid];
    uint32_t end_col = Env::end_col[tid];
    
    if(tid == 0) {
        JA[0] = 0;
        for(uint32_t j = start_col+1; j < end_col; j++) {
            JA[j] += JA[j-1];
        }
    }
    else {
        JA[start_col] = 0;
        for(int32_t i = 0; i < tid; i++) {
            JA[start_col] += (Env::offset_nnz[i] - Env::start_nnz[i]);
        }
        
        for(uint32_t j = start_col+1; j < end_col; j++) {
            JA[j] += JA[j-1];
        }
    }
    
    if((tid == Env::nthreads - 1)) {
        JA[end_col] += JA[end_col-1];
    }
    
    
    if(tid == 0) {
        idx = 0;
        for(uint32_t i = 0; i < Env::nthreads; i++) {    
            idx += (Env::offset_nnz[i] - Env::start_nnz[i]);
        }
    }
    
    uint32_t o_ncols = other_csc->numcols();
    uint32_t o_nnz = other_csc->numnonzeros();
    uint32_t o_idx = other_csc->idx;
    uint32_t *o_JA = other_csc->JA;
    uint32_t *o_IA = other_csc->IA;
    Weight   *o_A  = other_csc->A;
    
    if(ncols != o_ncols) {
        fprintf(stderr, "Error: Cannot repopulate CSC\n");
        exit(1);
    }
    
    if(!tid) {
        nnz = o_idx;
        nnzmax = o_idx;
        IA_blk->reallocate(&IA, nnz, (nnz * sizeof(uint32_t)));
        A_blk->reallocate(&A, nnz, (nnz * sizeof(Weight)));            
        clear();
    }
    #pragma omp barrier
    uint32_t start_col = Env::start_col[tid];
    uint32_t end_col = Env::end_col[tid];
    uint64_t offset = 0;
    uint64_t idx = 0;
    JA[start_col] = 0;
    if(tid) {
        JA[start_col] = o_JA[start_col];
        for(int32_t i = 0; i < tid; i++) {
            //JA[start_col] += (Env::offset_nnz[i] - Env::start_nnz[i]);
            offset += (Env::end_nnz[i] - Env::offset_nnz[i]);
        }
        idx = JA[start_col];
    }
    for(uint32_t j = start_col; j < end_col; j++) {
        JA[j+1] = JA[j];
        for(uint32_t i = o_JA[j] + offset; i < o_JA[j + 1] + offset; i++) {
            JA[j+1]++;
            IA[idx] = o_IA[i];
            A[idx]  = o_A[i];
            idx++;
        }
    }
*/    
    
    
    
    
}



#endif
