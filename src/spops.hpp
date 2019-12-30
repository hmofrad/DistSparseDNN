/*
 * spops.hpp: Sparse Matrix operations implementation
 * Sparse Matrix - Sparse Matrix (SpMM)
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com
 */
 
#ifndef SPOPS_H
#define SPOPS_H

#include "env.hpp"
#include "spmat.hpp"

template<typename Weight>
inline std::tuple<uint64_t, uint32_t, uint32_t> spmm_sym(std::shared_ptr<struct CSC<Weight>> A_CSC,
                                                         std::shared_ptr<struct CSC<Weight>> B_CSC,
                                                         std::shared_ptr<struct Data_Block<Weight>> s,
                                                         const uint32_t start_col,
                                                         const uint32_t end_col,
                                                         const int32_t tid) {

    uint64_t nnzmax = 0;
    uint32_t nrows = 0;
    uint32_t ncols = 0; 

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
        
    Weight*          s_A   = s->ptr;

    if(A_ncols != B_nrows) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "SpMM dimensions do not agree A[%d %d] B[%d %d]\n", A_nrows, A_ncols, B_nrows, B_ncols);
        std::exit(Env::finalize()); 
    }
    nrows = A_nrows;
    ncols = B_ncols;

    for(uint32_t j = start_col; j < end_col; j++) {
        for(uint32_t k = B_JA[j]; k < B_JA[j+1]; k++) {
            uint32_t l = B_IA[k];
            for(uint32_t n = A_JA[l]; n < A_JA[l+1]; n++) {
                s_A[A_IA[n]] = 1;
            }
        }
        for(uint32_t i = 0; i < A_nrows; i++) {
            if(s_A[i]){
                nnzmax++;
                s_A[i] = 0;
            }
        }
    }

    return std::make_tuple(nnzmax, nrows, ncols);
}

template<typename Weight>
inline void spmm(std::shared_ptr<struct CSC<Weight>> A_CSC,
                 std::shared_ptr<struct CSC<Weight>> B_CSC,
                 std::shared_ptr<struct CSC<Weight>> C_CSC,
                 std::shared_ptr<struct Data_Block<Weight>> s,
                 const std::shared_ptr<struct Data_Block<Weight>> b,
                 const uint32_t start_col,
                 const uint32_t end_col,
                 const uint32_t off_col,
                 uint64_t& idx_nnz,
                 const int32_t tid) {

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
        
    const uint64_t C_nnz   = C_CSC->nnz;
    const uint32_t C_nrows = C_CSC->nrows;
    const uint32_t C_ncols = C_CSC->ncols;
    const uint32_t* C_IA   = C_CSC->IA_blk->ptr;
    uint32_t*       C_JA   = C_CSC->JA_blk->ptr;
    const Weight*    C_A   = C_CSC->A_blk->ptr;
        
    Weight*          s_A   = s->ptr;
    const Weight*    b_A   = b->ptr;
        
    if(A_ncols != B_nrows) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "SpMM dimensions do not agree C[%d %d] != A[%d %d] B[%d %d]\n", C_nrows, C_ncols, A_nrows, A_ncols, B_nrows, B_ncols);
        std::exit(Env::finalize()); 
    }

    ///uint64_t& index = Env::index_nnz[tid];
    //uint64_t& index = Env::threads[tid].idx_nnz;
        
    for(uint32_t j = start_col; j < end_col; j++) {
        for(uint32_t k = B_JA[j]; k < B_JA[j+1]; k++) {
            uint32_t l = B_IA[k];
            for(uint32_t n = A_JA[l]; n < A_JA[l+1]; n++) {
                s_A[A_IA[n]] += (B_A[k] * A_A[n]);
            }
        }
        C_CSC->populate_spa(&s_A, b_A, off_col + j, idx_nnz, tid);
    }
}

template<typename Weight>
inline void adjust(std::shared_ptr<struct CSC<Weight>> C_CSC,
                   const int32_t tid) {
                       
    double start_time = Env::tic();                       
    C_CSC->adjust(tid);
    if(!tid) Env::memory_time += Env::toc(start_time);
    Env::memory_allocation_time[tid] += Env::toc(start_time);
}

template<typename Weight>
inline void walk_by_tid(std::shared_ptr<struct CSC<Weight>> C_CSC,
                        const int32_t tid) {
     C_CSC->walk(false, tid);
}

template<typename Weight>
inline void walk_by_rank(std::shared_ptr<struct CSC<Weight>> A_CSC) {
    A_CSC->walk(false);
}

template<typename Weight>
inline void repopulate(std::shared_ptr<struct CSC<Weight>> A_CSC,
                       std::shared_ptr<struct CSC<Weight>> C_CSC,
                       const uint32_t start_col,
                       const uint32_t end_col,
                       const uint32_t dis_nnz,
                       const int32_t tid) {
    double start_time = Env::tic();
    //A_CSC->repopulate(C_CSC, start_col, end_col, dis_nnz, tid);
    if(!tid) Env::memory_time += Env::toc(start_time);
    Env::memory_allocation_time[tid] += Env::toc(start_time);
}

template<typename Weight>
inline void repopulate(std::shared_ptr<struct CSC<Weight>> A_CSC,
                       std::shared_ptr<struct CSC<Weight>> C_CSC,
                      const int32_t leader,
                       const int32_t tid,
                       const std::vector<int32_t> my_follower_threads) {
                           
    double start_time = Env::tic();
    //A_CSC->repopulate(C_CSC, tid, leader, my_follower_threads);
    if(tid == leader) Env::memory_time += Env::toc(start_time);
    Env::memory_allocation_time[tid] += Env::toc(start_time);
}

template<typename Weight>
inline bool validate_prediction(const std::shared_ptr<struct CSC<Weight>> A_CSC,
                                const std::vector<uint32_t> trueCategories,
                                const uint32_t start_row,
                                const int32_t tid) {
    const uint64_t A_nnz   = A_CSC->nnz;
    const uint32_t A_nrows = A_CSC->nrows;
    const uint32_t A_ncols = A_CSC->ncols;
    const uint32_t* A_IA   = A_CSC->IA_blk->ptr;
    const uint32_t* A_JA   = A_CSC->JA_blk->ptr;
    const Weight*    A_A   = A_CSC->A_blk->ptr;
    std::vector<uint32_t> allCategories(A_nrows);

    for(uint32_t j = 0; j < A_ncols; j++) {
        for(uint32_t i = A_JA[j]; i < A_JA[j+1]; i++) {
            allCategories[A_IA[i]] = 1;
        }
    }

    bool me = 1;
    for(uint32_t i = 0; i < A_nrows; i++) {
        if(trueCategories[start_row + i] != allCategories[i]) {
            me = 0;
            break;
        }
    }
    
    Env::checkconv[tid] = me;

    pthread_barrier_wait(&Env::thread_barrier);
    int32_t us = std::accumulate(Env::checkconv.begin(), Env::checkconv.end(), 0);
    int32_t all = 0;
    bool converged = false;

    if(!tid) {
        MPI_Allreduce(&us, &all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        converged = (all == (Env::nranks * Env::nthreads));
    }
    else {
        converged = (us == Env::nthreads);
    }

    return(converged);
}

template<typename Weight>
inline bool validate_prediction(const std::shared_ptr<struct CSC<Weight>> A_CSC,
                                const std::vector<uint32_t> trueCategories,
                                const uint32_t start_row) {
    const uint64_t A_nnz   = A_CSC->nnz;
    const uint32_t A_nrows = A_CSC->nrows;
    const uint32_t A_ncols = A_CSC->ncols;
    const uint32_t* A_IA   = A_CSC->IA_blk->ptr;
    const uint32_t* A_JA   = A_CSC->JA_blk->ptr;
    const Weight*    A_A   = A_CSC->A_blk->ptr;
    
    std::vector<uint32_t> allCategories(A_nrows);

    for(uint32_t j = 0; j < A_ncols; j++) {
        for(uint32_t i = A_JA[j]; i < A_JA[j+1]; i++) {
            allCategories[A_IA[i]] = 1;
        }
    }
    
    char me = 1;
    uint32_t j = 0;
    for(uint32_t i = 0; i < A_nrows; i++) {
        if(trueCategories[start_row + i] != allCategories[i]) {
            me = 0;
            break;
        }
    }
    char all = 0;
    MPI_Allreduce(&me, &all, 1, MPI_CHAR, MPI_SUM, MPI_COMM_WORLD);

    return((all == Env::nranks));
}
#endif