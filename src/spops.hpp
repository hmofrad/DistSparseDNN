/*
 * spops.hpp: Sparse Matrix operations implementation
 * Sparse Matrix - Sparse Matrix (SpMM)
 * (c) Mohammad Hasanzadeh Mofrad, 2020
 * (e) m.hasanzadeh.mofrad@gmail.com
 */
 
#ifndef SPOPS_H
#define SPOPS_H

#include "env.hpp"
#include "spmat.hpp"

template<typename Weight>
inline std::tuple<uint64_t, uint32_t, uint32_t> spmm_symb(std::shared_ptr<struct Compressed_Format<Weight>> A_SPMAT,
                                                          std::shared_ptr<struct Compressed_Format<Weight>> B_SPMAT,
                                                          std::shared_ptr<struct Data_Block<Weight>> s,
                                                          const uint32_t start,
                                                          const uint32_t end,
                                                          const int32_t tid) {
    
    uint64_t nnzmax = 0;
    uint32_t nrows;
    uint32_t ncols;
    
    uint64_t A_nnz;
    uint32_t A_nrows;
    uint32_t A_ncols;
    uint32_t* A_IA;
    uint32_t* A_JA;
    Weight*    A_A;
        
    uint64_t B_nnz;
    uint32_t B_nrows;
    uint32_t B_ncols;
    uint32_t* B_IA;
    uint32_t* B_JA;
    Weight*    B_A;
    
    Weight* s_A   = s->ptr;
    
    COMPRESSED_FORMAT compression_type = A_SPMAT->compression_type;
    if(compression_type == COMPRESSED_FORMAT::_CSC_) {
        const std::shared_ptr<struct CSC<Weight>> A_CSC = std::static_pointer_cast<struct CSC<Weight>>(A_SPMAT);
        A_nnz   = A_CSC->nnz;
        A_nrows = A_CSC->nrows;
        A_ncols = A_CSC->ncols;
        A_IA   = A_CSC->IA_blk->ptr;
        A_JA   = A_CSC->JA_blk->ptr;
        A_A   = A_CSC->A_blk->ptr;
    
        const std::shared_ptr<struct CSC<Weight>> B_CSC = std::static_pointer_cast<struct CSC<Weight>>(B_SPMAT);
        B_nnz   = B_CSC->nnz;
        B_nrows = B_CSC->nrows;
        B_ncols = B_CSC->ncols;
        B_IA   = B_CSC->IA_blk->ptr;
        B_JA   = B_CSC->JA_blk->ptr;
        B_A   = B_CSC->A_blk->ptr;
        
        if((A_ncols != B_nrows) or (s->nitems < A_nrows)) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "SpMM dimensions do not agree A[%d %d] B[%d %d], SPA[%lu]\n", A_nrows, A_ncols, B_nrows, B_ncols, s->nitems);
            std::exit(1); 
        }
		//printf("SpMM dimensions tid=%d A[%d %d] B[%d %d], SPA[%lu] [%d %d]\n", tid, A_nrows, A_ncols, B_nrows, B_ncols, s->nitems, start, end);
		//printf("tid=%d start=%d end=%d\n", tid, start, end);
        for(uint32_t j = start; j < end; j++) {
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
    }
    else if(compression_type == COMPRESSED_FORMAT::_CSR_) {
        const std::shared_ptr<struct CSR<Weight>> A_CSR = std::static_pointer_cast<struct CSR<Weight>>(A_SPMAT);
        A_nnz   = A_CSR->nnz;
        A_nrows = A_CSR->nrows;
        A_ncols = A_CSR->ncols;
        A_IA   = A_CSR->IA_blk->ptr;
        A_JA   = A_CSR->JA_blk->ptr;
        A_A   = A_CSR->A_blk->ptr;
    
        const std::shared_ptr<struct CSR<Weight>> B_CSR = std::static_pointer_cast<struct CSR<Weight>>(B_SPMAT);
        B_nnz   = B_CSR->nnz;
        B_nrows = B_CSR->nrows;
        B_ncols = B_CSR->ncols;
        B_IA   = B_CSR->IA_blk->ptr;
        B_JA   = B_CSR->JA_blk->ptr;
        B_A   = B_CSR->A_blk->ptr;
        
        if((A_ncols != B_nrows) or (s->nitems < B_ncols)) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "SpMM dimensions do not agree A[%d %d] B[%d %d], SPA[%lu]\n", A_nrows, A_ncols, B_nrows, B_ncols, s->nitems);
            std::exit(1); 
        }

        for(uint32_t i = start; i < end; i++) {
            for(uint32_t k = A_IA[i]; k < A_IA[i+1]; k++) {
                uint32_t l = A_JA[k];
                for(uint32_t n = B_IA[l]; n < B_IA[l+1]; n++) {
                    s_A[B_JA[n]] = 1;
                }
            }
            for(uint32_t j = 0; j < B_ncols; j++) {
                if(s_A[j]){
                    nnzmax++;
                    s_A[j] = 0;
                }
            }
        }
    }
    else {
        Logging::print(Logging::LOG_LEVEL::ERROR, "%s compression not implemented\n", COMPRESSED_FORMATS[compression_type]);
        std::exit(Env::finalize());
    }
    
    nrows = A_nrows;
    ncols = B_ncols;

    return std::make_tuple(nnzmax, nrows, ncols);
}




template<typename Weight>
inline void spmm_real(std::shared_ptr<struct Compressed_Format<Weight>> A_SPMAT,
                      std::shared_ptr<struct Compressed_Format<Weight>> B_SPMAT,
                      std::shared_ptr<struct Compressed_Format<Weight>> C_SPMAT,
                      std::shared_ptr<struct Data_Block<Weight>> s,
                      const std::shared_ptr<struct Data_Block<Weight>> b,
					  Weight(*activation_function)(Weight),
                      const uint32_t start,
                      const uint32_t end,
                      const uint32_t off,
                      uint64_t& idx_nnz,
                      const int32_t tid) {
    
    uint64_t A_nnz;
    uint32_t A_nrows;
    uint32_t A_ncols;
    uint32_t* A_IA;
    uint32_t* A_JA;
    Weight*    A_A;
        
    uint64_t B_nnz;
    uint32_t B_nrows;
    uint32_t B_ncols;
    uint32_t* B_IA;
    uint32_t* B_JA;
    Weight*    B_A;
        
    uint64_t C_nnz;
    uint32_t C_nrows;
    uint32_t C_ncols;
    uint32_t* C_IA;
    uint32_t* C_JA;
    Weight*    C_A;
    
    Weight*       s_A = s->ptr;
    const Weight* b_A = b->ptr;
    
    COMPRESSED_FORMAT compression_type = A_SPMAT->compression_type;
    if(compression_type == COMPRESSED_FORMAT::_CSC_) {
        const std::shared_ptr<struct CSC<Weight>> A_CSC = std::static_pointer_cast<struct CSC<Weight>>(A_SPMAT);
        A_nnz   = A_CSC->nnz;
        A_nrows = A_CSC->nrows;
        A_ncols = A_CSC->ncols;
        A_IA   = A_CSC->IA_blk->ptr;
        A_JA   = A_CSC->JA_blk->ptr;
        A_A   = A_CSC->A_blk->ptr;
        
        const std::shared_ptr<struct CSC<Weight>> B_CSC = std::static_pointer_cast<struct CSC<Weight>>(B_SPMAT);          
        B_nnz   = B_CSC->nnz;
        B_nrows = B_CSC->nrows;
        B_ncols = B_CSC->ncols;
        B_IA   = B_CSC->IA_blk->ptr;
        B_JA   = B_CSC->JA_blk->ptr;
        B_A   = B_CSC->A_blk->ptr;
            
        const std::shared_ptr<struct CSC<Weight>> C_CSC = std::static_pointer_cast<struct CSC<Weight>>(C_SPMAT);              
        C_nnz   = C_CSC->nnz;
        C_nrows = C_CSC->nrows;
        C_ncols = C_CSC->ncols;
        C_IA   = C_CSC->IA_blk->ptr;
        C_JA   = C_CSC->JA_blk->ptr;
        C_A   = C_CSC->A_blk->ptr;
                        
        if((A_ncols != B_nrows) or (s->nitems < A_nrows)) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "SpMM dimensions do not agree C[%d %d] != A[%d %d] B[%d %d], SPA[%lu]\n", C_nrows, C_ncols, A_nrows, A_ncols, B_nrows, B_ncols, s->nitems);
            std::exit(1); 
        }
		//printf("SpMM dimensions tid=%d C[%d %d] != A[%d %d] B[%d %d], SPA[%lu] [%d %d]\n", tid, C_nrows, C_ncols, A_nrows, A_ncols, B_nrows, B_ncols, s->nitems, start, end);
        for(uint32_t j = start; j < end; j++) {
            for(uint32_t k = B_JA[j]; k < B_JA[j+1]; k++) {
                uint32_t l = B_IA[k];
                for(uint32_t n = A_JA[l]; n < A_JA[l+1]; n++) {
                    s_A[A_IA[n]] += (B_A[k] * A_A[n]);
                }
            }
			//printf("%d %d %d %d\n", j, B_JA[j], B_JA[j+1], B_JA[j+1]-B_JA[j]);
            //C_CSC->populate_spa(&s_A, b_A, off + j, idx_nnz, tid);
			C_CSC->populate_spa(&s_A, b_A, off + j, idx_nnz, activation_function, tid);
			//if(tid==Env::nthreads-1) printf("%d %d %lu\n", off, j, idx_nnz);
        }
    }
    else if(compression_type == COMPRESSED_FORMAT::_CSR_) {
        const std::shared_ptr<struct CSR<Weight>> A_CSR = std::static_pointer_cast<struct CSR<Weight>>(A_SPMAT);
        A_nnz   = A_CSR->nnz;
        A_nrows = A_CSR->nrows;
        A_ncols = A_CSR->ncols;
        A_IA   = A_CSR->IA_blk->ptr;
        A_JA   = A_CSR->JA_blk->ptr;
        A_A   = A_CSR->A_blk->ptr;
        
        const std::shared_ptr<struct CSR<Weight>> B_CSR = std::static_pointer_cast<struct CSR<Weight>>(B_SPMAT);          
        B_nnz   = B_CSR->nnz;
        B_nrows = B_CSR->nrows;
        B_ncols = B_CSR->ncols;
        B_IA   = B_CSR->IA_blk->ptr;
        B_JA   = B_CSR->JA_blk->ptr;
        B_A   = B_CSR->A_blk->ptr;
            
        const std::shared_ptr<struct CSR<Weight>> C_CSR = std::static_pointer_cast<struct CSR<Weight>>(C_SPMAT);              
        C_nnz   = C_CSR->nnz;
        C_nrows = C_CSR->nrows;
        C_ncols = C_CSR->ncols;
        C_IA   = C_CSR->IA_blk->ptr;
        C_JA   = C_CSR->JA_blk->ptr;
        C_A   = C_CSR->A_blk->ptr;

        if((A_ncols != B_nrows) or (s->nitems < B_ncols)) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "SpMM dimensions do not agree C[%d %d] != A[%d %d] B[%d %d], SPA[%lu] Bias[%lu]\n", C_nrows, C_ncols, A_nrows, A_ncols, B_nrows, B_ncols, s->nitems, b->nitems);
            std::exit(1); 
        }

        for(uint32_t i = start; i < end; i++) {
            for(uint32_t k = A_IA[i]; k < A_IA[i+1]; k++) {
                uint32_t l = A_JA[k];
                for(uint32_t n = B_IA[l]; n < B_IA[l+1]; n++) {
                    s_A[B_JA[n]] += (A_A[k] * B_A[n]);
                }
            }
            C_CSR->populate_spa(&s_A, b_A, off + i, idx_nnz, tid);
        }        
    }
    else {
        Logging::print(Logging::LOG_LEVEL::ERROR, "%s compression not implemented\n", COMPRESSED_FORMATS[compression_type]);
        std::exit(Env::finalize());
    }
}


template<typename Weight>
inline void data_x_model_1_iter(std::shared_ptr<struct Compressed_Format<Weight>> A_SPMAT, 
                                std::shared_ptr<struct Compressed_Format<Weight>> B_SPMAT, 
                                std::shared_ptr<struct Compressed_Format<Weight>> C_SPMAT, 
                                std::shared_ptr<struct Data_Block<Weight>> s_spa,
                                const std::shared_ptr<struct Data_Block<Weight>> b_bias,
								Weight(*activation_function)(Weight),
                                const uint32_t nrows,
                                const uint32_t ncols,
                                const uint32_t start,
                                const uint32_t end,
                                const uint32_t sub_start,
                                const uint32_t sub_end,
                                struct Env::thread_struct& thread_st,
                                const int32_t leader_tid, 
                                const int32_t tid) {
    
    COMPRESSED_FORMAT compression_type = A_SPMAT->compression_type;   
    if((compression_type == COMPRESSED_FORMAT::_CSC_) or (compression_type == COMPRESSED_FORMAT::_CSR_)) {
        double start_time = 0;
		//printf("spmm_symb start tid=%d\n", tid);
        start_time = Env::tic(); 
            std::tie(thread_st.off_nnz, std::ignore, std::ignore) =  spmm_symb(A_SPMAT, B_SPMAT, s_spa, start, end, tid);
            pthread_barrier_wait(&Env::thread_barrier);
        Env::spmm_symb_time[tid] += Env::toc(start_time);   
		//printf("spmm_symb done tid=%d %lu\n", tid, thread_st.off_nnz);
		
        start_time = Env::tic();
            uint64_t nnz = Env::adjust_nnz(leader_tid, tid);
            C_SPMAT->reallocate(nnz, nrows, ncols, leader_tid, tid);
        Env::memory_allocation_time[tid] += Env::toc(start_time);
		//printf("spmm_symb done tid=%d nnz=%lu\n", tid, nnz);
		//pthread_barrier_wait(&Env::thread_barrier);
		//std::exit(0);
        start_time = Env::tic();
            pthread_barrier_wait(&Env::thread_barrier);
            spmm_real(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, activation_function, start, end, sub_start, thread_st.idx_nnz, tid);
            pthread_barrier_wait(&Env::thread_barrier);
            Env::adjust_displacement(tid);
            C_SPMAT->adjust(leader_tid, tid);	
        Env::spmm_real_time[tid] += Env::toc(start_time);

        start_time = Env::tic();
            pthread_barrier_wait(&Env::thread_barrier);
            A_SPMAT->repopulate(C_SPMAT, thread_st.dis_nnz, leader_tid, tid);
        Env::memory_allocation_time[tid] += Env::toc(start_time);
        //A_SPMAT->walk_dxm(false, leader_tid, tid);
   }
   else {
        Logging::print(Logging::LOG_LEVEL::ERROR, "%s compression not implemented\n", COMPRESSED_FORMATS[compression_type]);
        std::exit(Env::finalize());
   }
}


template<typename Weight>
inline void data_x_model_validate_prediction(const std::shared_ptr<struct Compressed_Format<Weight>> A_SPMAT,
                                             const uint32_t start_row,
                                             const std::vector<uint32_t> trueCategories,
                                             const uint32_t npredicted_instances,
                                             const int32_t leader_tid, 
                                             const int32_t tid) {                  
    if(tid == leader_tid) {
        std::vector<uint32_t> allCategories;
        uint64_t A_nnz;
        uint32_t A_nrows;
        uint32_t A_ncols;
        uint32_t* A_IA;
        uint32_t* A_JA;
        Weight*    A_A;
        
        COMPRESSED_FORMAT compression_type = A_SPMAT->compression_type;  
        if(compression_type == COMPRESSED_FORMAT::_CSC_) {
            const std::shared_ptr<struct CSC<Weight>> A_CSC = std::static_pointer_cast<struct CSC<Weight>>(A_SPMAT);
            A_nnz   = A_CSC->nnz;
            A_nrows = A_CSC->nrows;
            A_ncols = A_CSC->ncols;
            A_IA   = A_CSC->IA_blk->ptr;
            A_JA   = A_CSC->JA_blk->ptr;
            A_A   = A_CSC->A_blk->ptr;
            
            allCategories.resize(A_nrows);
            for(uint32_t j = 0; j < A_ncols; j++) {
                for(uint32_t i = A_JA[j]; i < A_JA[j+1]; i++) {
                    allCategories[A_IA[i]] = 1;
                }
            }
            
        }
        else if(compression_type == COMPRESSED_FORMAT::_CSR_) {
            const std::shared_ptr<struct CSR<Weight>> A_CSR = std::static_pointer_cast<struct CSR<Weight>>(A_SPMAT);
            A_nnz   = A_CSR->nnz;
            A_nrows = A_CSR->nrows;
            A_ncols = A_CSR->ncols;
            A_IA   = A_CSR->IA_blk->ptr;
            A_JA   = A_CSR->JA_blk->ptr;
            A_A   = A_CSR->A_blk->ptr;
            
            allCategories.resize(A_nrows);
            for(uint32_t i = 0; i < A_nrows; i++) {
                for(uint32_t j = A_IA[i]; j < A_IA[i+1]; j++) {
                    allCategories[i] = 1;
                }
            }
        }
        else {
            Logging::print(Logging::LOG_LEVEL::ERROR, "%s compression not implemented\n", COMPRESSED_FORMATS[compression_type]);
            std::exit(Env::finalize());
        }

        int count = 0;
        for(uint32_t i = 0; i < A_nrows; i++) {
            if(trueCategories[start_row + i] != allCategories[i]) {
                break;
            }
            if(allCategories[i]) {
                count++;
            }
        }
        uint32_t counts = 0;
        MPI_Allreduce(&count, &counts, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
        
        bool passed = (counts == npredicted_instances);
        if(passed) {
            Logging::print(Logging::LOG_LEVEL::INFO, "Challenge PASSED.\n");
        }
        else {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Challenge FAILED.\n");
        }
    }
    pthread_barrier_wait(&Env::thread_barrier);
    Env::barrier();
}


template<typename Weight>
inline void data_x_data_1_iter(std::shared_ptr<struct Compressed_Format<Weight>> A_SPMAT, 
                               std::shared_ptr<struct Compressed_Format<Weight>> B_SPMAT, 
                               std::shared_ptr<struct Compressed_Format<Weight>> C_SPMAT, 
                               std::shared_ptr<struct Data_Block<Weight>> s_spa,
                               const std::shared_ptr<struct Data_Block<Weight>> b_bias,
							   Weight(*activation_function)(Weight),
                               const uint32_t nrows,
                               const uint32_t ncols,
                               const uint32_t start,
                               const uint32_t end,
                               const uint32_t off,
                               struct Env::thread_struct& thread_st,
                               int32_t leader_tid, 
                               const int32_t tid) {
    
    COMPRESSED_FORMAT compression_type = A_SPMAT->compression_type;    
    if((A_SPMAT->compression_type == COMPRESSED_FORMAT::_CSC_) or (A_SPMAT->compression_type == COMPRESSED_FORMAT::_CSR_)) {
        double start_time = 0;
        start_time = Env::tic();
            std::tie(thread_st.off_nnz, std::ignore, std::ignore) =  spmm_symb(A_SPMAT, B_SPMAT, s_spa, start, end, tid);
        Env::spmm_symb_time[tid] += Env::toc(start_time);      
        
        start_time = Env::tic();
            leader_tid = -1;
            uint64_t nnz = thread_st.off_nnz;
            C_SPMAT->reallocate(thread_st.off_nnz, nrows, ncols, leader_tid, tid);
        Env::memory_allocation_time[tid] += Env::toc(start_time);
        
        start_time = Env::tic();
            thread_st.idx_nnz = 0;
            spmm_real(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, activation_function, start, end, off, thread_st.idx_nnz, tid);
            Env::adjust_displacement(tid);
            C_SPMAT->adjust(tid);
        Env::spmm_real_time[tid] += Env::toc(start_time);                              
    
        //leader_tid = 0;
        //C_SPMAT->walk_dxd(false, leader_tid, tid);
   }
}

template<typename Weight>
inline void data_x_data_validate_prediction(const std::shared_ptr<struct Compressed_Format<Weight>> A_SPMAT,
                                const uint32_t start_row,
                                const std::vector<uint32_t> true_categories,
                                const uint32_t npredicted_instances,
                                const int32_t leader_tid, 
                                const int32_t tid) {

    uint64_t A_nnz;
    uint32_t A_nrows;
    uint32_t A_ncols;
    uint32_t* A_IA;
    uint32_t* A_JA;
    Weight*    A_A;   

    std::vector<uint32_t> allCategories;
    COMPRESSED_FORMAT compression_type = A_SPMAT->compression_type;  
    if(compression_type == COMPRESSED_FORMAT::_CSC_) {
        const std::shared_ptr<struct CSC<Weight>> A_CSC = std::static_pointer_cast<struct CSC<Weight>>(A_SPMAT);
        A_nnz   = A_CSC->nnz;
        A_nrows = A_CSC->nrows;
        A_ncols = A_CSC->ncols;
        A_IA   = A_CSC->IA_blk->ptr;
        A_JA   = A_CSC->JA_blk->ptr;
        A_A   = A_CSC->A_blk->ptr;
        
        allCategories.resize(A_nrows);
        for(uint32_t j = 0; j < A_ncols; j++) {
            for(uint32_t i = A_JA[j]; i < A_JA[j+1]; i++) {
                allCategories[A_IA[i]] = 1;
            }
        }
    }
    else if(compression_type == COMPRESSED_FORMAT::_CSR_) {
        const std::shared_ptr<struct CSR<Weight>> A_CSR = std::static_pointer_cast<struct CSR<Weight>>(A_SPMAT);
        A_nnz   = A_CSR->nnz;
        A_nrows = A_CSR->nrows;
        A_ncols = A_CSR->ncols;
        A_IA   = A_CSR->IA_blk->ptr;
        A_JA   = A_CSR->JA_blk->ptr;
        A_A   = A_CSR->A_blk->ptr;
        
        allCategories.resize(A_nrows);
        for(uint32_t i = 0; i < A_nrows; i++) {
            for(uint32_t j = A_IA[i]; j < A_IA[i+1]; j++) {
                allCategories[i] = 1;
            }
        }
    }
    else {
        Logging::print(Logging::LOG_LEVEL::ERROR, "%s compression not implemented\n", COMPRESSED_FORMATS[compression_type]);
        std::exit(Env::finalize());
    }    

    int count = 0;
    for(uint32_t i = 0; i < A_nrows; i++) {
        if(true_categories[start_row + i] != allCategories[i]) {
            break;
        }
        if(allCategories[i]) {
            count++;
        }
    }

    Env::counters[tid].checkcount = count;
    pthread_barrier_wait(&Env::thread_barrier);
    if(tid == leader_tid) {
        uint32_t counts = 0;
        for(auto counter: Env::counters) {
            counts += counter.checkcount;
        }
        uint32_t countss = 0;
        MPI_Allreduce(&counts, &countss, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
        
        bool passed = (countss == npredicted_instances);
        if(passed) {
            Logging::print(Logging::LOG_LEVEL::INFO, "Challenge PASSED.\n");
        }
        else {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Challenge FAILED.\n");
        }
    }
    pthread_barrier_wait(&Env::thread_barrier);
    Env::barrier();
}

template<typename Weight>
inline void data_x_model_hybrid_1_iter(std::shared_ptr<struct Compressed_Format<Weight>> A_SPMAT, 
                                std::shared_ptr<struct Compressed_Format<Weight>> B_SPMAT, 
                                std::shared_ptr<struct Compressed_Format<Weight>> C_SPMAT, 
                                std::shared_ptr<struct Data_Block<Weight>> s_spa,
                                const std::shared_ptr<struct Data_Block<Weight>> b_bias,
								Weight(*activation_function)(Weight),
                                const uint32_t nrows,
                                const uint32_t ncols,
                                const uint32_t start,
                                const uint32_t end,
                                const uint32_t off,
                                const std::deque<int32_t> my_threads,
                                struct Env::thread_struct& thread_st,
                                const int32_t leader_tid, 
                                const int32_t tid) {

    COMPRESSED_FORMAT compression_type = A_SPMAT->compression_type;    
    if((compression_type == COMPRESSED_FORMAT::_CSC_) or (compression_type == COMPRESSED_FORMAT::_CSR_)) {
        double start_time = 0;

        if(tid ==leader_tid) start_time = Env::tic(); 
            std::tie(thread_st.off_nnz, std::ignore, std::ignore) =  spmm_symb(A_SPMAT, B_SPMAT, s_spa, start, end, tid);
            pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
        if(tid ==leader_tid) Env::spmm_symb_time[tid] += Env::toc(start_time);   

        if(tid ==leader_tid) start_time = Env::tic();
            uint64_t nnz = Env::adjust_nnz(my_threads, leader_tid, tid);
            C_SPMAT->reallocate(nnz, nrows, ncols, leader_tid, tid);
        if(tid ==leader_tid) Env::memory_allocation_time[tid] += Env::toc(start_time);

        if(tid ==leader_tid) start_time = Env::tic();
            pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
            spmm_real(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, activation_function, start, end, off, thread_st.idx_nnz, tid);
            pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
            Env::adjust_displacement(my_threads, leader_tid, tid);
            C_SPMAT->adjust(my_threads, leader_tid, tid);	
        if(tid ==leader_tid) Env::spmm_real_time[tid] += Env::toc(start_time);

        if(tid ==leader_tid) start_time = Env::tic();
            pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
            A_SPMAT->repopulate(C_SPMAT, my_threads, leader_tid, tid);
        if(tid ==leader_tid) Env::memory_allocation_time[tid] += Env::toc(start_time);
    }
    else {
        Logging::print(Logging::LOG_LEVEL::ERROR, "%s compression not implemented\n", COMPRESSED_FORMATS[compression_type]);
        std::exit(Env::finalize());
    }    
}

template<typename Weight>
inline void manager_x_worker_validate_prediction(std::vector<std::vector<struct Tile<Weight>>> tiles,
                                const std::vector<uint32_t> true_categories,
                                const uint32_t npredicted_instances,
                                const int32_t leader_tid, 
                                const int32_t tid) {
    pthread_barrier_wait(&Env::thread_barrier);                                        
    if(tid == leader_tid) {
        int count = 0;
        for(uint32_t rowgroup:  Env::processed_rowgroups) {
            uint32_t start_row;
            
            uint64_t A_nnz;
            uint32_t A_nrows;
            uint32_t A_ncols;
            uint32_t* A_IA;
            uint32_t* A_JA;
            Weight*    A_A;
            
            struct Tile<Weight>& A_tile = tiles[rowgroup][0];
            start_row = A_tile.start_row;
            std::shared_ptr<struct Compressed_Format<Weight>> A_SPMAT = A_tile.spmat;
            std::vector<uint32_t> allCategories;
            COMPRESSED_FORMAT compression_type = A_SPMAT->compression_type;  
            if(compression_type == COMPRESSED_FORMAT::_CSC_) {
                const std::shared_ptr<struct CSC<Weight>> A_CSC = std::static_pointer_cast<struct CSC<Weight>>(A_SPMAT);
                A_nnz   = A_CSC->nnz;
                A_nrows = A_CSC->nrows;
                A_ncols = A_CSC->ncols;
                A_IA   = A_CSC->IA_blk->ptr;
                A_JA   = A_CSC->JA_blk->ptr;
                A_A   = A_CSC->A_blk->ptr;

                allCategories.resize(A_nrows);  
                for(uint32_t j = 0; j < A_ncols; j++) {
                    for(uint32_t i = A_JA[j]; i < A_JA[j+1]; i++) {
                        allCategories[A_IA[i]] = 1;
                    }
                }                    
            }
            else if(compression_type == COMPRESSED_FORMAT::_CSR_) {
                const std::shared_ptr<struct CSR<Weight>> A_CSR = std::static_pointer_cast<struct CSR<Weight>>(A_SPMAT);
                A_nnz   = A_CSR->nnz;
                A_nrows = A_CSR->nrows;
                A_ncols = A_CSR->ncols;
                A_IA   = A_CSR->IA_blk->ptr;
                A_JA   = A_CSR->JA_blk->ptr;
                A_A   = A_CSR->A_blk->ptr;
                
                allCategories.resize(A_nrows);
                for(uint32_t i = 0; i < A_nrows; i++) {
                    for(uint32_t j = A_IA[i]; j < A_IA[i+1]; j++) {
                        allCategories[i] = 1;
                    }
                } 
            }
            else {
                Logging::print(Logging::LOG_LEVEL::ERROR, "%s compression not implemented\n", COMPRESSED_FORMATS[compression_type]);
                std::exit(Env::finalize());
            }    
            
            for(uint32_t i = 0; i < A_nrows; i++) {
                if(true_categories[start_row + i] != allCategories[i]) {
                    break;
                }
                if(allCategories[i]) {
                    count++;
                }
            }
        }
        
        uint32_t counts = 0;
        MPI_Allreduce(&count, &counts, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
        bool passed = (counts == npredicted_instances);
        if(passed) {
            Logging::print(Logging::LOG_LEVEL::INFO, "Challenge PASSED.\n");
        }
        else {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Challenge FAILED.\n");
        }
    }
    pthread_barrier_wait(&Env::thread_barrier);
    Env::barrier();
}

template<typename Weight>
inline void work_x_stealing_validate_prediction(std::vector<std::vector<struct Tile<Weight>>> tiles,
                                const std::vector<uint32_t> true_categories,
                                const uint32_t npredicted_instances,
                                const int32_t leader_tid, 
                                const int32_t tid) {
    pthread_barrier_wait(&Env::thread_barrier);     
    if(tid == leader_tid) {
        for(auto p: Env::processed_rowgroups_per_thread) {
            Env::processed_rowgroups.insert(Env::processed_rowgroups.end(), p.begin(), p.end());

        }
    }
    pthread_barrier_wait(&Env::thread_barrier);   
    manager_x_worker_validate_prediction(tiles, true_categories, npredicted_instances, leader_tid, tid);
}



/*
template<typename Weight>
inline bool validate_prediction(const std::shared_ptr<struct CSC<Weight>> A_CSC,
                                const std::vector<uint32_t> true_categories,
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
        if(true_categories[start_row + i] != allCategories[i]) {
            me = 0;
            break;
        }
    }

    Env::counters[tid].checkconv = me;
    pthread_barrier_wait(&Env::thread_barrier);
    
    int32_t us = 0;
    for(auto counter: Env::counters) {
        us += counter.checkconv;
    }
    
    bool converged = false;
    int32_t all = 0;
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
                                const std::vector<uint32_t> true_categories,
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
    for(uint32_t i = 0; i < A_nrows; i++) {
        if(true_categories[start_row + i] != allCategories[i]) {
            me = 0;
            break;
        }
    }
    char all = 0;
    MPI_Allreduce(&me, &all, 1, MPI_CHAR, MPI_SUM, MPI_COMM_WORLD);

    return((all == Env::nranks));
}

template<typename Weight>
inline int32_t validate_prediction1(const std::shared_ptr<struct CSC<Weight>> A_CSC,
                                const std::vector<uint32_t> true_categories,
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
    
    int32_t count = 0;
    for(uint32_t i = 0; i < A_nrows; i++) {
        if(true_categories[start_row + i] != allCategories[i]) {
            count = -1;
            break;
        }
        if(allCategories[i]) {
            count++;
        }
    }
    return(count);
}
*/
#endif