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

#include <math.h>
template<typename Weight>
Weight sigmoid(Weight x) { return 1 / (1 + exp(-x)); }

/* input x layers */
//enum MULTIPLICATION_TYPE {_DENSE_X_DENSE_, _DENSE_X_COMPRESSED_, _COMPRESSED_X_COMPRESSED_, _COMPRESSED_X_DOUBLY_COMPRESSED_, _COMPRESSED_X_TRIPLY_COMPRESSED_, _M_SIZE_};
//const char* MULTIPLICATION_TYPES[] = {"_DENSE_X_DENSE_", "_DENSE_X_COMPRESSED_", "_COMPRESSED_X_COMPRESSED_", "_COMPRESSED_X_DOUBLY_COMPRESSED_", "_COMPRESSED_X_TRIPLY_COMPRESSED_", "_M_SIZE_"};


template<typename Weight>
inline std::tuple<uint64_t, uint32_t, uint32_t> spmm_symb(std::shared_ptr<struct Compressed_Format<Weight>> A_SPMAT,
                                                          std::shared_ptr<struct Compressed_Format<Weight>> B_SPMAT,
                                                          std::shared_ptr<struct Data_Block<Weight>> s,
                                                          const uint32_t start, const uint32_t end,
                                                          COMPRESSED_FORMAT input_compression_type, COMPRESSED_FORMAT layer_compression_type,
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
    if((input_compression_type == COMPRESSED_FORMAT::_CSC_) and (layer_compression_type == COMPRESSED_FORMAT::_CSC_)) {
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

        for(uint32_t j = start; j < end; j++) {
            for(uint32_t i = B_JA[j]; i < B_JA[j+1]; i++) {
                uint32_t r = B_IA[i];
                for(uint32_t k = A_JA[r]; k < A_JA[r+1]; k++) {
                    s_A[A_IA[k]] = 1;
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
    else if((input_compression_type == COMPRESSED_FORMAT::_CSR_) and (layer_compression_type == COMPRESSED_FORMAT::_CSR_)) {
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
            for(uint32_t j = A_IA[i]; j < A_IA[i+1]; j++) {
                uint32_t c = A_JA[j];
                for(uint32_t k = B_IA[c]; k < B_IA[c+1]; k++) {
                    s_A[B_JA[k]] = 1;
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
    else if((input_compression_type == COMPRESSED_FORMAT::_CSC_) and (layer_compression_type == COMPRESSED_FORMAT::_UDC_)) {
        const std::shared_ptr<struct CSC<Weight>> A_CSC = std::static_pointer_cast<struct CSC<Weight>>(A_SPMAT);
        A_nnz   = A_CSC->nnz;
        A_nrows = A_CSC->nrows;
        A_ncols = A_CSC->ncols;
        A_IA    = A_CSC->IA_blk->ptr;
        A_JA    = A_CSC->JA_blk->ptr;
        A_A     = A_CSC->A_blk->ptr;
        
        const std::shared_ptr<struct UDC<Weight>> B_UDC = std::static_pointer_cast<struct UDC<Weight>>(B_SPMAT);          
        B_nrows = B_UDC->nrows;
        B_ncols = B_UDC->ncols;
        B_A   = B_UDC->A_blk->ptr;
        
        for(uint32_t j = start; j < end; j++) {
            for(uint32_t i = 0; i < B_nrows; i++) {
                uint64_t k = j*B_nrows + i;
                if(B_A[k]) {
                    for(uint32_t l = A_JA[i]; l < A_JA[i+1]; l++) {
                        s_A[A_IA[l]] = 1;
                    }
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
    else if(not(((input_compression_type == COMPRESSED_FORMAT::_UDC_) and (layer_compression_type == COMPRESSED_FORMAT::_UDC_)) or
                ((input_compression_type == COMPRESSED_FORMAT::_UDC_) and (layer_compression_type == COMPRESSED_FORMAT::_UDC_)))) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "[%sx%s] multiplication not implemented\n", COMPRESSED_FORMATS[input_compression_type], COMPRESSED_FORMATS[layer_compression_type]);
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
                      const uint32_t start, const uint32_t end, const uint32_t off, uint64_t& idx_nnz,
                      COMPRESSED_FORMAT input_compression_type, COMPRESSED_FORMAT layer_compression_type,
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
    
    if((input_compression_type == COMPRESSED_FORMAT::_CSC_) and (layer_compression_type == COMPRESSED_FORMAT::_CSC_)) {
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

        for(uint32_t j = start; j < end; j++) {
            for(uint32_t i = B_JA[j]; i < B_JA[j+1]; i++) {
                uint32_t r = B_IA[i]; Weight v = B_A[i]; 
                for(uint32_t k = A_JA[r]; k < A_JA[r+1]; k++) {
                    s_A[A_IA[k]] += (A_A[k] * v);
                }
            }
            C_CSC->populate_spa(&s_A, b_A, off + j, idx_nnz, activation_function, tid);
        }
    }
    else if((input_compression_type == COMPRESSED_FORMAT::_CSR_) and (layer_compression_type == COMPRESSED_FORMAT::_CSR_)) {
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
            for(uint32_t j = A_IA[i]; j < A_IA[i+1]; j++) {
                uint32_t c = A_JA[j]; Weight v = A_A[j];
                for(uint32_t k = B_IA[c]; k < B_IA[c+1]; k++) {
                    s_A[B_JA[k]] += (v * B_A[k]);
                }
            }
            C_CSR->populate_spa(&s_A, b_A, off + i, idx_nnz, activation_function, tid);
        } 
    }
    else if((input_compression_type == COMPRESSED_FORMAT::_CSC_) and (layer_compression_type == COMPRESSED_FORMAT::_UDC_)) {
        const std::shared_ptr<struct CSC<Weight>> A_CSC = std::static_pointer_cast<struct CSC<Weight>>(A_SPMAT);
        A_nnz   = A_CSC->nnz;
        A_nrows = A_CSC->nrows;
        A_ncols = A_CSC->ncols;
        A_IA   = A_CSC->IA_blk->ptr;
        A_JA   = A_CSC->JA_blk->ptr;
        A_A   = A_CSC->A_blk->ptr;
        
        const std::shared_ptr<struct UDC<Weight>> B_UDC = std::static_pointer_cast<struct UDC<Weight>>(B_SPMAT);          
        B_nrows = B_UDC->nrows;
        B_ncols = B_UDC->ncols;
        B_A   = B_UDC->A_blk->ptr;
            
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
        
        for(uint32_t j = start; j < end; j++) {
            for(uint32_t i = 0; i < B_nrows; i++) {
                uint64_t k = j*B_nrows + i;
                if(B_A[k]) {
                    for(uint32_t l = A_JA[i]; l < A_JA[i+1]; l++) {
                        s_A[A_IA[l]] += (A_A[l] * B_A[k]);
                    }
                }
            }
            C_CSC->populate_spa(&s_A, b_A, off + j, idx_nnz, activation_function, tid);
        }
    }
    else if((input_compression_type == COMPRESSED_FORMAT::_UDC_) and (layer_compression_type == COMPRESSED_FORMAT::_UDC_)) {
        const std::shared_ptr<struct UDC<Weight>> A_UDC = std::static_pointer_cast<struct UDC<Weight>>(A_SPMAT);
        A_nrows = A_UDC->nrows;
        A_ncols = A_UDC->ncols;
        A_A   = A_UDC->A_blk->ptr;
        
        const std::shared_ptr<struct UDC<Weight>> B_UDC = std::static_pointer_cast<struct UDC<Weight>>(B_SPMAT);          
        B_nrows = B_UDC->nrows;
        B_ncols = B_UDC->ncols;
        B_A   = B_UDC->A_blk->ptr;
            
        const std::shared_ptr<struct UDC<Weight>> C_UDC = std::static_pointer_cast<struct UDC<Weight>>(C_SPMAT);              
        C_nrows = C_UDC->nrows;
        C_ncols = C_UDC->ncols;
        C_A   = C_UDC->A_blk->ptr;
        
        if((A_ncols != B_nrows) or (s->nitems < A_nrows)) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "SpMM dimensions do not agree C[%d %d] != A[%d %d] B[%d %d], SPA[%lu]\n", C_nrows, C_ncols, A_nrows, A_ncols, B_nrows, B_ncols, s->nitems);
            std::exit(1); 
        }

        for(uint32_t j = start; j < end; j++) {
            for(uint32_t i = 0; i < B_nrows; i++) {
                uint64_t k = j*B_nrows + i;
                if(B_A[k]) {
                    for(uint32_t l = 0; l < A_nrows; l++) {
                        uint32_t m = i*A_nrows + l;
                        s_A[l] += (A_A[m] * B_A[k]);
                    }
                }
            }
            C_UDC->populate_spa(&s_A, b_A, off + j, idx_nnz, activation_function, tid);            
        }
    }
    else if((input_compression_type == COMPRESSED_FORMAT::_UDC_) and (layer_compression_type == COMPRESSED_FORMAT::_CSC_)) {
        const std::shared_ptr<struct UDC<Weight>> A_UDC = std::static_pointer_cast<struct UDC<Weight>>(A_SPMAT);
        A_nrows = A_UDC->nrows;
        A_ncols = A_UDC->ncols;
        A_A   = A_UDC->A_blk->ptr;
        
        const std::shared_ptr<struct CSC<Weight>> B_CSC = std::static_pointer_cast<struct CSC<Weight>>(B_SPMAT);          
        B_nnz   = B_CSC->nnz;
        B_nrows = B_CSC->nrows;
        B_ncols = B_CSC->ncols;
        B_IA   = B_CSC->IA_blk->ptr;
        B_JA   = B_CSC->JA_blk->ptr;
        B_A   = B_CSC->A_blk->ptr;
        
        const std::shared_ptr<struct UDC<Weight>> C_UDC = std::static_pointer_cast<struct UDC<Weight>>(C_SPMAT);              
        C_nrows = C_UDC->nrows;
        C_ncols = C_UDC->ncols;
        C_A   = C_UDC->A_blk->ptr;
        
        if((A_ncols != B_nrows) or (s->nitems < A_nrows)) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "SpMM dimensions do not agree C[%d %d] != A[%d %d] B[%d %d], SPA[%lu]\n", C_nrows, C_ncols, A_nrows, A_ncols, B_nrows, B_ncols, s->nitems);
            std::exit(1); 
        }
        
        for(uint32_t j = start; j < end; j++) {
            for(uint32_t i = B_JA[j]; i < B_JA[j+1]; i++) {
                uint32_t r = B_IA[i]; Weight v = B_A[i]; 
                for(uint32_t k = 0; k < A_nrows; k++) {
                    uint32_t l = r*A_nrows + k;
                    s_A[k] += (A_A[l] * v);
                }
            }
            C_UDC->populate_spa(&s_A, b_A, off + j, idx_nnz, activation_function, tid);      
        }
    }
    else {
        Logging::print(Logging::LOG_LEVEL::ERROR, "+++++[%sx%s] multiplication not implemented\n", COMPRESSED_FORMATS[input_compression_type], COMPRESSED_FORMATS[layer_compression_type]);
        std::exit(Env::finalize());
    }
}

template<typename Weight>
inline void data_x_model_1_iter(std::shared_ptr<struct Compressed_Format<Weight>> A_SPMAT, 
                                std::shared_ptr<struct Compressed_Format<Weight>> B_SPMAT, 
                                std::shared_ptr<struct Compressed_Format<Weight>> C_SPMAT, 
                                std::shared_ptr<struct Data_Block<Weight>> s_spa,
                                const std::shared_ptr<struct Data_Block<Weight>> b_bias,
                                Weight(*noop_function)(Weight), Weight(*activation_function)(Weight),
                                const uint32_t nrows, const uint32_t ncols, 
                                const uint32_t start, const uint32_t end,
                                const uint32_t sub_start, const uint32_t sub_end,
                                struct Env::thread_struct& thread_st,
                                const bool last_layer,
                                COMPRESSED_FORMAT input_compression_type, COMPRESSED_FORMAT layer_compression_type,
                                const int32_t leader_tid,  const int32_t tid) {
    double start_time = 0;
    if((input_compression_type == COMPRESSED_FORMAT::_UDC_) and (layer_compression_type == COMPRESSED_FORMAT::_UDC_)) {
        start_time = Env::tic(); 
        C_SPMAT->reallocate(0, nrows, ncols, leader_tid, tid);
        thread_st.idx_nnz = start * nrows;
        Env::memory_allocation_time[tid] += Env::toc(start_time);
        
        start_time = Env::tic();
        pthread_barrier_wait(&Env::thread_barrier);
        if(not last_layer) { spmm_real(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, activation_function, start, end, sub_start, thread_st.idx_nnz, input_compression_type, layer_compression_type, tid); }
        else { spmm_real(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, noop_function, start, end, sub_start, thread_st.idx_nnz, input_compression_type, layer_compression_type, tid); }
        Env::spmm_real_time[tid] += Env::toc(start_time);
        
        //C_SPMAT->walk_dxm(false, leader_tid, tid);
    }
    else if((input_compression_type == COMPRESSED_FORMAT::_UDC_) and (layer_compression_type == COMPRESSED_FORMAT::_CSC_)) {
        start_time = Env::tic(); 
        C_SPMAT->reallocate(0, nrows, ncols, leader_tid, tid);
        thread_st.idx_nnz = start * nrows;
        Env::memory_allocation_time[tid] += Env::toc(start_time);
        
        start_time = Env::tic();
        pthread_barrier_wait(&Env::thread_barrier);
        if(not last_layer) { spmm_real(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, activation_function, start, end, sub_start, thread_st.idx_nnz, input_compression_type, layer_compression_type, tid); }
        else { spmm_real(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, noop_function, start, end, sub_start, thread_st.idx_nnz, input_compression_type, layer_compression_type, tid); }
        Env::spmm_real_time[tid] += Env::toc(start_time);
        
        //C_SPMAT->walk_dxm(false, leader_tid, tid);
    }
    else if(((input_compression_type == COMPRESSED_FORMAT::_CSC_) and (layer_compression_type == COMPRESSED_FORMAT::_CSC_)) or 
            ((input_compression_type == COMPRESSED_FORMAT::_CSC_) and (layer_compression_type == COMPRESSED_FORMAT::_UDC_))) {

        start_time = Env::tic(); 
        std::tie(thread_st.off_nnz, std::ignore, std::ignore) =  spmm_symb(A_SPMAT, B_SPMAT, s_spa, start, end, input_compression_type, layer_compression_type, tid);
        pthread_barrier_wait(&Env::thread_barrier);
        Env::spmm_symb_time[tid] += Env::toc(start_time);           
        pthread_barrier_wait(&Env::thread_barrier);
        
        start_time = Env::tic();
        uint64_t nnz = Env::adjust_nnz(leader_tid, tid);
        C_SPMAT->reallocate(nnz, nrows, ncols, leader_tid, tid);
        Env::memory_allocation_time[tid] += Env::toc(start_time);
        
        start_time = Env::tic();
        pthread_barrier_wait(&Env::thread_barrier);
        if(not last_layer) { spmm_real(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, activation_function, start, end, sub_start, thread_st.idx_nnz, input_compression_type, layer_compression_type, tid); }
        else { spmm_real(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, noop_function, start, end, sub_start, thread_st.idx_nnz, input_compression_type, layer_compression_type, tid); }
        pthread_barrier_wait(&Env::thread_barrier);
        
        Env::adjust_displacement(tid);
        C_SPMAT->adjust(leader_tid, tid);    
        Env::spmm_real_time[tid] += Env::toc(start_time);
        
        start_time = Env::tic();
        pthread_barrier_wait(&Env::thread_barrier);
        //A_SPMAT->repopulate(C_SPMAT, thread_st.dis_nnz, leader_tid, tid);
        A_SPMAT->repopulate(C_SPMAT, thread_st.dis_nnz, sub_start, sub_end, leader_tid, tid);
        Env::memory_allocation_time[tid] += Env::toc(start_time);
        
        //A_SPMAT->walk_dxm(false, leader_tid, tid);
   }
   else {
        Logging::print(Logging::LOG_LEVEL::ERROR, "[%sx%s] multiplication not implemented\n", COMPRESSED_FORMATS[input_compression_type], COMPRESSED_FORMATS[layer_compression_type]);
        std::exit(Env::finalize());
   }
}

template<typename Weight>
inline void data_x_data_1_iter(std::shared_ptr<struct Compressed_Format<Weight>> A_SPMAT, 
                               std::shared_ptr<struct Compressed_Format<Weight>> B_SPMAT, 
                               std::shared_ptr<struct Compressed_Format<Weight>> C_SPMAT, 
                               std::shared_ptr<struct Data_Block<Weight>> s_spa,
                               const std::shared_ptr<struct Data_Block<Weight>> b_bias,
                               Weight(*noop_function)(Weight), Weight(*activation_function)(Weight),
                               const uint32_t nrows, const uint32_t ncols,
                               const uint32_t start, const uint32_t end, const uint32_t off,
                               struct Env::thread_struct& thread_st,
                               const bool last_layer,
                               COMPRESSED_FORMAT input_compression_type, COMPRESSED_FORMAT layer_compression_type,
                               int32_t leader_tid,  const int32_t tid) {
    double start_time = 0;   
    if((input_compression_type == COMPRESSED_FORMAT::_UDC_) and (layer_compression_type == COMPRESSED_FORMAT::_UDC_)) {
        start_time = Env::tic();
        C_SPMAT->reallocate(0, nrows, ncols, tid, tid);
        thread_st.idx_nnz = 0;
        Env::memory_allocation_time[tid] += Env::toc(start_time);
        
        start_time = Env::tic();
        if(not last_layer) { spmm_real(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, activation_function, start, end, off, thread_st.idx_nnz, input_compression_type, layer_compression_type, tid); }
        else { spmm_real(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, noop_function, start, end, off, thread_st.idx_nnz, input_compression_type, layer_compression_type, tid); }
        Env::spmm_real_time[tid] += Env::toc(start_time);    
        
        //leader_tid = 0;
        //C_SPMAT->walk_dxd(false, leader_tid, tid);
    }
    else if((input_compression_type == COMPRESSED_FORMAT::_UDC_) and (layer_compression_type == COMPRESSED_FORMAT::_CSC_)) {
        start_time = Env::tic();
        C_SPMAT->reallocate(0, nrows, ncols, tid, tid);
        thread_st.idx_nnz = 0;
        Env::memory_allocation_time[tid] += Env::toc(start_time);
        
        start_time = Env::tic();
        if(not last_layer) { spmm_real(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, activation_function, start, end, off, thread_st.idx_nnz, input_compression_type, layer_compression_type, tid); }
        else { spmm_real(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, noop_function, start, end, off, thread_st.idx_nnz, input_compression_type, layer_compression_type, tid); }
        Env::spmm_real_time[tid] += Env::toc(start_time);  
        
        //leader_tid = 0;
        //C_SPMAT->walk_dxd(false, leader_tid, tid);
    }
    else if(((input_compression_type == COMPRESSED_FORMAT::_CSC_) and (layer_compression_type == COMPRESSED_FORMAT::_UDC_)) or
            ((input_compression_type == COMPRESSED_FORMAT::_CSC_) and (layer_compression_type == COMPRESSED_FORMAT::_CSC_)) or
            ((input_compression_type == COMPRESSED_FORMAT::_CSR_) and (layer_compression_type == COMPRESSED_FORMAT::_CSR_))) {
        
        start_time = Env::tic();
        std::tie(thread_st.off_nnz, std::ignore, std::ignore) =  spmm_symb(A_SPMAT, B_SPMAT, s_spa, start, end, input_compression_type, layer_compression_type, tid);
        Env::spmm_symb_time[tid] += Env::toc(start_time);      
        
        start_time = Env::tic();
        leader_tid = -1;
        uint64_t nnz = thread_st.off_nnz;
        C_SPMAT->reallocate(thread_st.off_nnz, nrows, ncols, leader_tid, tid);
        Env::memory_allocation_time[tid] += Env::toc(start_time);

        start_time = Env::tic();
        thread_st.idx_nnz = 0;
        if(not last_layer) { spmm_real(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, activation_function, start, end, off, thread_st.idx_nnz, input_compression_type, layer_compression_type, tid); }
        else { spmm_real(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, noop_function, start, end, off, thread_st.idx_nnz, input_compression_type, layer_compression_type, tid); }
        Env::adjust_displacement(tid);
        C_SPMAT->adjust(tid);
        Env::spmm_real_time[tid] += Env::toc(start_time);                              
    
        //leader_tid = 0;
        //C_SPMAT->walk_dxd(false, leader_tid, tid);
   }
    else {
        Logging::print(Logging::LOG_LEVEL::ERROR, "[%sx%s] multiplication not implemented\n", COMPRESSED_FORMATS[input_compression_type], COMPRESSED_FORMATS[layer_compression_type]);
        std::exit(Env::finalize());
   }
}


template<typename Weight>
inline void data_x_model_hybrid_1_iter(std::shared_ptr<struct Compressed_Format<Weight>> A_SPMAT, 
                                std::shared_ptr<struct Compressed_Format<Weight>> B_SPMAT, 
                                std::shared_ptr<struct Compressed_Format<Weight>> C_SPMAT, 
                                std::shared_ptr<struct Data_Block<Weight>> s_spa,
                                const std::shared_ptr<struct Data_Block<Weight>> b_bias,
                                Weight(*noop_function)(Weight), Weight(*activation_function)(Weight),
                                const uint32_t nrows, const uint32_t ncols,
                                const uint32_t start, const uint32_t end, const uint32_t off,
                                const std::deque<int32_t> my_threads,
                                struct Env::thread_struct& thread_st,
                                const bool last_layer,
                                COMPRESSED_FORMAT input_compression_type, COMPRESSED_FORMAT layer_compression_type,
                                const int32_t leader_tid,  const int32_t tid) {

    if(((input_compression_type == COMPRESSED_FORMAT::_CSC_) and (layer_compression_type == COMPRESSED_FORMAT::_CSC_)) or
       ((input_compression_type == COMPRESSED_FORMAT::_CSR_) and (layer_compression_type == COMPRESSED_FORMAT::_CSR_))) {
        double start_time = 0;

        if(tid ==leader_tid) start_time = Env::tic(); 
            std::tie(thread_st.off_nnz, std::ignore, std::ignore) =  spmm_symb(A_SPMAT, B_SPMAT, s_spa, start, end, input_compression_type, layer_compression_type, tid);
            pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
        if(tid ==leader_tid) Env::spmm_symb_time[tid] += Env::toc(start_time);   

        if(tid ==leader_tid) start_time = Env::tic();
        uint64_t nnz = Env::adjust_nnz(my_threads, leader_tid, tid);
        C_SPMAT->reallocate(nnz, nrows, ncols, leader_tid, tid);
        if(tid ==leader_tid) Env::memory_allocation_time[tid] += Env::toc(start_time);
        
        if(tid ==leader_tid) start_time = Env::tic();
        pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
        if(not last_layer) { spmm_real(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, activation_function, start, end, off, thread_st.idx_nnz, input_compression_type, layer_compression_type, tid); }
        else { spmm_real(A_SPMAT, B_SPMAT, C_SPMAT, s_spa, b_bias, noop_function, start, end, off, thread_st.idx_nnz, input_compression_type, layer_compression_type, tid); }
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
        Logging::print(Logging::LOG_LEVEL::ERROR, "[%sx%s] multiplication not implemented\n", COMPRESSED_FORMATS[input_compression_type], COMPRESSED_FORMATS[layer_compression_type]);
        std::exit(Env::finalize());
    }
}

template<typename Weight>
uint32_t infer(const std::shared_ptr<struct Compressed_Format<Weight>> C_SPMAT,
               const uint32_t C_start_row, 
               const std::vector<uint32_t> true_categories,
               const VALUE_TYPE category_type,
               const std::string classifier) {
                   
    std::vector<uint32_t> all_categories;
    
    uint64_t C_nnz;
    uint32_t C_nrows;
    uint32_t C_ncols;
    uint32_t* C_IA;
    uint32_t* C_JA;
    Weight*   C_A;   
    COMPRESSED_FORMAT compression_type = C_SPMAT->compression_type;  
    if(compression_type == COMPRESSED_FORMAT::_UDC_) {
        const std::shared_ptr<struct UDC<Weight>> C_UDC = std::static_pointer_cast<struct UDC<Weight>>(C_SPMAT);

        C_nrows = C_UDC->nrows;
        C_ncols = C_UDC->ncols;
        C_A   = C_UDC->A_blk->ptr;
        
        all_categories.resize(C_nrows);
        if(category_type == VALUE_TYPE::_NONZERO_INSTANCES_ONLY_) {
            for(uint64_t k = 0; k < C_nrows * C_ncols; k++) {
                if(C_A[k]) {
                    uint32_t row = k % C_nrows;
                    all_categories[row] = 1;
                }
            }
        }
        else if(category_type == VALUE_TYPE::_INSTANCE_AND_VALUE_PAIRS_) {
            if(classifier == "sigmoid") {
                for(uint64_t k = 0; k < C_nrows * C_ncols; k++) {
                    if(C_A[k]) {
                        uint32_t row = k % C_nrows;
                        all_categories[row] = sigmoid(C_A[k]) < 0.5 ? 0 : 1;;
                    }
                }
            }
            else if(classifier == "softmax") {  
                std::vector<Weight> values(C_nrows, std::numeric_limits<Weight>::min());
                for(uint64_t k = 0; k < C_nrows * C_ncols; k++) {
                    if(C_A[k]) {
                        uint32_t row = k % C_nrows;
                        uint32_t col = k / C_nrows;
                        if(values[row]<C_A[k]) {
                            values[row] = C_A[k];
                            all_categories[row]=col;    
                        }
                    }
                }
            }
        }
    }
    else if(compression_type == COMPRESSED_FORMAT::_CSC_) {
        const std::shared_ptr<struct CSC<Weight>> C_CSC = std::static_pointer_cast<struct CSC<Weight>>(C_SPMAT);
        C_nnz   = C_CSC->nnz;
        C_nrows = C_CSC->nrows;
        C_ncols = C_CSC->ncols;
        C_IA   = C_CSC->IA_blk->ptr;
        C_JA   = C_CSC->JA_blk->ptr;
        C_A   = C_CSC->A_blk->ptr;
        
        all_categories.resize(C_nrows);
        if(category_type == VALUE_TYPE::_NONZERO_INSTANCES_ONLY_) {
            for(uint32_t j = 0; j < C_ncols; j++) {
                for(uint32_t i = C_JA[j]; i < C_JA[j+1]; i++) {
                    all_categories[C_IA[i]] = 1;
                }
            }
        }
        else if(category_type == VALUE_TYPE::_INSTANCE_AND_VALUE_PAIRS_) {
            if(classifier == "sigmoid") {
                for(uint32_t j = 0; j < C_ncols; j++) {
                    for(uint32_t i = C_JA[j]; i < C_JA[j+1]; i++) {
                        all_categories[C_IA[i]]=sigmoid(C_A[i]) < 0.5 ? 0 : 1;
                    }
                }
            }
            else if(classifier == "softmax") {    
                std::vector<Weight> values(C_nrows, std::numeric_limits<Weight>::min());
                for(uint32_t j = 0; j < C_ncols; j++) {
                    for(uint32_t i = C_JA[j]; i < C_JA[j+1]; i++) {
                        //if(values[C_IA[i]]) {
                            if(values[C_IA[i]]<C_A[i]) {
                                values[C_IA[i]] = C_A[i];
                                all_categories[C_IA[i]]=j;    
                            }
                            /*
                        }
                        else {
                            values[C_IA[i]] = C_A[i];
                            all_categories[C_IA[i]]=j;
                        }
                        */
                    }
                }
            }
        }
    }
    else if(compression_type == COMPRESSED_FORMAT::_CSR_) {
        const std::shared_ptr<struct CSR<Weight>> C_CSR = std::static_pointer_cast<struct CSR<Weight>>(C_SPMAT);
        C_nnz   = C_CSR->nnz;
        C_nrows = C_CSR->nrows;
        C_ncols = C_CSR->ncols;
        C_IA   = C_CSR->IA_blk->ptr;
        C_JA   = C_CSR->JA_blk->ptr;
        C_A    = C_CSR->A_blk->ptr;
        
        all_categories.resize(C_nrows);
        if(category_type == VALUE_TYPE::_NONZERO_INSTANCES_ONLY_) {
            for(uint32_t i = 0; i < C_nrows; i++) all_categories[i]= C_IA[i+1]-C_IA[i] ? 1 : 0;
        }
        else if(category_type == VALUE_TYPE::_INSTANCE_AND_VALUE_PAIRS_) {
            if(classifier == "sigmoid") {
                for(uint32_t i = 0; i < C_nrows; i++) {
                    for(uint32_t j=C_IA[i]; j < C_IA[i+1]; j++) {
                        all_categories[i]=sigmoid(C_A[j]) < 0.5 ? 0 : 1;
                    }
                }
            }
            else if(classifier == "softmax") {    
                for(uint32_t i = 0; i < C_nrows; i++) {
                    int index = 0;
                    Weight value = 0;
                    for(uint32_t j=C_IA[i]; j < C_IA[i+1]; j++) {
                        if(C_A[j]>value) { value = C_A[j]; index = C_JA[j]; }
                    }
                    all_categories[i]= index;
                }
            }
        }
    }
    else {
        Logging::print(Logging::LOG_LEVEL::ERROR, "%s compression not implemented\n", COMPRESSED_FORMATS[compression_type]);
        std::exit(Env::finalize());
    }
    
    int count = 0;
    if(category_type == VALUE_TYPE::_NONZERO_INSTANCES_ONLY_) {
        for(uint32_t i = 0; i < C_nrows; i++) count += (true_categories[C_start_row + i] and true_categories[C_start_row + i] == all_categories[i]) ? 1 : 0;
    }
    else if(category_type == VALUE_TYPE::_INSTANCE_AND_VALUE_PAIRS_) {
        for(uint32_t i = 0; i < C_nrows; i++) count += (true_categories[C_start_row + i] == all_categories[i]) ? 1 : 0;
    }
    /*
    for(uint32_t i = 0; i < C_nrows; i++) {
        if(true_categories[C_start_row + i] != all_categories[i]) {
            printf("i=%d groundtruth=%d != inferred=%d\n", i, true_categories[C_start_row + i], all_categories[i]);
        }
    }
    */
    return count;
}

template<typename Weight>
inline void data_x_model_validate_prediction(const std::shared_ptr<struct Compressed_Format<Weight>> C_SPMAT,
                                             const uint32_t C_start_row,
                                             const std::vector<uint32_t> true_categories,
                                             const uint32_t predicted_nistances,
                                             const VALUE_TYPE category_type,
                                             const std::string classifier,
                                             const int32_t leader_tid, 
                                             const int32_t tid) {                  
    if(tid == leader_tid) {
        uint32_t count = infer(C_SPMAT, C_start_row, true_categories, category_type, classifier);
        uint32_t counts = 0;
        MPI_Allreduce(&count, &counts, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
        
        bool passed = (counts == predicted_nistances);
        if(passed) { Logging::print(Logging::LOG_LEVEL::INFO, "Challenge PASSED.\n"); }
        else { Logging::print(Logging::LOG_LEVEL::ERROR, "Challenge FAILED.\n"); }
        Logging::print(Logging::LOG_LEVEL::INFO, "Inference accuracy=%f [%d|%d]\n", (double) counts/predicted_nistances, counts, predicted_nistances-counts);
    }
    pthread_barrier_wait(&Env::thread_barrier);
    Env::barrier();
}


template<typename Weight>
inline void data_x_data_validate_prediction(const std::shared_ptr<struct Compressed_Format<Weight>> C_SPMAT,
                                const uint32_t C_start_row,
                                const std::vector<uint32_t> true_categories,
                                const uint32_t predicted_nistances,
                                const VALUE_TYPE category_type,
                                const std::string classifier,
                                const int32_t leader_tid, 
                                const int32_t tid) {
                                    
    uint32_t count = infer(C_SPMAT, C_start_row, true_categories, category_type, classifier);
        
    Env::counters[tid].checkcount = count;
    pthread_barrier_wait(&Env::thread_barrier);
    if(tid == leader_tid) {
        uint32_t counts = 0;
        for(auto counter: Env::counters) { counts += counter.checkcount; }
        uint32_t countss = 0;
        MPI_Allreduce(&counts, &countss, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
        bool passed=(countss == predicted_nistances);
        if(passed) { Logging::print(Logging::LOG_LEVEL::INFO, "Challenge PASSED.\n"); }
        else { Logging::print(Logging::LOG_LEVEL::ERROR, "Challenge FAILED.\n"); }
        Logging::print(Logging::LOG_LEVEL::INFO, "Inference accuracy=%f [%d|%d]\n", (double) countss/predicted_nistances, countss, predicted_nistances-countss);
    }
    pthread_barrier_wait(&Env::thread_barrier);
    Env::barrier();
}


template<typename Weight>
inline void manager_x_worker_validate_prediction(std::vector<std::vector<struct Tile<Weight>>> tiles,
                                const std::vector<uint32_t> true_categories,
                                const uint32_t predicted_nistances,
                                const VALUE_TYPE category_type,
                                const std::string classifier,
                                const int32_t leader_tid, 
                                const int32_t tid) {
    pthread_barrier_wait(&Env::thread_barrier);                                        
    if(tid == leader_tid) {
        int count = 0;
        for(uint32_t rowgroup:  Env::processed_rowgroups) {
            struct Tile<Weight>& C_tile = tiles[rowgroup][0];
            uint32_t C_start_row = C_tile.start_row;;
            std::shared_ptr<struct Compressed_Format<Weight>> C_SPMAT = C_tile.spmat;
            count += infer(C_SPMAT, C_start_row, true_categories, category_type, classifier);    
        }
        
        uint32_t counts = 0;
        MPI_Allreduce(&count, &counts, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
        bool passed = (counts == predicted_nistances);
        if(passed) {
            Logging::print(Logging::LOG_LEVEL::INFO, "Challenge PASSED.\n");
        }
        else {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Challenge FAILED.\n");
        }
        Logging::print(Logging::LOG_LEVEL::INFO, "Inference accuracy=%f [%d|%d]\n", (double) counts/predicted_nistances, counts, predicted_nistances-counts);
    }
    pthread_barrier_wait(&Env::thread_barrier);
    Env::barrier();
}

template<typename Weight>
inline void work_x_stealing_validate_prediction(std::vector<std::vector<struct Tile<Weight>>> tiles,
                                const std::vector<uint32_t> true_categories,
                                const uint32_t predicted_nistances,
                                const VALUE_TYPE category_type,
                                const std::string classifier,
                                const int32_t leader_tid, 
                                const int32_t tid) {
    pthread_barrier_wait(&Env::thread_barrier);     
    if(tid == leader_tid) {
        for(auto p: Env::processed_rowgroups_per_thread) {
            Env::processed_rowgroups.insert(Env::processed_rowgroups.end(), p.begin(), p.end());

        }
    }
    pthread_barrier_wait(&Env::thread_barrier);   
    manager_x_worker_validate_prediction(tiles, true_categories, predicted_nistances, category_type, classifier, leader_tid, tid);
}

#endif