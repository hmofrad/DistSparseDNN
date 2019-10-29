/*
 * spops.cpp: Sparse Matrix operations implementation
 * Sparse Matrix - Sparse Matrix (SpMM)
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com
 */
 
#ifndef SPOPS_H
#define SPOPS_H

#include "env.hpp"
#include "spmat.hpp"


//inline void spmm_sym();
//inline void spmm();

 
template<typename Weight>
inline std::tuple<uint64_t, uint32_t, uint32_t> spmm_sym(std::shared_ptr<struct Compressed_Format<Weight>> A,
                                                         std::shared_ptr<struct Compressed_Format<Weight>> B,
                                                         std::vector<Weight> s) {

    uint64_t nnzmax = 0;
    uint32_t nrows = 0;
    uint32_t ncols = 0;
    if((A->compression_type == COMPRESSED_FORMAT::_CSR_) and (B->compression_type == COMPRESSED_FORMAT::_CSR_)) {
        const std::shared_ptr<struct CSR<Weight>> A_CSR = std::static_pointer_cast<struct CSR<Weight>>(A);
        const uint64_t A_nnz   = A_CSR->nnz;
        const uint32_t A_nrows = A_CSR->nrows;
        const uint32_t A_ncols = A_CSR->ncols;
        const uint32_t* A_IA   = A_CSR->IA_blk->ptr;
        const uint32_t* A_JA   = A_CSR->JA_blk->ptr;
        const Weight*    A_A   = A_CSR->A_blk->ptr;
        
        const std::shared_ptr<struct CSR<Weight>> B_CSR = std::static_pointer_cast<struct CSR<Weight>>(B);
        const uint64_t B_nnz   = B_CSR->nnz;
        const uint32_t B_nrows = B_CSR->nrows;
        const uint32_t B_ncols = B_CSR->ncols;
        const uint32_t* B_IA   = B_CSR->IA_blk->ptr;
        const uint32_t* B_JA   = B_CSR->JA_blk->ptr;
        const Weight*    B_A   = B_CSR->A_blk->ptr;
        
        if(A_ncols != B_nrows) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "SpMM dimensions do not agree A[%d %d] B[%d %d]\n", A_nrows, A_ncols, B_nrows, B_ncols);
            std::exit(Env::finalize()); 
        }
                
        nrows = A_nrows;
        ncols = B_ncols;
        
        Logging::print(Logging::LOG_LEVEL::ERROR, "SpMM _CSR_ not implemented.\n");
        std::exit(Env::finalize()); 
    }
    else if((A->compression_type == COMPRESSED_FORMAT::_CSC_) and (B->compression_type == COMPRESSED_FORMAT::_CSC_)) {
        const std::shared_ptr<struct CSC<Weight>> A_CSC = std::static_pointer_cast<struct CSC<Weight>>(A);
        const uint64_t A_nnz   = A_CSC->nnz;
        const uint32_t A_nrows = A_CSC->nrows;
        const uint32_t A_ncols = A_CSC->ncols;
        const uint32_t* A_IA   = A_CSC->IA_blk->ptr;
        const uint32_t* A_JA   = A_CSC->JA_blk->ptr;
        const Weight*    A_A   = A_CSC->A_blk->ptr;
        
        const std::shared_ptr<struct CSC<Weight>> B_CSC = std::static_pointer_cast<struct CSC<Weight>>(B);
        const uint64_t B_nnz   = B_CSC->nnz;
        const uint32_t B_nrows = B_CSC->nrows;
        const uint32_t B_ncols = B_CSC->ncols;
        const uint32_t* B_IA   = B_CSC->IA_blk->ptr;
        const uint32_t* B_JA   = B_CSC->JA_blk->ptr;
        const Weight*    B_A   = B_CSC->A_blk->ptr;
        
        if(A_ncols != B_nrows) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "SpMM dimensions do not agree A[%d %d] B[%d %d]\n", A_nrows, A_ncols, B_nrows, B_ncols);
            std::exit(Env::finalize()); 
        }
        nrows = A_nrows;
        ncols = B_ncols;
       //Logging::print(Logging::LOG_LEVEL::ERROR, "SpMM dimensions do not agree A[%d %d] B[%d %d]\n", A_nrows, A_ncols, B_nrows, B_ncols);
       //std::vector<bool> ss(A_nrows);
       //uint64_t ii = 0;
       
       /*
        for(uint32_t j = 0; j < B_ncols; j++) {
            for(uint32_t k = B_JA[j]; k < B_JA[j+1]; k++) {
                uint32_t l = B_IA[k];
                for(uint32_t m = A_JA[l]; m < A_JA[l+1]; m++) {
                    ii++;
                    ss[A_IA[m]] = 1;
                }
            }
            for(uint32_t i = 0; i < A_nrows; i++) {
                if(ss[i]){
                    nnzmax++;
                    ss[i] = 0;
                }
            }
        }
        */
        
        for(uint32_t j = 0; j < B_ncols; j++) {
            //printf("%d nnzmax=%lu\n", j, nnzmax);
            for(uint32_t k = B_JA[j]; k < B_JA[j+1]; k++) {
                uint32_t l = B_IA[k];
                for(uint32_t m = A_JA[l]; m < A_JA[l+1]; m++) {
                    //ii++;
                    s[A_IA[m]] = 1;
                }
            }
            for(uint32_t i = 0; i < A_nrows; i++) {
                if(s[i]){
                    nnzmax++;
                    s[i] = 0;
                }
            }
        }
        
        //A_CSC->walk(); 
        //B_CSC->walk();
    }
 
    //printf("nnzmax=%lu\n", nnzmax);
    return std::make_tuple(nnzmax, nrows, ncols);
}

template<typename Weight>
inline void spmm(std::shared_ptr<struct Compressed_Format<Weight>> A,
                 std::shared_ptr<struct Compressed_Format<Weight>> B,
                 std::shared_ptr<struct Compressed_Format<Weight>> C,
                 std::vector<Weight> s,
                 std::vector<Weight> b) {
                //printf("SPMM %d\n", Env::rank);
                  
    if((A->compression_type == COMPRESSED_FORMAT::_CSC_) and 
       (B->compression_type == COMPRESSED_FORMAT::_CSC_) and
       (C->compression_type == COMPRESSED_FORMAT::_CSC_)) {  
       
        const std::shared_ptr<struct CSC<Weight>> A_CSC = std::static_pointer_cast<struct CSC<Weight>>(A);
        const uint64_t A_nnz   = A_CSC->nnz;
        const uint32_t A_nrows = A_CSC->nrows;
        const uint32_t A_ncols = A_CSC->ncols;
        const uint32_t* A_IA   = A_CSC->IA_blk->ptr;
        const uint32_t* A_JA   = A_CSC->JA_blk->ptr;
        const Weight*    A_A   = A_CSC->A_blk->ptr;
        
        const std::shared_ptr<struct CSC<Weight>> B_CSC = std::static_pointer_cast<struct CSC<Weight>>(B);
        const uint64_t B_nnz   = B_CSC->nnz;
        const uint32_t B_nrows = B_CSC->nrows;
        const uint32_t B_ncols = B_CSC->ncols;
        const uint32_t* B_IA   = B_CSC->IA_blk->ptr;
        const uint32_t* B_JA   = B_CSC->JA_blk->ptr;
        const Weight*    B_A   = B_CSC->A_blk->ptr;
        
        const std::shared_ptr<struct CSC<Weight>> C_CSC = std::static_pointer_cast<struct CSC<Weight>>(C);
        const uint64_t C_nnz   = C_CSC->nnz;
        const uint32_t C_nrows = C_CSC->nrows;
        const uint32_t C_ncols = C_CSC->ncols;
        const uint32_t* C_IA   = C_CSC->IA_blk->ptr;
        const uint32_t* C_JA   = C_CSC->JA_blk->ptr;
        const Weight*    C_A   = C_CSC->A_blk->ptr;
        
        if(A_ncols != B_nrows) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "SpMM dimensions do not agree C[%d %d] != A[%d %d] B[%d %d]\n", C_nrows, C_ncols, A_nrows, A_ncols, B_nrows, B_ncols);
            std::exit(Env::finalize()); 
        }

       // printf("1.ii=%lu %lu\n", ii, nnzmax);
      //  printf("XXXXXXXXXXXXXXXXXXXXXXXXXX\n");
        
        
        
        
        //printf("0.nnz_i=%lu \n", C_CSC->nnz_i);
        for(uint32_t j = 0; j < B_ncols; j++) {
            //printf("%d %lu/%lu\n", j, C_CSC->nnz_i, C_CSC->nnz);
            for(uint32_t k = B_JA[j]; k < B_JA[j+1]; k++) {
                uint32_t l = B_IA[k];
                for(uint32_t m = A_JA[l]; m < A_JA[l+1]; m++) {
                    s[A_IA[m]] += (B_A[k] * A_A[m]);
                    //s[A_IA[m]] = 1;
                    //ii++;
                }
            }
            C_CSC->populate_spa(s, b, j);
        }
        //printf("1.nnz_i=%lu \n", C_CSC->nnz_i);
        //printf("%lu %lu\n", C_CSC->nnz, C_CSC->nnz_i);
        C_CSC->adjust();
        //if(C_CSC->nnz != C_CSC->nnz_i) {
            
        //}
        
       //C_CSC->walk(); 
     //A_CSC->repopulate(C_CSC);
     
    /*
    if(C_ncols != b_nitems) {
        fprintf(stderr, "Error: SpMV_EW dimensions do not agree [%d != %d]\n", C_ncols, b_nitems);
        exit(1);
    }
  */     
       

       
       
   }
   else {
        Logging::print(Logging::LOG_LEVEL::ERROR, "SpMM not implemented.\n");
        std::exit(Env::finalize()); 
   }
    
}


template<typename Weight>
inline void validate_prediction(std::shared_ptr<struct Compressed_Format<Weight>> A,
                                      std::vector<uint32_t> trueCategories) {
    const std::shared_ptr<struct CSC<Weight>> A_CSC = std::static_pointer_cast<struct CSC<Weight>>(A);
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
    

    
//printf("%d %d %lu %lu\n", A_nrows, A_ncols, predictedCategories.size(), trueCategories.size());
    bool tf = true;
  //  if(trueCategories.size() == predictedCategories.size()) {        
        uint32_t j = 0;
        for(uint32_t i = 0; i < A_nrows; i++) {
            //printf("%d %d %d\n", i, trueCategories[i], allCategories[i]);
            if(trueCategories[i] != allCategories[i]) {
                tf = false;
                break;
                //if(trueCategories[j] != (i - 1)) {
                    //tf = false;
                   // break;
                ///}
             //   j++;
            }
            //if(trueCategories[i] != allCategories[i]) {
              //  tf = false;
                //break;
            //}
        }
    //} 
    //else {
      //  tf = false;
    //}
    
    if(tf) {
        printf("INFO: Challenge PASSED\n");
    }
    else {
        printf("INFO: Challenge FAILED\n");
    }
    
    
}



#endif
/*

template<typename Weight>
inline void SpMM(struct CSC<Weight> *A_CSC, struct CSC<Weight> *B_CSC, struct CSC<Weight> *C_CSC,
                  struct DenseVec<Weight> *s, struct DenseVec<Weight> *b, int tid) {  
    uint32_t *A_JA = A_CSC->JA;
    uint32_t *A_IA = A_CSC->IA;      
    Weight   *A_A  = A_CSC->A;
    uint32_t A_nrows = A_CSC->nrows;  
    uint32_t A_ncols = A_CSC->ncols;    
    
    uint32_t *B_JA = B_CSC->JA;
    uint32_t *B_IA = B_CSC->IA;      
    Weight   *B_A  = B_CSC->A;
    uint32_t B_nrows = B_CSC->nrows;
    uint32_t B_ncols = B_CSC->ncols;

    uint32_t C_nrows = C_CSC->nrows;
    uint32_t C_ncols = C_CSC->ncols;;       
                 
    uint32_t b_nitems = b->nitems;
    
    if((A_ncols != B_nrows) or (A_nrows != C_nrows) or (B_ncols != C_ncols)) {
        fprintf(stderr, "Error: SpMM dimensions do not agree C[%d %d] != A[%d %d] B[%d %d]\n", C_nrows, C_ncols, A_nrows, A_ncols, B_nrows, B_ncols);
        exit(1);
    }
    
    if(C_ncols != b_nitems) {
        fprintf(stderr, "Error: SpMV_EW dimensions do not agree [%d != %d]\n", C_ncols, b_nitems);
        exit(1);
    }

    uint32_t start = Env::start_col[tid];
    uint32_t end = Env::end_col[tid];
    auto *s_A = s->A;

    for(uint32_t j = start; j < end; j++) {
        for(uint32_t k = B_JA[j]; k < B_JA[j+1]; k++) {
            uint32_t l = B_IA[k];
            for(uint32_t m = A_JA[l]; m < A_JA[l+1]; m++) {
                s_A[A_IA[m]] += B_A[k] * A_A[m];
            }
        }
        C_CSC->spapopulate_t(b, s, j, tid);
    }
    #pragma omp barrier
    C_CSC->postpopulate_t(tid);
    A_CSC->repopulate(C_CSC, tid);
    #pragma omp barrier
}
*/