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

enum COMPRESSED_FORMAT {_CSR_, _DCSR_, _TCSR_, _CSC_, _DCSC_, _TCSC_};
const char* COMPRESSED_FORMATS[] = {"_CSR_", "_DCSR_", "_TCSR_", "_CSC_", "_DCSC_", "_TCSC_"};

template<typename Weight>
struct Compressed_Format {
    public:
        Compressed_Format() {};
        virtual ~Compressed_Format() {};
        virtual void populate(std::vector<struct Triple<Weight>>& triples, uint32_t tile_height, uint32_t tile_width) {};
        virtual void walk() {};
        
        uint64_t nnz;
        uint32_t nrows;
        uint32_t ncols;
        
        std::shared_ptr<struct Data_Block<uint32_t>> IA_blk;
        std::shared_ptr<struct Data_Block<uint32_t>> JA_blk;
        std::shared_ptr<struct Data_Block<Weight>>  A_blk;
};


template<typename Weight>
struct CSR : public Compressed_Format<Weight> {
    public:
        CSR(uint64_t nnz_, uint32_t nrows_, uint32_t ncols_);//CSR::nnz = 0; CSR::nrows = 0; CSR::ncols = 0;};
        ~CSR(){};
        
        void populate(std::vector<struct Triple<Weight>>& triples, uint32_t tile_height, uint32_t tile_width);
        void walk();
};


template<typename Weight>
CSR<Weight>::CSR(uint64_t nnz_, uint32_t nrows_, uint32_t ncols_) {
    CSR::nnz = nnz_;
    CSR::nrows = nrows_; 
    CSR::ncols = ncols_;
    
    CSR::IA_blk = std::move(std::make_shared<struct Data_Block<uint32_t>>(CSR::nrows + 1));
    CSR::JA_blk = std::move(std::make_shared<struct Data_Block<uint32_t>>(CSR::nnz));
    CSR::A_blk = std::move(std::make_shared<struct Data_Block<Weight>>(CSR::nnz));
}

template<typename Weight>
void CSR<Weight>::populate(std::vector<struct Triple<Weight>>& triples, uint32_t tile_height, uint32_t tile_width) {
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
}

template<typename Weight>
void CSR<Weight>::walk() {    
    uint32_t* IA = CSR::IA_blk->ptr;
    uint32_t* JA = CSR::JA_blk->ptr;
    Weight* A = CSR::A_blk->ptr;
    
    double sum = 0;
    uint64_t count = 0;
    for(uint32_t i = 0; i < CSR::nrows; i++) {
        //std::cout << "i=" << i << ": " << IA[i + 1] - IA[i] << std::endl;
        for(uint32_t j = IA[i]; j < IA[i + 1]; j++) {
            (void) JA[j];
            (void) A[j];
            sum += A[j];
            count++;
            //std::cout << "    i=" << i << ",j=" << JA[j] <<  ",value=" << A[j] << std::endl;
        }
    }
    
    if(count != CSR::nnz) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Compression failed\n");
        std::exit(Env::finalize());     
    }
    
    //std::cout << "Checksum=" << sum << ",Count=" << count << std::endl;
}


/*
template<typename Weight>
struct CSC : public Compressed_Format<Weight> {
    public:
        CSC() {CSC::nnz = 0; CSC::nrows = 0; CSC::ncols = 0;};
        ~CSC(){};
        
        void populate(std::vector<struct Triple<Weight>>& triples, uint32_t tile_height, uint32_t tile_width);
};


template<typename Weight>
void CSC<Weight>::populate(std::vector<struct Triple<Weight>>& triples, uint32_t tile_height, uint32_t tile_width) {
    
}
*/


/*

template<typename Weight>
struct CSC {
    public:
        CSC() { nrows = 0, ncols = 0; nnz = 0;  nbytes = 0; idx = 0; JA = nullptr; IA = nullptr; A = nullptr; }
        CSC(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_, bool page_aligned_ = true);
        CSC(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_, std::vector<struct Triple<Weight>> &triples, bool page_aligned_ = true);
        ~CSC();
        inline void initialize(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_);
        inline void reinitialize(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_);
        inline void prepopulate(std::vector<struct Triple<Weight>> &triples);
        inline void populate(std::vector<struct Triple<Weight>> &triples);
        inline void postpopulate_t(int tid);
        inline void repopulate(struct CSC<Weight> *other_csc);
        inline void repopulate(struct CSC<Weight> *other_csc, int tid);
        inline void spapopulate(struct DenseVec<Weight> *x_vector, struct DenseVec<Weight> *spa_vector, uint32_t col_idx);
        inline void spapopulate(struct DenseVec<Weight> *spa_vector, uint32_t col_idx);
        inline void spapopulate_t(struct DenseVec<Weight> *x_vector, struct DenseVec<Weight> *spa_vector, uint32_t col_idx, int tid);
        inline void walk();
        inline uint64_t numnonzeros() const { return(nnz); };
        inline uint32_t numrows()   const { return(nrows); };
        inline uint32_t numcols()   const { return(ncols); };
        inline uint64_t size()        const { return(nbytes); };
        inline void clear();
        uint32_t nrows;
        uint32_t ncols;
        uint64_t nnz;
        uint64_t nnzmax;
        uint64_t nbytes;
        uint64_t idx;
        uint32_t *JA; // Cols
        uint32_t *IA; // Rows
        Weight   *A;  // Vals
        struct Data_Block<uint32_t> *IA_blk;
        struct Data_Block<uint32_t> *JA_blk;
        struct Data_Block<Weight>  *A_blk;
        bool page_aligned;
};

template<typename Weight>
CSC<Weight>::CSC(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_, bool page_aligned_) {
    nrows = nrows_;
    ncols = ncols_;
    nnz   = nnz_;
    nnzmax   = nnz_;
    page_aligned = page_aligned_;
    JA = nullptr;
    IA = nullptr;
    A  = nullptr;
    if(nrows and ncols and nnz) {
        JA_blk = new Data_Block<uint32_t>(&JA, (ncols + 1), (ncols + 1) * sizeof(uint32_t), page_aligned);
        IA_blk = new Data_Block<uint32_t>(&IA, nnz, nnz * sizeof(uint32_t), page_aligned);
        A_blk  = new Data_Block<Weight>(&A,  nnz, nnz * sizeof(Weight), page_aligned);
        nbytes = IA_blk->nbytes + JA_blk->nbytes + A_blk->nbytes;
        JA[0] = 0;
    }
    idx = 0;
}

template<typename Weight>
CSC<Weight>::CSC(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_, std::vector<struct Triple<Weight>> &triples, bool page_aligned_) {
    nrows = nrows_;
    ncols = ncols_;
    nnz   = nnz_;
    nnzmax   = nnz_;
    nbytes = 0;
    page_aligned = page_aligned_;
    JA = nullptr;
    IA = nullptr;
    A  = nullptr;
    prepopulate(triples);
    if(nrows and ncols and nnz) {
        JA_blk = new Data_Block<uint32_t>(&JA, (ncols + 1), (ncols + 1) * sizeof(uint32_t), page_aligned);
        IA_blk = new Data_Block<uint32_t>(&IA, nnz, nnz * sizeof(uint32_t), page_aligned);
        A_blk  = new Data_Block<Weight>(&A,  nnz, nnz * sizeof(Weight), page_aligned);
        nbytes = IA_blk->nbytes + JA_blk->nbytes  + A_blk->nbytes;
    }
    JA[0] = 0;
    idx = 0;
    populate(triples);
}

template<typename Weight>
CSC<Weight>::~CSC(){
    delete JA_blk;
    JA = nullptr;
    delete IA_blk;
    IA = nullptr;
    delete  A_blk;
    A  = nullptr;
}

template<typename Weight>
inline void CSC<Weight>::initialize(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_) {    
    if(nnz) {
        nrows = nrows_;
        ncols = ncols_;
        nnz = nnz_;
        nnzmax = nnz_;
        JA_blk->reallocate(&JA, (ncols + 1), ((ncols + 1) * sizeof(uint32_t)));
        IA_blk->reallocate(&IA, nnz, (nnz * sizeof(Weight)));
        A_blk->reallocate(&A, nnz, (nnz * sizeof(Weight)));
        nbytes = JA_blk->nbytes + IA_blk->nbytes + A_blk->nbytes;
        JA[0] = 0;
        idx = 0;
        clear();
    }
    else {
        nrows = nrows_;
        ncols = ncols_;
        nnz = nnz_;
        nnzmax = nnz_;
        JA_blk = new Data_Block<uint32_t>(&JA, (ncols + 1), (ncols + 1) * sizeof(uint32_t), page_aligned);
        IA_blk = new Data_Block<uint32_t>(&IA, nnz, nnz * sizeof(uint32_t), page_aligned);
        A_blk  = new Data_Block<Weight>(&A,  nnz, nnz * sizeof(Weight), page_aligned);
        nbytes = IA_blk->nbytes + JA_blk->nbytes + A_blk->nbytes;
        JA[0] = 0;
        idx = 0;
    }
}

template<typename Weight>
inline void CSC<Weight>::prepopulate(std::vector<struct Triple<Weight>> &triples) {
    uint64_t triples_size = triples.size();
    if(triples_size) {
        ColSort<Weight> f_col;
        std::sort(triples.begin(), triples.end(), f_col);
        for(uint64_t i = 1; i < triples_size; i++) {
            if(triples[i-1].col == triples[i].col) {
                if(triples[i-1].row == triples[i].row) {
                    triples[i].weight += triples[i-1].weight;
                    triples.erase(triples.begin()+i-1);
                }
            }
        }
        nnz = triples.size();
    }
}


template<typename Weight>
inline void CSC<Weight>::populate(std::vector<struct Triple<Weight>> &triples) {
    if(ncols and nnz and triples.size()) {
        uint32_t i = 0;
        uint32_t j = 1;        
        JA[0] = 0;
        for(auto &triple : triples) {
            while((j - 1) != triple.col) {
                j++;
                JA[j] = JA[j - 1];
            }                  
            JA[j]++;
            IA[i] = triple.row;
            A[i] = triple.weight;
            i++;
        }
        while((j + 1) <= ncols) {
            j++;
            JA[j] = JA[j - 1];
        }
    }
}

template<typename Weight>
inline void CSC<Weight>::spapopulate(struct DenseVec<Weight> *x_vector, struct DenseVec<Weight> *spa_vector, uint32_t col_idx) {
    Weight YMIN = 0;
    Weight YMAX = 32;
    Weight   *x_A = x_vector->A;
    Weight   *spa_A = spa_vector->A;
    Weight value = 0;
    JA[col_idx+1] += JA[col_idx];
    for(uint32_t i = 0; i < nrows; i++) {
        if(spa_A[i]) {
            JA[col_idx+1]++;
            IA[idx] = i;
            spa_A[i] += x_A[col_idx];
            if(spa_A[i] < YMIN) {
                A[idx] = YMIN;
            }
            else if(spa_A[i] > YMAX) {
                A[idx] = YMAX;
            }
            else {
                A[idx] = spa_A[i];
            }
            idx++;
            spa_A[i] = 0;
        }
    }
}

template<typename Weight>
inline void CSC<Weight>::spapopulate(struct DenseVec<Weight> *spa_vector, uint32_t col_idx) {
    Weight YMIN = 0;
    Weight YMAX = 32;
    Weight   *spa_A = spa_vector->A;
    Weight value = 0;
    JA[col_idx+1] += JA[col_idx];
    for(uint32_t i = 0; i < nrows; i++) {
        if(spa_A[i]) {
            JA[col_idx+1]++;
            IA[idx] = i;
            A[idx] = spa_A[i];
            idx++;
            spa_A[i] = 0;
        }
    }
}

template<typename Weight>
inline void CSC<Weight>::repopulate(struct CSC<Weight> *other_csc){
    uint32_t o_ncols = other_csc->numcols();
    uint32_t o_nnz = other_csc->numnonzeros();
    uint32_t *o_IA = other_csc->IA;
    uint32_t *o_JA = other_csc->JA;
    Weight   *o_A  = other_csc->A;
    if(ncols != o_ncols) {
        fprintf(stderr, "Error: Cannot repopulate CSC\n");
        exit(1);
    }
    if(nnz < o_nnz) {
        IA_blk->reallocate(&IA, o_nnz, (o_nnz * sizeof(uint32_t)));
        A_blk->reallocate(&A, o_nnz, (o_nnz * sizeof(Weight)));
    }
    clear();
    idx = 0;
    for(uint32_t j = 0; j < o_ncols; j++) {
        JA[j+1] = JA[j];
        for(uint32_t i = o_JA[j]; i < o_JA[j + 1]; i++) {
            if(o_A[i]) {
                JA[j+1]++;
                IA[idx] = o_IA[i];
                A[idx]  = o_A[i];
                idx++;
            }
        }
    }
    nnz = idx;
}

template<typename Weight>
inline void CSC<Weight>::spapopulate_t(struct DenseVec<Weight> *x_vector, struct DenseVec<Weight> *spa_vector, uint32_t col_idx, int tid) {
    Weight YMIN = 0;
    Weight YMAX = 32;
    Weight   *x_A = x_vector->A;
    Weight   *spa_A = spa_vector->A;
    Weight value = 0;
    auto &idx = Env::offset_nnz[tid];
    
    for(uint32_t i = 0; i < nrows; i++) {
        if(spa_A[i]) {
            spa_A[i] += x_A[col_idx];
            if(spa_A[i] < YMIN) {
                spa_A[i] = YMIN;
            }
            else if(spa_A[i] > YMAX) {
                spa_A[i] = YMAX;
            }
            if(spa_A[i]) {
                JA[col_idx+1]++;
                IA[idx] = i;
                A[idx] = spa_A[i];
                idx++;
                spa_A[i] = 0;
            }
        }
    }
}

template<typename Weight>
inline void CSC<Weight>::postpopulate_t(int tid) {
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
}

template<typename Weight>
inline void CSC<Weight>::repopulate(struct CSC<Weight> *other_csc, int tid){
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
}

template<typename Weight>
inline void CSC<Weight>::clear() {
    JA_blk->clear();
    IA_blk->clear();
    A_blk->clear();
}    

template<typename Weight>
inline void CSC<Weight>::walk() {
    double sum = 0;
    uint64_t k = 0;
    for(uint32_t j = 0; j < ncols; j++) {
        printf("j=%d: %d\n", j, JA[j + 1] - JA[j]);
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            IA[i];
            A[i];
            sum += A[i];
            k++;
            std::cout << "    i=" << IA[i] << ",j=" << j <<  ",value=" << A[i] << std::endl;
        }
    }
    printf("Checksum=%f, Count=%lu\n", sum, k);
}

*/

#endif
