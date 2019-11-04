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
        virtual void populate_spa(std::vector<Weight>& spa, std::vector<Weight> bias, const uint32_t& col, int32_t tid) {};
        virtual void adjust(int32_t tid) {};
        virtual void reallocate(uint64_t nnz_, uint32_t nrows_, uint32_t ncols_) {};
        virtual void walk(int32_t tid) {};
        virtual void statistics(int32_t tid) {};
        
        COMPRESSED_FORMAT compression_type;
        
        uint64_t nnz;
        uint32_t nrows;
        uint32_t ncols;
        
        std::shared_ptr<struct Data_Block<uint32_t>> IA_blk;
        std::shared_ptr<struct Data_Block<uint32_t>> JA_blk;
        std::shared_ptr<struct Data_Block<Weight>>   A_blk;
};

template<typename Weight>
struct CSC: public Compressed_Format<Weight> {
    public:
        CSC(uint64_t nnz_, uint32_t nrows_, uint32_t ncols_);
        ~CSC(){};
        
        void populate(std::vector<struct Triple<Weight>>& triples, uint32_t tile_height, uint32_t tile_width);
        void populate_spa(std::vector<Weight>& spa, std::vector<Weight> bias, const uint32_t col, int32_t tid);
        void walk(int32_t tid);
        void reallocate(uint64_t nnz_, uint32_t nrows_, uint32_t ncols_);
        void adjust(int32_t tid);
        void statistics(int32_t tid);
        
        uint64_t nnz_i = 0;
};


template<typename Weight>
CSC<Weight>::CSC(uint64_t nnz_, uint32_t nrows_, uint32_t ncols_) {
    Compressed_Format<Weight>::compression_type = COMPRESSED_FORMAT::_CSC_;
    CSC::nnz = nnz_;
    CSC::nrows = nrows_; 
    CSC::ncols = ncols_;
    
    CSC::IA_blk = std::move(std::make_shared<struct Data_Block<uint32_t>>(CSC::nnz));
    CSC::JA_blk = std::move(std::make_shared<struct Data_Block<uint32_t>>(CSC::ncols + 1));
    CSC::A_blk = std::move(std::make_shared<struct Data_Block<Weight>>(CSC::nnz));
}

template<typename Weight>
void CSC<Weight>::populate(std::vector<struct Triple<Weight>>& triples, uint32_t tile_height, uint32_t tile_width) {
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
}

template<typename Weight>
void CSC<Weight>::walk(int32_t tid) {  
    uint32_t* IA = CSC::IA_blk->ptr;
    uint32_t* JA = CSC::JA_blk->ptr;
    Weight*    A = CSC::A_blk->ptr;
    
    uint32_t start_col = Env::start_col[tid];
    uint32_t end_col = Env::end_col[tid];
    uint32_t displacement_nnz = Env::displacement_nnz[tid];
    Env::checksum[tid] = 0;
    Env::checkcount[tid] = 0;
    
    for(uint32_t j = start_col; j < end_col; j++) {        
        uint32_t k = (j == start_col) ? displacement_nnz : 0;
        //std::cout << "j=" << j << ": " << JA[j + 1] - JA[j] << std::endl;
        for(uint32_t i = JA[j] + k; i < JA[j + 1]; i++) {
            (void) IA[i];
            (void) A[i];
            Env::checksum[tid] += A[i];
            Env::checkcount[tid]++;
            //std::cout << "    i=" << i << ",i=" << IA[i] <<  ",value=" << A[i] << std::endl;
        }
    }    

    Env::barrier();
    #pragma omp barrier  
    if(!tid) {
        double   sum_threads = std::accumulate(Env::checksum.begin(), Env::checksum.end(), 0);
        uint64_t count_threads = std::accumulate(Env::checkcount.begin(), Env::checkcount.end(), 0);
        
        double sum_ranks = 0;
        uint64_t count_ranks = 0;
        
        MPI_Allreduce(&sum_threads, &sum_ranks, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&count_threads, &count_ranks, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
        
        if(count_ranks != CSC::nnz_i) {
            Logging::print(Logging::LOG_LEVEL::WARN, "Compression checksum warning!!\n");
        }
        Logging::print(Logging::LOG_LEVEL::INFO, "Checksum= %f, Count=%d\n", sum_ranks, count_ranks);
    }
    #pragma omp barrier    
}

template<typename Weight>
void CSC<Weight>::statistics(int32_t tid) {  
    if(!tid) {
        //double nnz_ranks = 0;
        //uint64_t count_ranks = 0;
        //MPI_Allreduce(&sum_threads, &sum_ranks, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        Logging::print(Logging::LOG_LEVEL::INFO, "%lu %lu %f\n", CSC::nnz, CSC::nnz_i, (double) CSC::nnz_i/CSC::nnz);
    }
}

template<typename Weight>
inline void CSC<Weight>::populate_spa(std::vector<Weight>& spa, std::vector<Weight> bias, const uint32_t col, int32_t tid) {
    uint64_t&  k = Env::index_nnz[tid];
    uint32_t   c = col + Env::start_col[tid] + 1;
    uint32_t* IA = CSC::IA_blk->ptr;
    uint32_t* JA = CSC::JA_blk->ptr;
    Weight*    A = CSC::A_blk->ptr;
    
    Weight YMIN = 0;
    Weight YMAX = 32;

    JA[c] = k;
    for(uint32_t i = 0; i < CSC::nrows; i++) {
        if(spa[i]) {
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
}

template<typename Weight>
void CSC<Weight>::reallocate(uint64_t nnz_, uint32_t nrows_, uint32_t ncols_) {
    if(CSC::ncols != ncols_) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Cannot reallocate.\n");
        std::exit(Env::finalize());     
    }
    
    CSC::nnz_i = 0;
    CSC::nnz = nnz_;
    CSC::nrows = nrows_; 
    CSC::ncols = ncols_;
    
    CSC::IA_blk->reallocate(CSC::nnz);
    CSC::A_blk->reallocate(CSC::nnz);
    CSC::IA_blk->clear();
    CSC::JA_blk->clear();
    CSC::A_blk->clear();
}

template<typename Weight>
void CSC<Weight>::adjust(int32_t tid){
    uint32_t displacement = (tid == 0) ? 0 : Env::offset_nnz[tid] - Env::index_nnz[tid-1];
    Env::displacement_nnz[tid] = displacement;
    #pragma omp barrier
    if(!tid) {
        CSC::nnz_i = 0;
        for(int32_t i = 0; i < Env::nthreads; i++) {    
            CSC::nnz_i += (Env::index_nnz[i] - Env::offset_nnz[i]);
        }
    }
    #pragma omp barrier   
}

#endif
