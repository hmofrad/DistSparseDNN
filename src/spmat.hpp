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
        virtual void populate(const std::vector<struct Triple<Weight>> triples, const uint32_t tile_height, const uint32_t tile_width) {};
        //virtual void populate(const std::vector<struct Triple<Weight>>& triples, uint32_t nrows, uint32_t tile_height, uint32_t tile_width) {};
        //virtual void populate(const std::vector<struct Triple<Weight>> triples, const uint32_t nrows, const uint32_t ncols,
          //                    const uint32_t tile_height, const uint32_t tile_width) {};
        virtual void refine_both(const uint32_t nrows) {};
        virtual void refine_cols() {};
        virtual void refine_rows(const uint32_t nrows) {};
        virtual void populate_spa(std::vector<Weight>& spa, const std::vector<Weight> bias, const uint32_t col, const int32_t tid) {};
        virtual void adjust(const int32_t tid) {};
        virtual void reallocate(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_) {};
        virtual void walk(const int32_t tid) {};
        virtual void walk() {};
        
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
        //void populate(std::vector<struct Triple<Weight>>& triples, uint32_t nrows, uint32_t tile_height, uint32_t tile_width) {};
        //void populate(const std::vector<struct Triple<Weight>> triples, const uint32_t nrows, const uint32_t ncols, 
        //              const uint32_t tile_height, const uint32_t tile_width) {};
        void refine_both(const uint32_t nrows);
        void refine_cols();
        void refine_rows(const uint32_t nrows);
        void populate_spa(std::vector<Weight>& spa, const std::vector<Weight> bias, const uint32_t col, const int32_t tid);
        void walk(const int32_t tid);
        void walk();
        void reallocate(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_);
        void adjust(const int32_t tid);
        
        uint64_t nnz_i = 0;
};


template<typename Weight>
CSC<Weight>::CSC(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const bool one_rank_) {
    Compressed_Format<Weight>::compression_type = COMPRESSED_FORMAT::_CSC_;
    CSC::nnz = nnz_;
    CSC::nrows = nrows_; 
    //CSC::ncols_dummy = Env::nthreads;
    CSC::ncols = ncols_;// + Env::nthreads;
    CSC::one_rank = one_rank_;
    
    CSC::IA_blk = std::move(std::make_shared<struct Data_Block<uint32_t>>(CSC::nnz));
    CSC::JA_blk = std::move(std::make_shared<struct Data_Block<uint32_t>>(CSC::ncols + 1));
    CSC::A_blk = std::move(std::make_shared<struct Data_Block<Weight>>(CSC::nnz));
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

/*
template<typename Weight>
void populate(const std::vector<struct Triple<Weight>> triples, const uint32_t nrows, const uint32_t ncols, const uint32_t tile_height, const uint32_t tile_width) {
    
    
    
    //uint32_t l = B_IA[k];
    //            l +=  1 + l/((1028 - Env::nthreads) / Env::nthreads);
    
};
*/


template<typename Weight>
inline void CSC<Weight>::populate_spa(std::vector<Weight>& spa, const std::vector<Weight> bias, const uint32_t col, const int32_t tid) {
    uint64_t&  k = Env::index_nnz[tid];
    uint32_t   c = col + 1;// + Env::start_col[tid] + 1;// + tid;
    //uint32_t   c = col + Env::start_col[tid];
    uint32_t* IA = CSC::IA_blk->ptr;
    uint32_t* JA = CSC::JA_blk->ptr;
    Weight*    A = CSC::A_blk->ptr;
    
    Weight YMIN = 0;
    Weight YMAX = 32;
    //printf("%d %d %d\n", c, JA[c-1], JA[c]);
    //int jj = 0;
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
                //if(col==1025) jj++;
            }
        }
    }
    
    
   // printf("col=%d c=%d JA[c]=%d k=%lu %d %d %d %lu %d\n", col, c, JA[c], k, Env::start_col[tid], Env::end_col[tid], CSC::ncols, CSC::JA_blk->nitems, jj);
    //if(tid) {
      //  if((c < Env::start_col[tid] + 10 ) or (c > Env::end_col[tid] - 10))
        // printf("%d %d %d %lu %d\n", col, c, JA[c], k, Env::start_col[tid]);
        //if(c == 520) {
        //if(c == 10) {
          //  for(uint32_t ii = Env::start_col[tid]; ii < Env::start_col[tid]+10; ii++) 
            //    printf(" %d %d\n", ii, JA[ii]);
            
            //std::exit(0);
        //}
    //}
}



template<typename Weight>
void CSC<Weight>::walk(const int32_t tid) {  
    uint32_t* IA = CSC::IA_blk->ptr;
    uint32_t* JA = CSC::JA_blk->ptr;
    Weight*    A = CSC::A_blk->ptr;
    
    
    uint32_t start_col = Env::start_col[tid];// + tid + 1;
    uint32_t end_col = Env::end_col[tid];
    
    //printf("tid=%d start=%d/%d end=%d/%d len=%d/%d\n", tid, start_col, start_col - 1, end_col, end_col-1, end_col - start_col + 1, end_col - start_col + 1 -1 );
    #pragma omp barrier
    
    /*
    if(tid) {
        int t = JA[start_col];
        //JA[start_col] = JA[start_col-1];    
        JA[start_col+1] = Env::offset_nnz[tid];

        for(uint32_t j = 0; j < CSC::ncols; j++) {  
            int d = (j < CSC::ncols/2) ? 0 : tid;
            int jj = j - d;
            if(j < 520)
            std::cout << "j=" << j << ": jj=" << jj << "--" << jj+1 << ":" << JA[jj] << "--" << JA[jj + 1] << ": " <<  JA[jj + 1] - JA[jj] << std::endl;
        }
        printf("%d JA=%d\n", start_col, t);
            printf("JA=%d\n", JA[start_col]);
    }
    */
    
   // std::exit(0);
    
    //start_col = (((CSC::ncols )/Env::nthreads) *  tid  ) + 1;
    //end_col   =  ((CSC::ncols )/Env::nthreads) * (tid+1);
    //start_col = 0; 
    //end_col = CSC::ncols;
    //printf("2.%d %d %d %d\n", tid, start_col, end_col, end_col - start_col);
    //end_col++;
    //uint32_t displacement_nnz = Env::displacement_nnz[tid];
    //printf("%d %d %d\n", tid, start_col, end_col);
    Env::checksum[tid] = 0;
    Env::checkcount[tid] = 0;
    /*
    uint32_t displacement = (tid == 0) ? 0 : Env::offset_nnz[tid] - Env::index_nnz[tid-1];
    Env::displacement_nnz[tid] = displacement;
    JA[start_col] += Env::displacement_nnz[tid];
    */ 
 
    for(uint32_t j = start_col; j < end_col; j++) {  
    //for(uint32_t j = 0; j < CSC::ncols; j++) {  
       // uint32_t k = (j == start_col) ? displacement_nnz : 0;
       //uint32_t k = 0;
       // if(tid and ((j < start_col + 10) or (j > end_col - 10)))
        //if(!tid)
        //    std::cout << "j=" << j << "," << j-(tid+1) << ": " << JA[j] << "--" << JA[j + 1] << ": " <<  JA[j + 1] - JA[j] << std::endl;

        
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            (void) IA[i];
            (void) A[i];
            Env::checksum[tid] += A[i];
            Env::checkcount[tid]++;
            //std::cout << "    i=" << i << ",i=" << IA[i] <<  ",value=" << A[i] << std::endl;
        }
    }   
  // printf("tid=%d start=%d %d %d %d\n", tid, start_col, end_col, CSC::ncols, CSC::one_rank);
    //}
//printf("%d --> %lu %d\n", tid, Env::checkcount[tid], CSC::one_rank);
    Env::barrier();
    #pragma omp barrier  
    if(!tid) {
        double     sum_threads = std::accumulate(Env::checksum.begin(),   Env::checksum.end(),   0);
        uint64_t count_threads = std::accumulate(Env::checkcount.begin(), Env::checkcount.end(), 0);
        if(CSC::one_rank) {
            Logging::print(Logging::LOG_LEVEL::INFO, "Iteration=%d, Total checksum=%f, Total count=%d\n", Env::iteration, sum_threads, count_threads);
        }
        else {
            double sum_ranks = 0;
            uint64_t count_ranks = 0;
            
            MPI_Allreduce(&sum_threads, &sum_ranks, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&count_threads, &count_ranks, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
            
            if(count_threads != CSC::nnz_i) {
                Logging::print(Logging::LOG_LEVEL::WARN, "Compression checksum warning!!\n");
            }
            //printf("%f %lu\n", sum_threads, count_threads);
            Logging::print(Logging::LOG_LEVEL::INFO, "Iteration=%d, Total checksum=%f, Total count=%d\n", Env::iteration, sum_ranks, count_ranks);
        }
    }
    #pragma omp barrier    
}


template<typename Weight>
void CSC<Weight>::walk() {  
    uint32_t* IA = CSC::IA_blk->ptr;
    uint32_t* JA = CSC::JA_blk->ptr;
    Weight*    A = CSC::A_blk->ptr;
    
    double checksum = 0;
    uint64_t checkcount = 0;
    uint32_t displacement = 0;
    int t = 0;
    for(uint32_t j = 0; j < CSC::ncols; j++) { 
    
        if(j == Env::start_col[t]-1) {
            displacement = Env::displacement_nnz[t]; 
            t++;
        }
        else {
            displacement = 0;        
        }

        std::cout << "j" << j << ": " << JA[j] << "--" << JA[j + 1] << ": " <<  JA[j + 1] - JA[j] - displacement << "x " << displacement << "," << t << std::endl;
        //if(j > 700) break;
        for(uint32_t i = JA[j] + displacement; i < JA[j + 1]; i++) {
            (void) IA[i];
            (void) A[i];
            checksum += A[i];
            checkcount++;
        }
    }
    //printf("Rank = %d checksum= %f checkcount=%lu nnz=%lu nnzi=%lu\n", Env::rank, checksum, checkcount, CSC::nnz, CSC::nnz_i);
    //printf("ncols = %d\n", CSC::ncols);
    
    printf(" %lu %d %d -- %lu %d %d -- %lu %d %d\n", Env::offset_nnz[0], Env::start_col[0], Env::end_col[0], Env::offset_nnz[1], Env::start_col[1], Env::end_col[1], Env::offset_nnz[2], Env::start_col[2], Env::end_col[2]);
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
        //printf("%lu %lu\n", count_ranks, nnz_ranks);
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
    
    CSC::nnz_i = 0;
    CSC::nnz = nnz_;
    CSC::nrows = nrows_; 
    CSC::ncols = ncols_;
    
    CSC::IA_blk->reallocate(CSC::nnz);
    CSC::A_blk->reallocate(CSC::nnz);
    CSC::IA_blk->clear();
    CSC::JA_blk->clear();
    CSC::A_blk->clear();
    
    Env::memory_time += Env::toc(start_time);
}

template<typename Weight>
void CSC<Weight>::adjust(const int32_t tid){
    uint32_t displacement = (tid == 0) ? 0 : Env::offset_nnz[tid] - Env::index_nnz[tid-1];
    Env::displacement_nnz[tid] = displacement;
    
    uint32_t* JA = CSC::JA_blk->ptr;
    JA[Env::start_col[tid]] = Env::offset_nnz[tid];
    #pragma omp barrier
    if(!tid) {
        CSC::nnz_i = 0;
        for(int32_t i = 0; i < Env::nthreads; i++) {    
            CSC::nnz_i += (Env::index_nnz[i] - Env::offset_nnz[i]);
        }
       
        //for(int32_t i = 1; i < Env::nthreads; i++) {    
        //    JA[Env::start_col[i]-1] = JA[Env::start_col[i]];
        //}
        //JA[CSC::ncols] = JA[CSC::ncols-1];
    }
    #pragma omp barrier  

    if(!tid) {
        Env::nnz_ranks.push_back(CSC::nnz);
        Env::nnz_i_ranks.push_back(CSC::nnz_i);
    }
    
}

#endif
