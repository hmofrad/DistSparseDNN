/*
 * spmat.hpp: Sparse Matrix implementation 
 * (c) Mohammad Hasanzadeh Mofrad, 2020
 * (e) m.hasanzadeh.mofrad@gmail.com
 *  Compression types:
 *  Compressed Sparse Row (CSR)
 *  Compressed Sparse Column (CSC)
 *  Doubly Compressed Sparse Row (DCSR)
 *  Doubly Compressed Sparse Column (DCSC)
 *  Triply Compressed Sparse Row (TCSR)
 *  Triply Compressed Sparse Column (TCSC)
 *  Uncompressed types:    
 *  Uncompressed Dense Row (UDR)
 *  Uncompressed Dense Column (UDC)
 */
 
#ifndef SPMAT_HPP
#define SPMAT_HPP

#include <numeric>
#include <limits.h>
#include <tuple>

#include "allocator.hpp"
#include "triple.hpp"
#include "env.hpp"

enum COMPRESSED_FORMAT {_CSR_, _DCSR_, _TCSR_, _CSC_, _DCSC_, _TCSC_, _UDR_, _UDC_};
const char* COMPRESSED_FORMATS[] = {"_CSR_", "_DCSR_", "_TCSR_", "_CSC_", "_DCSC_", "_TCSC_", "_UDR_", "_UDC_"};


template<typename Weight>
struct Compressed_Format {
    public:
        Compressed_Format() {}
        virtual ~Compressed_Format() {}
        virtual void populate(std::vector<struct Triple<Weight>>& triples) {Logging::print(Logging::LOG_LEVEL::ERROR, "Not implemented\n"); std::exit(Env::finalize());}
        virtual void populate_spa(Weight** spa, const Weight* bias, const uint32_t col,  uint64_t& index, Weight (*)(Weight), const int32_t tid){Logging::print(Logging::LOG_LEVEL::ERROR, "Not implemented\n"); std::exit(Env::finalize());}
        virtual void walk_dxm1(const bool one_rank, const int32_t leader_tid, const int32_t tid) {Logging::print(Logging::LOG_LEVEL::ERROR, "Not implemented\n"); std::exit(Env::finalize());}
        virtual void walk_dxm(const bool one_rank, const int32_t leader_tid, const int32_t tid) {Logging::print(Logging::LOG_LEVEL::ERROR, "Not implemented\n"); std::exit(Env::finalize());}
        virtual void walk_dxd(const bool one_rank, const int32_t leader_tid, const int32_t tid) {Logging::print(Logging::LOG_LEVEL::ERROR, "Not implemented\n"); std::exit(Env::finalize());}
        virtual void reallocate(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t leader_tid, const int32_t tid) {Logging::print(Logging::LOG_LEVEL::ERROR, "Not implemented\n"); std::exit(Env::finalize());}
        virtual void adjust(const int32_t tid) {Logging::print(Logging::LOG_LEVEL::ERROR, "Not implemented\n"); std::exit(Env::finalize());}
        virtual void adjust(const int32_t leader_tid, const int32_t tid) {Logging::print(Logging::LOG_LEVEL::ERROR, "Not implemented\n"); std::exit(Env::finalize());}
        virtual void adjust(const std::deque<int32_t> my_threads, const int32_t leader_tid, const int32_t tid) {Logging::print(Logging::LOG_LEVEL::ERROR, "Not implemented\n"); std::exit(Env::finalize());}
        virtual void repopulate(const std::shared_ptr<struct Compressed_Format<Weight>> other_spmat, const uint32_t dis_nnz, const int32_t leader_tid, const int32_t tid) {Logging::print(Logging::LOG_LEVEL::ERROR, "Not implemented\n"); std::exit(Env::finalize());}
        virtual void repopulate(const std::shared_ptr<struct Compressed_Format<Weight>> other_spmat, const std::deque<int32_t> my_threads, const int32_t leader_tid, const int32_t tid) {Logging::print(Logging::LOG_LEVEL::ERROR, "Not implemented\n"); std::exit(Env::finalize());}
        
        COMPRESSED_FORMAT compression_type;
        
        uint64_t nnz   = 0;
        uint64_t nnz_i = 0;
        uint32_t nrows = 0;
        uint32_t ncols = 0;
        
        std::shared_ptr<struct Data_Block<uint32_t>> IA_blk;
        std::shared_ptr<struct Data_Block<uint32_t>> JA_blk;
        std::shared_ptr<struct Data_Block<Weight>>   A_blk;
};

template<typename Weight>
struct CSR: public Compressed_Format<Weight> {
    public:
        CSR(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t socket_id);
        ~CSR(){};
        
        void populate(std::vector<struct Triple<Weight>>& triples);
        void populate_spa(Weight** spa, const Weight* bias, const uint32_t col,  uint64_t& index, Weight (*)(Weight), const int32_t tid);
        void walk_dxm1(const bool one_rank, const int32_t leader_tid, const int32_t tid){};
        void walk_dxm(const bool one_rank, const int32_t leader_tid, const int32_t tid);
        void walk_dxd(const bool one_rank, const int32_t leader_tid, const int32_t tid);
        void reallocate(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t leader_tid, const int32_t tid);
        void adjust(const int32_t tid);
        void adjust(const int32_t leader_tid, const int32_t tid);
        void adjust(const std::deque<int32_t> my_threads, const int32_t leader_tid, const int32_t tid);
        void repopulate(const std::shared_ptr<struct Compressed_Format<Weight>> other_spmat, const uint32_t dis_nnz, const int32_t leader_tid, const int32_t tid);
        void repopulate(const std::shared_ptr<struct Compressed_Format<Weight>> other_spmat, const std::deque<int32_t> my_threads, const int32_t leader_tid, const int32_t tid);
        
        uint64_t nnz   = 0;
        uint64_t nnz_i = 0;
        uint32_t nrows = 0;
        uint32_t ncols = 0;
        
        std::shared_ptr<struct Data_Block<uint32_t>> IA_blk;
        std::shared_ptr<struct Data_Block<uint32_t>> JA_blk;
        std::shared_ptr<struct Data_Block<Weight>>   A_blk;
};


/* Compressed Sparse Row (CSR) */
template<typename Weight>
CSR<Weight>::CSR(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t socket_id) {
    Compressed_Format<Weight>::compression_type = COMPRESSED_FORMAT::_CSR_;
    Compressed_Format<Weight>::nnz = nnz_;
    Compressed_Format<Weight>::nnz_i = nnz_;
    Compressed_Format<Weight>::nrows = nrows_; 
    Compressed_Format<Weight>::ncols = ncols_;
    
    CSR::compression_type = COMPRESSED_FORMAT::_CSR_;
    CSR::nnz = nnz_;
    CSR::nnz_i = nnz_;
    CSR::nrows = nrows_; 
    CSR::ncols = ncols_;
    
    CSR::IA_blk = std::move(std::make_shared<struct Data_Block<uint32_t>>((CSR::nrows + 1), socket_id));
    CSR::JA_blk = std::move(std::make_shared<struct Data_Block<uint32_t>>(CSR::nnz, socket_id));
    CSR::A_blk = std::move(std::make_shared<struct Data_Block<Weight>>(CSR::nnz, socket_id));
}

template<typename Weight>
void CSR<Weight>::populate(std::vector<struct Triple<Weight>>& triples) {
    const RowSort<Weight> f_row;
    std::sort(triples.begin(), triples.end(), f_row);    
    
    uint32_t* IA = CSR::IA_blk->ptr;
    uint32_t* JA = CSR::JA_blk->ptr;
    Weight* A = CSR::A_blk->ptr;
        
    uint32_t i = 1;
    uint32_t j = 0; 
    IA[0] = 0;
    for(struct Triple<Weight>& triple: triples) {
        uint32_t row = triple.row % CSR::nrows;
        uint32_t col = triple.col % CSR::ncols;
        Weight weight = triple.weight;
        while((i - 1) != row) {
            i++;
            IA[i] = IA[i - 1];
        }                  
        IA[i]++;
        JA[j] = col;
        A[j] = weight;
        j++;
    }
    
    while(i < CSR::nrows) {
        i++;
        IA[i] = IA[i - 1];
    }
}

template<typename Weight>
void CSR<Weight>::populate_spa(Weight** spa, const Weight* bias, const uint32_t row, uint64_t& index, Weight(*activation_function)(Weight), const int32_t tid) {
    uint64_t&  k = index;
    uint32_t   r = row + 1;
    uint32_t* IA = CSR::IA_blk->ptr;
    uint32_t* JA = CSR::JA_blk->ptr;
    Weight*    A = CSR::A_blk->ptr;
    Weight*    s = *spa;
    const Weight* b = bias;
    
    for(uint32_t j = 0; j < CSR::ncols; j++) {
        if(s[j]) {
            s[j] += b[j];
            s[j]=activation_function(s[j]);
            if(s[j]) {
                JA[k] = j;
                A[k] = s[j];
                k++;
                s[j] = 0;
            }
       }
    }
    IA[r] = k;
}

template<typename Weight>
void CSR<Weight>::walk_dxm(const bool one_rank, const int32_t leader_tid, const int32_t tid) {  
    if(tid == leader_tid) {
        uint32_t* IA = CSR::IA_blk->ptr;
        uint32_t* JA = CSR::JA_blk->ptr;
        Weight*    A = CSR::A_blk->ptr;
        
        double checksum = 0;
        uint64_t checkcount = 0;
        for(uint32_t i = 0; i < CSR::nrows; i++) {
            //if(!Env::rank)
            //    std::cout << "i=" << i << "," << i << ": " << IA[i] << "--" << IA[i + 1] << ": " <<  IA[i + 1] - IA[i] << std::endl;
            for(uint32_t j = IA[i]; j < IA[i + 1]; j++) {
                (void) JA[j];
                (void) A[j];
                checksum += A[j];
                checkcount++;
                //if(!Env::rank)
                //    std::cout << "    j=" << j << ",j=" << JA[j] <<  ",value=" << A[j] << std::endl;
            }
        }

        Env::barrier();
        if(one_rank) {
            Logging::print(Logging::LOG_LEVEL::INFO, "CSR: Iteration=%d, Total checksum=%f, Total count=%d\n", Env::iteration, checksum, checkcount);
        }
        else {
            uint64_t nnz_ = CSR::nnz_i;
            uint64_t nnz_ranks = 0;
            double sum_ranks = 0;
            uint64_t count_ranks = 0;
            MPI_Allreduce(&nnz_, &nnz_ranks, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&checksum, &sum_ranks, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&checkcount, &count_ranks, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);

            if(count_ranks != nnz_ranks) {
                Logging::print(Logging::LOG_LEVEL::WARN, "Compression checksum warning!!\n");
            }
            Logging::print(Logging::LOG_LEVEL::INFO, "tid=%d CSR: Iteration=%d, Total checksum=%f, Total count=%d\n", tid, Env::iteration, sum_ranks, count_ranks);
        } 
    }    
}

template<typename Weight>
void CSR<Weight>::walk_dxd(const bool one_rank, const int32_t leader_tid, const int32_t tid) {
    uint32_t* IA = CSR::IA_blk->ptr;
    uint32_t* JA = CSR::JA_blk->ptr;
    Weight*    A = CSR::A_blk->ptr;
    
    double&   checksum   = Env::counters[tid].checksum;
    uint64_t& checkcount = Env::counters[tid].checkcount;
    uint64_t& checknnz   = Env::counters[tid].checknnz;
    
    checksum   = 0;
    checkcount = 0;    
    checknnz   = CSR::nnz_i;

    for(uint32_t i = 0; i < CSR::nrows; i++) { 
        //if(!Env::rank and !tid)
            //std::cout << "i=" << i << "," << i << ": " << IA[i] << "--" << IA[i + 1] << ": " <<  IA[i + 1] - IA[i] << std::endl;    
        for(uint32_t j = IA[i]; j < IA[i + 1]; j++) {
            (void) JA[j];
            (void) A[j];
            checksum += A[j];
            checkcount++;
            //if(!Env::rank and !tid)
            //std::cout << "    j=" << j << ",j=" << JA[j] <<  ",value=" << A[j] << std::endl;
        }
    }   
    
    Env::barrier();
    pthread_barrier_wait(&Env::thread_barrier);
    if(tid == leader_tid) {
        double     sum_threads = 0;
        uint64_t count_threads = 0;
        uint64_t   nnz_threads = 0;
        
        for(auto it = Env::counters.begin(); it != Env::counters.end(); it++) {
            sum_threads   += (*it).checksum;
            count_threads += (*it).checkcount;
            nnz_threads   += (*it).checknnz;
        }
        
        if(one_rank) {
            Logging::print(Logging::LOG_LEVEL::INFO, "CSR: Iteration=%d, Total checksum=%f, Total count=%d\n", Env::iteration, sum_threads, count_threads);
        }
        else {
            double     sum_ranks = 0;
            uint64_t count_ranks = 0;
            
            MPI_Allreduce(&sum_threads, &sum_ranks, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&count_threads, &count_ranks, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
            
            if(count_threads != nnz_threads) {
                Logging::print(Logging::LOG_LEVEL::WARN, "Compression checksum warning!! (%lu != %lu)\n", count_threads, nnz_threads);
            }
            Logging::print(Logging::LOG_LEVEL::INFO, "CSR: Iteration=%d, Total checksum=%f, Total count=%d\n", Env::iteration, sum_ranks, count_ranks);
        }
    }
}

template<typename Weight>
void CSR<Weight>::reallocate(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t leader_tid, const int32_t tid) {
    if((leader_tid == -1) or (tid == leader_tid)) {
        CSR::nnz = nnz_;
        CSR::nnz_i = 0;
        CSR::nrows = nrows_; 
        CSR::ncols = ncols_;
        CSR::IA_blk->reallocate(CSR::nrows+1);
        CSR::IA_blk->clear();
        CSR::JA_blk->reallocate(CSR::nnz);
        CSR::JA_blk->clear();
        CSR::A_blk->reallocate(CSR::nnz);
        CSR::A_blk->clear();    
        Compressed_Format<Weight>::nnz = nnz_;
        Compressed_Format<Weight>::nnz_i = 0;
        Compressed_Format<Weight>::nrows = nrows_; 
        Compressed_Format<Weight>::ncols = ncols_;
    }
}

template<typename Weight>
void CSR<Weight>::adjust(const int32_t tid){
    CSR::nnz_i = Env::threads[tid].idx_nnz;
}

template<typename Weight>
void CSR<Weight>::adjust(const int32_t leader_tid, const int32_t tid){
    if((leader_tid == -1) or (tid == leader_tid)) {
        CSR::nnz_i = 0;
        for(uint32_t i = 0; i < Env::threads.size(); i++) {    
            CSR::nnz_i += (Env::threads[i].idx_nnz - Env::threads[i].off_nnz);
        }
    }
    pthread_barrier_wait(&Env::thread_barrier);
}

template<typename Weight>
void CSR<Weight>::adjust(const std::deque<int32_t> my_threads, const int32_t leader_tid, const int32_t tid){    
    if((leader_tid == -1) or (tid == leader_tid)) {
        CSR::nnz_i = 0;
        for(uint32_t i = 0; i < my_threads.size(); i++) {    
            int32_t t = my_threads[i];
            CSR::nnz_i += (Env::threads[t].idx_nnz - Env::threads[t].off_nnz);
        }
    }
    pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
}

template<typename Weight>
void CSR<Weight>::repopulate(const std::shared_ptr<struct Compressed_Format<Weight>> other_spmat, const uint32_t dis_nnz, const int32_t leader_tid, const int32_t tid) {
    std::shared_ptr<struct CSR<Weight>> other_csr = std::static_pointer_cast<struct CSR<Weight>>(other_spmat);
    
    uint32_t  o_ncols = other_csr->ncols;
    uint32_t  o_nrows = other_csr->nrows;
    uint64_t  o_nnz   = other_csr->nnz;
    uint64_t  o_nnz_i = other_csr->nnz_i;
    uint32_t* o_JA    = other_csr->JA_blk->ptr;
    uint32_t* o_IA    = other_csr->IA_blk->ptr;
    Weight*   o_A     = other_csr->A_blk->ptr;

    if(tid == leader_tid) {
        CSR::nnz = o_nnz_i;
        CSR::nnz_i = o_nnz_i;
        CSR::nrows = o_nrows;
        CSR::ncols = o_ncols;
        CSR::IA_blk->reallocate(CSR::nrows+1);
        CSR::IA_blk->clear();
        CSR::JA_blk->reallocate(CSR::nnz_i);
        CSR::JA_blk->clear();
        CSR::A_blk->reallocate(CSR::nnz_i);
        CSR::A_blk->clear();
        
        Compressed_Format<Weight>::nnz = CSR::nnz_i;
        Compressed_Format<Weight>::nnz_i = CSR::nnz_i;
        Compressed_Format<Weight>::nrows = CSR::nrows;
        Compressed_Format<Weight>::ncols = CSR::ncols;
    }    
    pthread_barrier_wait(&Env::thread_barrier);
    
    uint32_t* IA = CSR::IA_blk->ptr;
    uint32_t* JA = CSR::JA_blk->ptr;
    Weight*    A = CSR::A_blk->ptr;

    const uint32_t start_row = Env::threads[tid].start_row;
    const uint32_t end_row   = Env::threads[tid].end_row;

    for(int32_t i = 0; i < tid; i++) {
        IA[start_row+1] += (Env::threads[i].idx_nnz - Env::threads[i].off_nnz);
    }

    for(uint32_t i = start_row; i < end_row; i++) {
        IA[i+1] = (i == start_row) ? IA[i+1] : IA[i];
        uint32_t& k = IA[i+1];
        uint32_t m = (i == start_row) ? dis_nnz : 0;
        for(uint32_t j = o_IA[i] + m; j < o_IA[i + 1]; j++) {
            JA[k] = o_JA[j];
            A[k]  = o_A[j];
            k++;
        }
    }
    
    pthread_barrier_wait(&Env::thread_barrier);
}

template<typename Weight>
void CSR<Weight>::repopulate(const std::shared_ptr<struct Compressed_Format<Weight>> other_spmat, const std::deque<int32_t> my_threads, const int32_t leader_tid,  const int32_t tid) {
    std::shared_ptr<struct CSR<Weight>> other_csr = std::static_pointer_cast<struct CSR<Weight>>(other_spmat);
    
    uint32_t  o_ncols = other_csr->ncols;
    uint32_t  o_nrows = other_csr->nrows;
    uint64_t  o_nnz   = other_csr->nnz;
    uint64_t  o_nnz_i = other_csr->nnz_i;
    uint32_t* o_JA    = other_csr->JA_blk->ptr;
    uint32_t* o_IA    = other_csr->IA_blk->ptr;
    Weight*   o_A     = other_csr->A_blk->ptr;

    if(tid == leader_tid) {
        CSR::nnz = o_nnz_i;
        CSR::nnz_i = o_nnz_i;
        CSR::nrows = o_nrows;
        CSR::ncols = o_ncols;
        CSR::IA_blk->reallocate(CSR::nrows+1);
        CSR::IA_blk->clear();
        CSR::JA_blk->reallocate(CSR::nnz_i);
        CSR::JA_blk->clear();
        CSR::A_blk->reallocate(CSR::nnz_i);
        CSR::A_blk->clear();
        
        Compressed_Format<Weight>::nnz = CSR::nnz_i;
        Compressed_Format<Weight>::nnz_i = CSR::nnz_i;
        Compressed_Format<Weight>::nrows = CSR::nrows;
        Compressed_Format<Weight>::ncols = CSR::ncols;
    }    
    pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
    
    uint32_t* JA = CSR::JA_blk->ptr;
    uint32_t* IA = CSR::IA_blk->ptr;
    Weight*    A = CSR::A_blk->ptr;
    
    const uint32_t start_row = Env::threads[tid].start_row;
    const uint32_t end_row   = Env::threads[tid].end_row;    
   
    for(uint32_t j = 0; j < Env::threads[tid].index; j++) {
        int32_t tt = Env::my_threads[leader_tid][j];
        IA[start_row+1] += (Env::threads[tt].idx_nnz - Env::threads[tt].off_nnz);
    }
    
    for(uint32_t i = start_row; i < end_row; i++) {
        IA[i+1] = (i == start_row) ? IA[i+1] : IA[i];
        uint32_t& k = IA[i+1];
        uint32_t m = (i == start_row) ? Env::threads[tid].dis_nnz : 0;
        for(uint32_t j = o_IA[i] + m; j < o_IA[i + 1]; j++) {
            JA[k] = o_JA[j];
            A[k]  = o_A[j];
            k++;
        }
    }
    pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
}

/* Compressed Sparse Column (CSC) */
template<typename Weight>
struct CSC: public Compressed_Format<Weight> {
    public:
        CSC(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t socket_id);        
        ~CSC(){};
        
        void populate(std::vector<struct Triple<Weight>>& triples);
        void populate_spa(Weight** spa, const Weight* bias, const uint32_t col,  uint64_t& index, Weight (*)(Weight), const int32_t tid);
        void walk_dxm1(const bool one_rank, const int32_t leader_tid, const int32_t tid);
        void walk_dxm(const bool one_rank, const int32_t leader_tid, const int32_t tid);
        void walk_dxd(const bool one_rank, const int32_t leader_tid, const int32_t tid);
        void reallocate(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t leader_tid, const int32_t tid);
        void adjust(const int32_t tid);
        void adjust(const int32_t leader_tid, const int32_t tid);
        void adjust(const std::deque<int32_t> my_threads, const int32_t leader_tid, const int32_t tid);
        void repopulate(const std::shared_ptr<struct Compressed_Format<Weight>> other_spmat, const uint32_t dis_nnz, const int32_t leader_tid, const int32_t tid);
        void repopulate(const std::shared_ptr<struct Compressed_Format<Weight>> other_spmat, const std::deque<int32_t> my_threads, const int32_t leader_tid, const int32_t tid);
        
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
    Compressed_Format<Weight>::compression_type = COMPRESSED_FORMAT::_CSC_;
    Compressed_Format<Weight>::nnz = nnz_;
    Compressed_Format<Weight>::nnz_i = nnz_;
    Compressed_Format<Weight>::nrows = nrows_; 
    Compressed_Format<Weight>::ncols = ncols_;
    
    CSC::compression_type = COMPRESSED_FORMAT::_CSC_;
    CSC::nnz = nnz_;
    CSC::nnz_i = nnz_;
    CSC::nrows = nrows_; 
    CSC::ncols = ncols_;
    
    CSC::JA_blk = std::move(std::make_shared<struct Data_Block<uint32_t>>((CSC::ncols + 1), socket_id));
    CSC::IA_blk = std::move(std::make_shared<struct Data_Block<uint32_t>>(CSC::nnz, socket_id));
    CSC::A_blk = std::move(std::make_shared<struct Data_Block<Weight>>(CSC::nnz, socket_id));
}

template<typename Weight>
void CSC<Weight>::populate(std::vector<struct Triple<Weight>>& triples) {
    const ColSort<Weight> f_col;
    std::sort(triples.begin(), triples.end(), f_col);  
    
    uint32_t* IA = CSC::IA_blk->ptr;
    uint32_t* JA = CSC::JA_blk->ptr;
    Weight* A = CSC::A_blk->ptr;
    
    uint32_t i = 0;
    uint32_t j = 1; 
    JA[0] = 0;
    for(struct Triple<Weight>& triple: triples) {
        uint32_t row = triple.row % CSC::nrows;
        uint32_t col = triple.col % CSC::ncols;
        Weight weight = triple.weight;
        while((j - 1) != col) {
            j++;
            JA[j] = JA[j - 1];
        }                  
        JA[j]++;
        IA[i] = row;
        A[i] = weight;
        i++;
    }

    while(j < CSC::ncols) {
        j++;
        JA[j] = JA[j - 1];
    }
}

template<typename Weight>
void CSC<Weight>::populate_spa(Weight** spa, const Weight* bias, const uint32_t col, uint64_t& index, Weight(*activation_function)(Weight), const int32_t tid) {
    uint64_t&  k = index;
    uint32_t   c = col + 1;
    uint32_t* IA = CSC::IA_blk->ptr;
    uint32_t* JA = CSC::JA_blk->ptr;
    Weight*    A = CSC::A_blk->ptr;
    Weight*    s = *spa;
    const Weight* b = bias;
    
    for(uint32_t i = 0; i < CSC::nrows; i++) {
        if(s[i]) {
            s[i] += b[c-1];
            s[i]=activation_function(s[i]);
            if(s[i]) {
                IA[k] = i;
                A[k] = s[i];
                k++;
                s[i] = 0;
            }
        }
    }
    JA[c] = k;
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
            //    std::cout << "j=" << j << "," << j << ": " << JA[j] << "--" << JA[j + 1] << ": " <<  JA[j + 1] - JA[j] << std::endl;
            for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
                (void) IA[i];
                (void) A[i];
                checksum += A[i];
                checkcount++;
                //if(!Env::rank)
                //    std::cout << "    i=" << i << ",i=" << IA[i] <<  ",value=" << A[i] << std::endl;
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
                Logging::print(Logging::LOG_LEVEL::WARN, "Compression checksum warning!! (%lu != %lu)\n", count_ranks, nnz_ranks);
            }
            Logging::print(Logging::LOG_LEVEL::INFO, "tid=%d CSC: Iteration=%d, Total checksum=%f, Total count=%d\n", tid, Env::iteration, sum_ranks, count_ranks);
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
           // std::cout << "j=" << j << "," << j+1 << ": " << JA[j] << "--" << JA[j + 1] << ": " <<  JA[j + 1] - JA[j] << std::endl;    
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
        double     sum_threads = 0;
        uint64_t count_threads = 0;
        uint64_t   nnz_threads = 0;
        
        for(auto it = Env::counters.begin(); it != Env::counters.end(); it++) {
            sum_threads   += (*it).checksum;
            count_threads += (*it).checkcount;
            nnz_threads   += (*it).checknnz;
        }
        
        if(one_rank) {
            Logging::print(Logging::LOG_LEVEL::INFO, "CSC: Iteration=%d, Total checksum=%f, Total count=%d\n", Env::iteration, sum_threads, count_threads);
        }
        else {
            double     sum_ranks = 0;
            uint64_t count_ranks = 0;
            
            MPI_Allreduce(&sum_threads, &sum_ranks, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&count_threads, &count_ranks, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
            
            if(count_threads != nnz_threads) {
                Logging::print(Logging::LOG_LEVEL::WARN, "Compression checksum warning!! (%lu != %lu)\n", count_threads, nnz_threads);
            }
            Logging::print(Logging::LOG_LEVEL::INFO, "CSC: Iteration=%d, Total checksum=%f, Total count=%d\n", Env::iteration, sum_ranks, count_ranks);
        }
    }    
}

template<typename Weight>
void CSC<Weight>::reallocate(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t leader_tid, const int32_t tid) {
    if((leader_tid == -1) or (tid == leader_tid)) {
        CSC::nnz = nnz_;
        CSC::nnz_i = 0;
        CSC::nrows = nrows_; 
        CSC::ncols = ncols_;
        CSC::JA_blk->reallocate(CSC::ncols+1);
        CSC::JA_blk->clear();
        CSC::IA_blk->reallocate(CSC::nnz);
        CSC::IA_blk->clear();
        CSC::A_blk->reallocate(CSC::nnz);
        CSC::A_blk->clear();
        
        Compressed_Format<Weight>::nnz = nnz_;
        Compressed_Format<Weight>::nnz_i = 0;
        Compressed_Format<Weight>::nrows = nrows_; 
        Compressed_Format<Weight>::ncols = ncols_;
    }
}

template<typename Weight>
void CSC<Weight>::adjust(const int32_t tid){
    CSC::nnz_i = Env::threads[tid].idx_nnz;
}

template<typename Weight>
void CSC<Weight>::adjust(const int32_t leader_tid, const int32_t tid){
    if((leader_tid == -1) or (tid == leader_tid)) {
        CSC::nnz_i = 0;
        for(uint32_t i = 0; i < Env::threads.size(); i++) {    
            CSC::nnz_i += (Env::threads[i].idx_nnz - Env::threads[i].off_nnz);
        }
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
        }
    }
    pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
}

template<typename Weight>
void CSC<Weight>::repopulate(const std::shared_ptr<struct Compressed_Format<Weight>> other_spmat, const uint32_t dis_nnz, const int32_t leader_tid, const int32_t tid) {
    std::shared_ptr<struct CSC<Weight>> other_csc = std::static_pointer_cast<struct CSC<Weight>>(other_spmat);
    
    uint32_t  o_ncols = other_csc->ncols;
    uint32_t  o_nrows = other_csc->nrows;
    uint64_t  o_nnz   = other_csc->nnz;
    uint64_t  o_nnz_i = other_csc->nnz_i;
    uint32_t* o_JA    = other_csc->JA_blk->ptr;
    uint32_t* o_IA    = other_csc->IA_blk->ptr;
    Weight*   o_A     = other_csc->A_blk->ptr;

    if(tid == leader_tid) {
        CSC::nnz = o_nnz_i;
        CSC::nnz_i = o_nnz_i;
        CSC::ncols = o_ncols;
        CSC::nrows = o_nrows;
        CSC::JA_blk->reallocate(CSC::ncols+1);
        CSC::JA_blk->clear();
        CSC::IA_blk->reallocate(CSC::nnz_i);
        CSC::IA_blk->clear();
        CSC::A_blk->reallocate(CSC::nnz_i);
        CSC::A_blk->clear();
        Compressed_Format<Weight>::nnz = CSC::nnz_i;
        Compressed_Format<Weight>::nnz_i = CSC::nnz_i;
        Compressed_Format<Weight>::nrows = CSC::nrows;
        Compressed_Format<Weight>::ncols = CSC::ncols;
    }
    pthread_barrier_wait(&Env::thread_barrier);
    
    uint32_t* JA = CSC::JA_blk->ptr;
    uint32_t* IA = CSC::IA_blk->ptr;
    Weight*    A = CSC::A_blk->ptr;
    
    const uint32_t start_col = Env::threads[tid].start_col;
    const uint32_t end_col   = Env::threads[tid].end_col;
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
}

template<typename Weight>
void CSC<Weight>::repopulate(const std::shared_ptr<struct Compressed_Format<Weight>> other_spmat, const std::deque<int32_t> my_threads, const int32_t leader_tid,  const int32_t tid) {
    std::shared_ptr<struct CSC<Weight>> other_csc = std::static_pointer_cast<struct CSC<Weight>>(other_spmat);
    
    uint32_t  o_ncols = other_csc->ncols;
    uint32_t  o_nrows = other_csc->nrows;
    uint64_t  o_nnz   = other_csc->nnz;
    uint64_t  o_nnz_i = other_csc->nnz_i;
    uint32_t* o_JA    = other_csc->JA_blk->ptr;
    uint32_t* o_IA    = other_csc->IA_blk->ptr;
    Weight*   o_A     = other_csc->A_blk->ptr;

    if(tid == leader_tid) {
        CSC::nnz = o_nnz_i;
        CSC::nnz_i = o_nnz_i;
        CSC::ncols = o_ncols;
        CSC::nrows = o_nrows;
        CSC::JA_blk->reallocate(CSC::ncols+1);
        CSC::JA_blk->clear();
        CSC::IA_blk->reallocate(CSC::nnz_i);
        CSC::IA_blk->clear();
        CSC::A_blk->reallocate(CSC::nnz_i);
        CSC::A_blk->clear();
        Compressed_Format<Weight>::nnz = CSC::nnz_i;
        Compressed_Format<Weight>::nnz_i = CSC::nnz_i;
        Compressed_Format<Weight>::nrows = CSC::nrows;
        Compressed_Format<Weight>::ncols = CSC::ncols;
    }
    pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
    
    uint32_t* JA = CSC::JA_blk->ptr;
    uint32_t* IA = CSC::IA_blk->ptr;
    Weight*    A = CSC::A_blk->ptr;

    const uint32_t start_col = Env::threads[tid].start_col;
    const uint32_t end_col   = Env::threads[tid].end_col;
    for(uint32_t j = 0; j < Env::threads[tid].index; j++) {
        int32_t tt = Env::my_threads[leader_tid][j];
        JA[start_col+1] += (Env::threads[tt].idx_nnz - Env::threads[tt].off_nnz);
    }
    
    for(uint32_t j = start_col; j < end_col; j++) {
        JA[j+1] = (j == start_col) ? JA[j+1] : JA[j];
        uint32_t& k = JA[j+1];
        uint32_t m = (j == start_col) ? Env::threads[tid].dis_nnz : 0;
        for(uint32_t i = o_JA[j] + m; i < o_JA[j+1]; i++) {
            IA[k] = o_IA[i];
            A[k]  = o_A[i];
            k++;
        }
    }
    pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
}

template<typename Weight>
void CSC<Weight>::walk_dxm1(const bool one_rank, const int32_t leader_tid, const int32_t tid) {  
    if(tid == leader_tid) {
        uint32_t* IA = CSC::IA_blk->ptr;
        uint32_t* JA = CSC::JA_blk->ptr;
        Weight*    A = CSC::A_blk->ptr;
        
        double checksum = 0;
        uint64_t checkcount = 0;
         uint32_t displacement = 0;
        int t = 0;
        for(uint32_t j = 0; j < CSC::ncols; j++) { 
            int32_t tt = Env::my_threads[leader_tid][t];
            if(j == Env::threads[tt].start_col) {
                displacement = Env::threads[tt].dis_nnz; 
                t++;
            }
            else {
                displacement = 0;        
            }
           if(!Env::rank)
                std::cout << "j=" << j << ": " << JA[j] << "--" << JA[j + 1] << ": " <<  JA[j + 1] - JA[j] << std::endl;

            for(uint32_t i = JA[j] + displacement; i < JA[j + 1]; i++) {
                (void) IA[i];
                (void) A[i];
                checksum += A[i];
                checkcount++;
            }
        }

        Env::barrier();
        if(one_rank) {
            Logging::print(Logging::LOG_LEVEL::INFO, "Iteration=%d, Total checksum=%f, Total count=%d\n", Env::iteration, checksum, checkcount);
            if(checkcount==0) std::exit(0);
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
                Logging::print(Logging::LOG_LEVEL::WARN, "Compression checksum warning!! 1\n");
            }
            Logging::print(Logging::LOG_LEVEL::INFO, "Iteration=%d, Total checksum=%f, Total count=%d 1\n", Env::iteration, sum_ranks, count_ranks);
        } 
    }    
}


template<typename Weight>
struct UDC: public Compressed_Format<Weight> {
    public:
        UDC(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t socket_id);
        ~UDC(){};
        
        void populate(std::vector<struct Triple<Weight>>& triples);
        void reallocate(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t leader_tid, const int32_t tid);
        void populate_spa(Weight** spa, const Weight* bias, const uint32_t col,  uint64_t& index, Weight (*)(Weight), const int32_t tid);
        void walk_dxm(const bool one_rank, const int32_t leader_tid, const int32_t tid);
        
        uint64_t nnz   = 0;
        uint64_t nnz_i = 0;
        uint32_t nrows = 0;
        uint32_t ncols = 0;
};

template<typename Weight>
UDC<Weight>::UDC(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t socket_id) {
    Compressed_Format<Weight>::compression_type = COMPRESSED_FORMAT::_UDC_;
    Compressed_Format<Weight>::nnz = nnz_;
    Compressed_Format<Weight>::nnz_i = nnz_;
    Compressed_Format<Weight>::nrows = nrows_; 
    Compressed_Format<Weight>::ncols = ncols_;
    
    UDC::compression_type = COMPRESSED_FORMAT::_UDC_;
    UDC::nnz = nnz_;
    UDC::nnz_i = nnz_;
    UDC::nrows = nrows_; //tile_height
    UDC::ncols = ncols_; //tile_width
    
    UDC::A_blk = std::move(std::make_shared<struct Data_Block<Weight>>(UDC::nrows * UDC::ncols, socket_id));  
}

//(row*width) + col
template<typename Weight>
void UDC<Weight>::populate(std::vector<struct Triple<Weight>>& triples) {
    // Not necessary, however, it provides sequential access to triples
    const ColSort<Weight> f_col;
    std::sort(triples.begin(), triples.end(), f_col);  
    
    Weight* A = UDC::A_blk->ptr;
    for(struct Triple<Weight>& triple: triples) {
        uint32_t row = (triple.row % UDC::nrows);
        uint32_t col = (triple.col % UDC::ncols);
        Weight weight = triple.weight;
        uint64_t index = (col * UDC::nrows) + row;
        A[index]= weight;
    }
}
template<typename Weight>
void UDC<Weight>::reallocate(const uint64_t nnz_, const uint32_t nrows_, const uint32_t ncols_, const int32_t leader_tid, const int32_t tid) {
    if((leader_tid == -1) or (tid == leader_tid)) {
        UDC::nrows = nrows_; 
        UDC::ncols = ncols_;
        UDC::A_blk->reallocate(UDC::nrows * UDC::ncols);
        UDC::A_blk->clear();
        
        Compressed_Format<Weight>::nrows = nrows_; 
        Compressed_Format<Weight>::ncols = ncols_;
    }
}

template<typename Weight>
void UDC<Weight>::populate_spa(Weight** spa, const Weight* bias, const uint32_t col, uint64_t& index, Weight(*activation_function)(Weight), const int32_t tid) {
    uint64_t&  k = index;
    uint32_t   c = col + 1;
    Weight*    A = UDC::A_blk->ptr;
    Weight*    s = *spa;
    const Weight* b = bias;
    
    for(uint32_t i = 0; i < UDC::nrows; i++) {
        if(s[i]) {
            s[i] += b[c-1];
            s[i] = activation_function(s[i]);
            if(s[i]) {
                A[k] = s[i];
                s[i] = 0;
            }
        }
        k++;
    }
}

template<typename Weight>
void UDC<Weight>::walk_dxm(const bool one_rank, const int32_t leader_tid, const int32_t tid) {
    uint32_t nrows = UDC::nrows;
    uint32_t ncols = UDC::ncols;
    Weight*    A = UDC::A_blk->ptr;
    
    double&   checksum   = Env::counters[tid].checksum;
    uint64_t& checkcount = Env::counters[tid].checkcount;
    
    checksum   = 0;
    checkcount = 0;    
    
    for(uint64_t k = 0; k < nrows * ncols; k++) {
        if(A[k]) {
            checksum += A[k];
            checkcount++;
        }
    }

    Env::barrier();
    pthread_barrier_wait(&Env::thread_barrier);
    if(tid == leader_tid) {
        double     sum_threads = 0;
        uint64_t count_threads = 0;

        for(auto it = Env::counters.begin(); it != Env::counters.end(); it++) {
            sum_threads   += (*it).checksum;
            count_threads += (*it).checkcount;
        }
        
        if(one_rank) {
            Logging::print(Logging::LOG_LEVEL::INFO, "CSC: Iteration=%d, Total checksum=%f, Total count=%d\n", Env::iteration, sum_threads, count_threads);
        }
        else {
            double     sum_ranks = 0;
            uint64_t count_ranks = 0;
            
            MPI_Allreduce(&sum_threads, &sum_ranks, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&count_threads, &count_ranks, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
            
            Logging::print(Logging::LOG_LEVEL::INFO, "CSC: Iteration=%d, Total checksum=%f, Total count=%d\n", Env::iteration, sum_ranks, count_ranks);
        }
    }
}

#endif