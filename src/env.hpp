/*
 * env.hpp: MPI runtime environment
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef ENV_HPP
#define ENV_HPP

#include <stdarg.h>
#include <unistd.h>

#include <mpi.h>
#include <omp.h>
#include <thread>
#include <sys/sysinfo.h>
#include <numa.h>
//#include </ihome/rmelhem/moh18/numactl/libnuma/usr/local/include/numa.h> 
#include "types.hpp"


namespace Env {
    int nranks = 0;
    int rank = 0;
    int nthreads = 0;
    int ncores = 0;
    int nsockets = 0;
    int ncores_per_socket = 0;
    int rank_core_id = 0;
    int rank_socket_id = 0;
    std::vector<int> threads_core_id;
    std::vector<int> threads_socket_id;
    int num_unique_cores = 0;
    //std::vector<int> Env::core_ids_unique;
    const uint64_t PAGE_SIZE = sysconf(_SC_PAGESIZE);
    bool NUMA_ALLOC = false;
    
    std::vector<uint32_t> tile_index;
    std::vector<struct thread_struct> threads;
    std::vector<struct counter_struct> counters; 
    
    int iteration = 0;
    
    double io_time = 0;
    double spmm_sym_time = 0;
    double spmm_time = 0;
    double memory_time = 0;
    double exec_time = 0;
    double end_to_end_time = 0;
    
    std::vector<uint64_t> nnz_ranks;
    std::vector<std::vector<uint64_t>> nnz_threads;
    std::vector<uint64_t> nnz_i_ranks;
    std::vector<double>   time_ranks;
    
    std::vector<uint64_t> nnz_mean_thread_ranks;
    std::vector<uint64_t> nnz_std_dev_thread_ranks;
    std::vector<uint64_t> nnz_min_thread_ranks;
    std::vector<uint64_t> nnz_max_thread_ranks;
    std::vector<uint64_t> nnz_i_mean_thread_ranks;
    std::vector<uint64_t> nnz_i_std_dev_thread_ranks;
    std::vector<uint64_t> nnz_i_min_thread_ranks;
    std::vector<uint64_t> nnz_i_max_thread_ranks;
    
    std::vector<uint64_t> offset_nnz; /* Thread Offset from the beginning of the compressed format data */
    std::vector<uint64_t> index_nnz;  /* Current index of thread pointing to where the new data will be inserted */
    std::vector<uint32_t> displacement_nnz; /* The part that a thread may skip cuasing some internal fragmentation */  
    std::vector<uint64_t> count_nnz;
    std::vector<uint64_t> count_nnz_i;
    std::vector<uint32_t> start_col;
    std::vector<uint32_t> end_col;
    std::vector<uint32_t> start_row;
    std::vector<uint32_t> end_row;
    std::vector<uint64_t> start_nnz;
    std::vector<uint64_t> end_nnz;
    std::vector<double>   checksum;
    std::vector<uint64_t> checkcount;
    std::vector<uint64_t> checknnz;
    std::vector<bool>     checkconv;
    
    std::vector<double> spmm_symb_time;
    std::vector<double> spmm_real_time;
    std::vector<double> memory_allocation_time;
    std::vector<double> execution_time;
    
    pthread_barrier_t thread_barrier;
    std::vector<pthread_barrier_t> thread_barriers;
    std::vector<pthread_barrier_t> thread_barriers1;
    pthread_cond_t thread_cond; 
    pthread_mutex_t thread_mutex;
    std::vector<pthread_cond_t> thread_conds; 
    std::vector<pthread_mutex_t> thread_mutexes;
    std::vector<pthread_cond_t> thread_conds1; 
    std::vector<pthread_mutex_t> thread_mutexes1;
    std::vector<pthread_cond_t> thread_conds2; 
    std::vector<pthread_mutex_t> thread_mutexes2;
    std::vector<uint32_t> thread_counters;
    std::vector<uint32_t> num_follower_threads;
    std::vector<int32_t> follower_threads;
    std::vector<std::vector<struct helper_thread_info>> follower_threads_info;
    bool done;
    std::vector<int32_t> follower_to_leader;
    
    int init();
    void barrier();
    int finalize();
    
    //void assign_row(uint32_t nrows, int32_t tid);
    void assign_col(uint32_t ncols, int32_t tid);
    uint64_t assign_nnz();
    void assign_cols();
    double clock();
    double tic();
    double toc(double start_time);
    template<typename Type>
    std::tuple<Type, Type, Type, Type, Type> statistics(const Type value);
    template<typename Type>
    std::tuple<Type, Type, Type, Type, Type> statistics_t(const std::vector<Type> values_t);
    template<typename Type>
    void stats(const std::vector<Type> vec, Type& sum, Type& mean, Type& std_dev, Type& min, Type& max);
    int get_nsockets();
    bool numa_configure();
    bool set_thread_affinity(const int32_t tid);
    
    struct thread_struct {
        thread_struct(){};
        ~thread_struct(){};
        int32_t thread_id;
        uint32_t start_col;
        uint32_t end_col;
        uint32_t off_col; 
        uint64_t idx_nnz; // Index
        uint64_t off_nnz; // Offset 
        uint64_t dis_nnz; // Displacement 
    };
    
    struct helper_thread_info {
            helper_thread_info(){};
            helper_thread_info(int32_t thread_, uint32_t rowgroup_, uint32_t layer_, uint32_t start_col_, uint32_t end_col_) :
            thread(thread_), rowgroup(rowgroup_), layer(layer_), start_col(start_col_), end_col(end_col_) {};
           ~helper_thread_info(){};
            uint32_t thread;
            uint32_t rowgroup;
            uint32_t layer;
            uint32_t start_col;
            uint32_t end_col;
            uint64_t nnz;
    };
    
    struct counter_struct {
        double   checksum;
        uint64_t checkcount;
        uint64_t checknnz;
        bool     checkconv;
    };
    
    void adjust_displacement(const int32_t tid);
    void adjust_nnz(uint64_t& nnz, const int32_t leader_tid, const int32_t tid);
    //void pthread_barrier_wait_for_leader(int32_t leader, std::vector<int32_t> threads)

}

int Env::init() {
    int status = 0;
    int required = MPI_THREAD_MULTIPLE;
    int provided = -1;

    MPI_Init_thread(nullptr, nullptr, required, &provided);
    if((provided < MPI_THREAD_SINGLE) or (provided > MPI_THREAD_MULTIPLE)) {
        status = 1;
    } 

    MPI_Comm_size(MPI_COMM_WORLD, &Env::nranks);
    if(Env::nranks < 0) {
        status = 1;
    }
    
    MPI_Comm_rank(MPI_COMM_WORLD, &Env::rank);
    if(Env::rank < 0) {
        status = 1;
    }
    
    bool isNuma = numa_configure();
    if(not(Env::NUMA_ALLOC and isNuma)) {
        Env::NUMA_ALLOC = false;
    }
    
    pthread_barrier_init(&thread_barrier, NULL, Env::nthreads);

    offset_nnz.resize(Env::nthreads);
    index_nnz.resize(Env::nthreads);
    displacement_nnz.resize(Env::nthreads);
    start_col.resize(Env::nthreads);
    end_col.resize(Env::nthreads);
    start_nnz.resize(Env::nthreads);
    end_nnz.resize(Env::nthreads);
    checksum.resize(Env::nthreads);
    checkcount.resize(Env::nthreads);
    checknnz.resize(Env::nthreads);
    checkconv.resize(Env::nthreads);
    count_nnz.resize(Env::nthreads);
    count_nnz_i.resize(Env::nthreads);
    tile_index.resize(Env::nthreads);
    
    
    spmm_symb_time.resize(Env::nthreads);
    spmm_real_time.resize(Env::nthreads);
    memory_allocation_time.resize(Env::nthreads);
    execution_time.resize(Env::nthreads);
    nnz_threads.resize(Env::nthreads);
    
    thread_mutex = PTHREAD_MUTEX_INITIALIZER;
    thread_cond = PTHREAD_COND_INITIALIZER;
    thread_mutexes.resize(Env::nthreads);
    thread_conds.resize(Env::nthreads);
    thread_mutexes1.resize(Env::nthreads);
    thread_conds1.resize(Env::nthreads);
    thread_mutexes2.resize(Env::nthreads);
    thread_conds2.resize(Env::nthreads);
    thread_counters.resize(Env::nthreads);
    thread_barriers.resize(Env::nthreads);
    thread_barriers1.resize(Env::nthreads);
    follower_to_leader.resize(Env::nthreads, -1);
    //follower_to_leader.resize(Env::nthreads);
    for(int32_t i = 0; i < Env::nthreads; i++) {
        thread_mutexes[i] = PTHREAD_MUTEX_INITIALIZER;
        thread_conds[i] = PTHREAD_COND_INITIALIZER;
        thread_mutexes1[i] = PTHREAD_MUTEX_INITIALIZER;
        thread_conds1[i] = PTHREAD_COND_INITIALIZER;
        thread_mutexes2[i] = PTHREAD_MUTEX_INITIALIZER;
        thread_conds2[i] = PTHREAD_COND_INITIALIZER;
        thread_counters[i] = 0;
        //pthread_barrier_init(&thread_barriers[i], NULL, 1);
        //follower_to_leader[i] = i;
    }
    
    //done = false;
    
    num_follower_threads.resize(Env::nthreads);
    follower_threads_info.resize(Env::nthreads);
    for(int32_t i = 0; i < Env::nthreads; i++)
        follower_threads_info[i].resize(Env::nthreads);
    
    threads.resize(Env::nthreads);
    counters.resize(Env::nthreads);
    
    
    //for(uint32_t i = 0; i < Env::nthreads; i++) {
    //    threads[i].thread_id = i;
    //}
    //std::iota(threads.begin(),threads.end(),0);

    
    //pthread_mutex_lock(&mutex);
    //pthread_mutex_unlock(&mutex);

    MPI_Barrier(MPI_COMM_WORLD);  
    return(status);
}


int Env::get_nsockets() {
    const char* command = "lscpu | grep 'Socket(s)' | sed 's/[^0-9]*//g'";
    
    FILE* fid = popen(command, "r");
    if(not fid) {
        //printf("ERROR[rank=%d] Cannot get the number of sockets.\n", Env::rank);
        return(-1);
    }
    
    char c = 0;
    int  n = fread(&c, sizeof(c), 1, fid);
    if(n != 1) {
        //printf("ERROR[rank=%d] Cannot read the number of sockets.\n", Env::rank);
        return(-1);
    }
    pclose(fid);
    
    int nsockets = (int)(c - '0');
    return(nsockets);
}

bool Env::set_thread_affinity(const int32_t tid) {
    int cid = Env::threads_core_id[tid % Env::num_unique_cores];
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cid, &cpuset);
    pthread_t current_thread = pthread_self();    
    int ret = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    return(ret == 0);
}

bool Env::numa_configure() {
    bool status = true;
    Env::nthreads = omp_get_max_threads(); 
    
    Env::ncores = get_nprocs_conf();
    Env::nsockets = Env::get_nsockets();
    if(Env::nsockets < 1) {
        Env::nsockets = 1;
        status = false;
    }
    Env::ncores_per_socket = Env::ncores / Env::nsockets;
    
    Env::rank_core_id = sched_getcpu();
    Env::rank_socket_id = Env::rank_core_id / Env::ncores_per_socket;
    
    Env::threads_core_id.resize(Env::nthreads);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Env::threads_core_id[tid] = sched_getcpu();
        if(Env::threads_core_id[tid] == -1) {
            Env::threads_core_id[tid] = 0;
        }
    }
    std::sort(Env::threads_core_id.begin(), Env::threads_core_id.end());
    Env::threads_core_id.erase(std::unique(Env::threads_core_id.begin(), Env::threads_core_id.end()), Env::threads_core_id.end());
    num_unique_cores = Env::threads_core_id.size();
    
    Env::threads_socket_id.resize(Env::nthreads);
    for(int i = 0; i < Env::nthreads; i++) {
        int cid = Env::threads_core_id[i % Env::num_unique_cores];
        Env::threads_socket_id[i] = cid / Env::ncores_per_socket;
    }
    
    if(Env::nthreads != num_unique_cores) {
        //printf("WARN[rank=%d] CPU oversubscription %d/%d.\n", Env::rank, num_unique_cores, Env::nthreads);
        status = false;
    }
    
    return(status);
}

void Env::assign_col(uint32_t ncols, int32_t tid) {
    //if(refine)
      //  Env::start_col[tid] = ((ncols/Env::nthreads) *  tid  )+1;
    //else 
    Env::start_col[tid] = ((ncols/Env::nthreads) *  tid  );
    Env::end_col[tid]   =  (ncols/Env::nthreads) * (tid+1);
}

uint64_t Env::assign_nnz() {
    uint64_t nnz = std::accumulate(Env::offset_nnz.begin(), Env::offset_nnz.end(), 0);
    
    uint64_t sum = 0;
    for(int32_t i = Env::nthreads - 1; i > 0; i--) {
        sum += Env::offset_nnz[i];
        Env::offset_nnz[i] = nnz - sum;
        Env::index_nnz[i] = Env::offset_nnz[i];
    }
    Env::offset_nnz[0] = 0;                               
    Env::index_nnz[0] = 0;
    return(nnz);
}

void Env::adjust_nnz(uint64_t& nnz, const int32_t leader_tid, const int32_t tid) {
    if(tid == leader_tid) {
        nnz = 0;
        //std::vector<struct Env::thread_struct>::iterator it;
        for(auto it = Env::threads.begin(); it != Env::threads.end(); it++) {
            nnz += (*it).off_nnz;
        }
        
        uint64_t sum = 0;
        //std::vector<struct Env::thread_struct>::reverse_iterator rit;
        for (auto rit = Env::threads.rbegin(); rit != Env::threads.rend()-1; rit++) {
            sum += (*rit).off_nnz;
            (*rit).off_nnz = nnz - sum;
            (*rit).idx_nnz = (*rit).off_nnz;
        }
        Env::threads[0].idx_nnz = 0;
        Env::threads[0].off_nnz = 0;
    }
}

void Env::adjust_displacement(const int32_t tid) {
    Env::threads[tid].dis_nnz = (tid == 0) ? 0 : Env::threads[tid].off_nnz - Env::threads[tid-1].idx_nnz;
}

/*
void Env::assign_row(uint32_t nrows, int32_t tid) {
    Env::start_row[tid] = (nrows/Env::nthreads) *  tid;
    Env::end_row[tid]   = (tid == (Env::nthreads - 1)) ? nrows : (nrows/Env::nthreads) * (tid + 1);
}



void Env::assign_col(uint32_t ncols, int32_t tid) {
    Env::start_col[tid] = 0;
    Env::end_col[tid]   =  ncols;
}

uint64_t Env::assign_nnz() {
    uint64_t nnz = std::accumulate(Env::offset_nnz.begin(), Env::offset_nnz.end(), 0);

    uint64_t sum = 0;
    for(int32_t i = Env::nthreads - 1; i > 0; i--) {
        sum += Env::offset_nnz[i];
        Env::offset_nnz[i] = nnz - sum;
        Env::index_nnz[i] = Env::offset_nnz[i];
    }
    Env::offset_nnz[0] = 0;                               
    Env::index_nnz[0] = 0;

    return(nnz);
}
*/
template<typename Type>
std::tuple<Type, Type, Type, Type, Type> Env::statistics(const Type value) {
    std::vector<Type> values(Env::nranks);
    MPI_Datatype MPI_TYPE = MPI_Types::get_mpi_data_type<Type>();
    MPI_Allgather(&value, 1, MPI_TYPE, values.data(), 1, MPI_TYPE, MPI_COMM_WORLD); 
    Type sum = 0.0, mean = 0.0, std_dev = 0.0, min = 0.0, max = 0.0;
    stats(values, sum, mean, std_dev, min, max);    
    return(std::make_tuple(sum, mean, std_dev, min, max));
}

template<typename Type>
std::tuple<Type, Type, Type, Type, Type> Env::statistics_t(const std::vector<Type> values_t) {
    std::vector<Type> values(Env::nranks * Env::nthreads);
    MPI_Datatype MPI_TYPE = MPI_Types::get_mpi_data_type<Type>();
    MPI_Allgather(values_t.data(), Env::nthreads, MPI_TYPE, values.data(), Env::nthreads, MPI_TYPE, MPI_COMM_WORLD); 
    Type sum = 0.0, mean = 0.0, std_dev = 0.0, min = 0.0, max = 0.0;
    stats(values, sum, mean, std_dev, min, max);    
    return(std::make_tuple(sum, mean, std_dev, min, max));
}

template<typename Type>
void Env::stats(const std::vector<Type> vec, Type& sum, Type& mean, Type& std_dev, Type& min, Type& max) {
    sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    mean = sum / vec.size();
    Type sq_sum = std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0);
    std_dev = std::sqrt(sq_sum / vec.size() - mean * mean);
    std::pair bounds = std::minmax_element(vec.begin(), vec.end());
    min = *bounds.first;
    max = *bounds.second;
}

double Env::clock() {
    return(MPI_Wtime());
}

double Env::tic() {
    return(Env::clock());
}

double Env::toc(double start_time) {
    return(Env::clock() - start_time);
}

void Env::barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
}

int Env::finalize() {
    MPI_Barrier(MPI_COMM_WORLD);
    int ret = MPI_Finalize();
    return((ret == MPI_SUCCESS) ? 0 : 1);
}

#endif