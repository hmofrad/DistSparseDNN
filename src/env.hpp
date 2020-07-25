/*
 * env.hpp: MPI runtime environment
 * (c) Mohammad Hasanzadeh Mofrad, 2020
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
//#include <numa.h>
#include </ihome/rmelhem/moh18/numactl/libnuma/usr/local/include/numa.h> 
#include "types.hpp"

namespace Env {
    int nranks = 0;
    int rank = 0;
    int nthreads = 0;
    int ncores = 0;
    int nsockets = 0;
    int ncores_per_socket = 0;
    int nmachines = 0;;
    int nranks_per_machine = 0;
    std::vector<uint32_t> nthreads_per_socket;
    int rank_core_id = 0;
    int rank_socket_id = 0;
    std::vector<int> threads_core_id;
    std::vector<int> threads_socket_id;
    int num_unique_cores = 0;
    const uint64_t PAGE_SIZE = sysconf(_SC_PAGESIZE);  
    const uint64_t L1_ICACHE_SIZE = sysconf(_SC_LEVEL1_ICACHE_SIZE);
    const uint64_t L1_DCACHE_SIZE = sysconf(_SC_LEVEL1_DCACHE_SIZE);
    const uint64_t L2_CACHE_SIZE = sysconf(_SC_LEVEL2_CACHE_SIZE);
    const uint64_t L3_CACHE_SIZE = sysconf(_SC_LEVEL3_CACHE_SIZE);
    bool NUMA_ALLOC = true; 
    int32_t ALL_DONE = 0;

    std::vector<uint32_t> thread_rowgroup;
    std::vector<std::deque<uint32_t>> threads_rowgroups;
    std::deque<uint32_t> rank_rowgroups;
    std::deque<uint32_t> processed_rowgroups;
    std::vector<std::deque<uint32_t>> processed_rowgroups_per_thread;
    std::vector<struct counter_struct> counters; 
    std::vector<std::vector<uint32_t>> scores;
    std::deque<std::deque<int32_t>> my_threads;
    
    int iteration = 0;
    
    double io_time;
    double end_to_end_time;
    std::vector<double> spmm_symb_time;
    std::vector<double> spmm_real_time;
    std::vector<double> memory_allocation_time;
    std::vector<double> execution_time;
    std::vector<double> hybrid_probe_time;
    
    pthread_barrier_t thread_barrier;
    std::vector<pthread_barrier_t> thread_barriers;
    pthread_cond_t thread_cond; 
    pthread_mutex_t thread_mutex;
    std::vector<pthread_cond_t> thread_conds; 
    std::vector<pthread_mutex_t> thread_mutexes;
    pthread_cond_t manager_cond; 
    
    std::deque<int32_t> follower_threads;

    uint32_t num_finished_threads = 0;
    std::vector<bool> finished_threads;

    std::vector<uint32_t> numa_num_finished_threads;
    std::vector<std::deque<int32_t>> numa_follower_threads;
    std::vector<pthread_cond_t> numa_thread_cond;
    std::vector<pthread_mutex_t> numa_thread_mutex;
    
    struct thread_struct {
        thread_struct(){};
        ~thread_struct(){};
        uint32_t index;
        int32_t leader;
        uint32_t rowgroup;
        uint32_t start_layer;
        uint32_t current_layer;
        uint32_t start_row;
        uint32_t end_row;
        uint32_t start_col;
        uint32_t end_col;
        uint32_t off_col; 
        uint32_t off; 
        uint64_t idx_nnz; /* Current index of thread pointing to where the new data will be inserted */
        uint64_t off_nnz; /* Thread Offset from the beginning of the compressed format data */ 
        uint64_t dis_nnz; /* The part that a thread may skip cuasing some internal fragmentation */
    };

    struct counter_struct {
        double   checksum;
        uint64_t checkcount;
        uint64_t checknnz;
        bool     checkconv;
        uint32_t layer_index;
        uint32_t score;
    };
    
    int init();
    void read_options(int argc, char** argv, std::string input, std::string network, 
                      uint32_t& input_ninstances, uint32_t& input_nfeatures, uint32_t& ncategories, std::string& input_path,
                      uint32_t& nneurons, uint32_t& nmax_layers, std::string& layers_path,  
                      uint32_t& ci,  uint32_t& cl, uint32_t& p, uint32_t& h);
    void barrier();
    int finalize();
    
    double clock();
    double tic();
    double toc(double start_time);
    template<typename Type>
    std::tuple<Type, Type, Type, Type, Type> statistics(const Type value);
    template<typename Type>
    void stats(const std::vector<Type> vec, Type& min, Type& max, Type& mean, Type& std_dev, Type& sum);
    int get_nsockets();
    bool numa_configure();
    bool set_thread_affinity(const int32_t tid);
    int32_t get_socket_id(const int32_t tid);
    
    int get_num_machines();
    
    uint64_t adjust_nnz(const int32_t leader_tid, const int32_t tid);
    uint64_t adjust_nnz(const std::deque<int32_t> my_threads, const int32_t leader_tid, const int32_t tid);
    void adjust_displacement(const int32_t tid);
    void adjust_displacement(const std::deque<int32_t> my_threads, const int32_t leader_tid, const int32_t tid);
    
    std::vector<struct thread_struct> threads;
    std::vector<uint32_t> num_threads;
    void     init_num_threads(const uint32_t value, const int32_t leader_tid, const int32_t tid);
    void increase_num_threads(const uint32_t value, const int32_t leader_tid, const int32_t tid);
    void decrease_num_threads(const uint32_t value, const int32_t leader_tid, const int32_t tid);
    
    std::vector<std::pair<uint32_t, uint32_t>> queue_indices;
    
    pthread_mutex_t thread_mutex_q;
    std::vector<pthread_mutex_t> thread_mutexes_qs;
    pthread_mutex_t manager_mutex;
}

int Env::init() {
    int status = 0;
    int required = MPI_THREAD_MULTIPLE;
    int provided = -1;

    MPI_Init_thread(nullptr, nullptr, required, &provided);
    if((provided < MPI_THREAD_SINGLE) or (provided > MPI_THREAD_MULTIPLE)) { status = 1; } 

    MPI_Comm_size(MPI_COMM_WORLD, &Env::nranks);
    if(Env::nranks < 0) { status = 1; }
    
    MPI_Comm_rank(MPI_COMM_WORLD, &Env::rank);
    if(Env::rank < 0) { status = 1; }
    
    bool isNuma = numa_configure();
    if(not(Env::NUMA_ALLOC and isNuma)) { Env::NUMA_ALLOC = false; }
    
    processed_rowgroups_per_thread.resize(Env::nthreads);
    counters.resize(Env::nthreads); 
    
    scores.resize(Env::nsockets);
    for(int32_t s = 0; s < Env::nsockets; s++) { 
        scores[s].resize(Env::nthreads); 
    }
    
    spmm_symb_time.resize(Env::nthreads);
    spmm_real_time.resize(Env::nthreads);
    memory_allocation_time.resize(Env::nthreads);
    execution_time.resize(Env::nthreads);
    hybrid_probe_time.resize(Env::nthreads);
    
    pthread_barrier_init(&Env::thread_barrier, NULL, Env::nthreads);
    Env::thread_mutex = PTHREAD_MUTEX_INITIALIZER;
    Env::thread_cond = PTHREAD_COND_INITIALIZER;
    
    Env::thread_mutex_q = PTHREAD_MUTEX_INITIALIZER;
    Env::manager_mutex = PTHREAD_MUTEX_INITIALIZER;
    
    Env::thread_mutexes_qs.resize(Env::nthreads);
    for(int32_t i = 0; i < Env::nthreads; i++) { 
        Env::thread_mutexes_qs[i] = PTHREAD_MUTEX_INITIALIZER; 
    }
    
    Env::my_threads.resize(Env::nthreads);
    
    Env::thread_mutexes.resize(Env::nthreads);
    Env::thread_conds.resize(Env::nthreads);
    Env::thread_barriers.resize(Env::nthreads);
    for(int32_t i = 0; i < Env::nthreads; i++) {
        Env::thread_mutexes[i] = PTHREAD_MUTEX_INITIALIZER;
        Env::thread_conds[i] = PTHREAD_COND_INITIALIZER;
        pthread_barrier_init(&Env::thread_barriers[i], NULL, 1);
    }
    
    numa_num_finished_threads.resize(Env::nsockets);
    numa_follower_threads.resize(Env::nsockets);
    numa_thread_mutex.resize(Env::nsockets);
    numa_thread_cond.resize(Env::nsockets);
    
    
    for(int32_t i = 0; i < Env::nsockets; i++) {
        Env::numa_thread_mutex[i] = PTHREAD_MUTEX_INITIALIZER;
        Env::numa_thread_cond[i] = PTHREAD_COND_INITIALIZER;
    }
    
    threads.resize(Env::nthreads);
    num_threads.resize(Env::nthreads);
    for(int32_t i = 0; i < Env::nthreads; i++) {
        init_num_threads(0, i, i);
    }
    
    Env::nmachines = Env::get_num_machines();
    Env::nranks_per_machine = Env::nranks / Env::nmachines; 
    
    if(Env::rank==0) {
        printf("INFO[rank=%d] #Machines = %d, #MPI ranks  = %d, #Threads per rank = %d\n", Env::rank, Env::nmachines, Env::nranks, Env::nthreads);
        printf("INFO[rank=%d] #Sockets  = %d, #Processors = %d\n", Env::rank, Env::nsockets, Env::ncores);
        std::string numa = Env::NUMA_ALLOC ? "enabled" : "disabled";
        printf("INFO[rank=%d] NUMA is %s.\n", Env::rank, numa.c_str()); 
    }
    
    MPI_Barrier(MPI_COMM_WORLD);  
    return(status);
}

void Env::read_options(int argc, char** argv, std::string input, std::string network, 
                       uint32_t& input_ninstances, uint32_t& input_nfeatures, uint32_t& ncategories, std::string& input_path,
                       uint32_t& nneurons, uint32_t& nmax_layers, std::string& layers_path,  
                       uint32_t& ci,  uint32_t& cl, uint32_t& p, uint32_t& h) { 
   if(argc != 17) {
        printf("USAGE = %s -i <input_ninstances input_nfeatures ncategories input_path> -n <nneurons nmax_layers layers_path> -c <compression_type[0-4]> -m <multiplication_type[0-3]> -p <parallelism_type[0-4]> -h <hashing_type[0-3]>\n", argv[0]);
        std::exit(Env::finalize());     
    }
    
    if(Env::rank==0) { printf("INFO[rank=%d] %s Sparse DNN for %s dataset\n", Env::rank, network.c_str(), input.c_str()); }
    input_ninstances = atoi(argv[2]);
    input_nfeatures = atoi(argv[3]);
    ncategories = atoi(argv[4]);
    input_path = ((std::string) argv[5]);
    nneurons = atoi(argv[7]);
    nmax_layers = atoi(argv[8]);
    layers_path = ((std::string) argv[9]);
    ci = atoi(argv[11]);
    cl = atoi(argv[12]);
    p = atoi(argv[14]);
    h = atoi(argv[16]);
    if(Env::rank==0) {
        printf("INFO[rank=%d] Input [dim=%dx%d][ncategories=%d][path=%s]\n", Env::rank, input_ninstances, input_nfeatures, ncategories, input_path.c_str());
        printf("INFO[rank=%d] Layer [dim=%dx%d][nmax_layers=%d][path=%s]\n", Env::rank, nneurons, nneurons, nmax_layers, layers_path.c_str());
        printf("INFO[rank=%d] Config[compression=%dx%d][parallelism=%d][hashing=%d]\n", Env::rank, ci, cl, p, h);
    }
}


int Env::get_num_machines() {
    int num_machines = 0;
    char core_name[MPI_MAX_PROCESSOR_NAME];
    memset(core_name, '\0', MPI_MAX_PROCESSOR_NAME*sizeof(char));

    int cpu_name_len;
    MPI_Get_processor_name(core_name, &cpu_name_len);

    int total_length = MPI_MAX_PROCESSOR_NAME * Env::nranks; 
    std::string total_string(total_length, '\0');
    MPI_Allgather(core_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, (void*) total_string.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);
    
    // Tokenizing the string!
    int offset = 0;
    std::vector<std::string> machines_all;
    for(int i = 0; i < nranks; i++) {
        machines_all.push_back(total_string.substr(offset, MPI_MAX_PROCESSOR_NAME));
        offset += MPI_MAX_PROCESSOR_NAME;
    }

    // Find unique machines
    std::vector<std::string> machines = machines_all; 
    sort(machines.begin(), machines.end());
    machines.erase(unique(machines.begin(), machines.end()), machines.end()); 
    num_machines = machines.size();

    return(num_machines);
}

int Env::get_nsockets() {
    const char* command = "lscpu | grep 'Socket(s)' | sed 's/[^0-9]*//g'";
    
    FILE* fid = popen(command, "r");
    if(not fid) { return(-1); }
    
    char c = 0;
    int  n = fread(&c, sizeof(c), 1, fid);
    if(n != 1) { return(-1); }
    pclose(fid);
    
    int nsockets = (int)(c - '0');
    return(nsockets);
}

int32_t Env::get_socket_id(const int32_t tid) {
    return(Env::threads_core_id[tid % Env::num_unique_cores]/Env::ncores_per_socket);
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
    
    Env::nthreads_per_socket.resize(Env::nsockets);
    for(int i = 0; i < Env::nthreads; i++) {
        Env::nthreads_per_socket[Env::threads_socket_id[i]]++;
    }
    
    queue_indices.resize(Env::nsockets);
    bool balanced = true;
    for(uint32_t n_th_per_socket: Env::nthreads_per_socket) {
        if(n_th_per_socket != (uint32_t) Env::ncores_per_socket) {
            balanced = false;
            break;
        }
    }

    if(balanced) {
        for(int s = 0; s < Env::nsockets; s++) {
            queue_indices[s] = {Env::ncores_per_socket * s, Env::ncores_per_socket * (s+1)};
        }
    }
    else {
        for(int s = 0; s < Env::nsockets; s++) {
            queue_indices[s] = {0, Env::nthreads};
        }
    }
  
    
    if(Env::nthreads != num_unique_cores) {
        //printf("WARN[rank=%d] CPU oversubscription %d/%d.\n", Env::rank, num_unique_cores, Env::nthreads);
        status = false;
    }
    
    return(status);
}

template<typename Type>
std::tuple<Type, Type, Type, Type, Type> Env::statistics(const Type value) {
    std::vector<Type> values(Env::nranks);
    MPI_Datatype MPI_TYPE = MPI_Types::get_mpi_data_type<Type>();
    MPI_Allgather(&value, 1, MPI_TYPE, values.data(), 1, MPI_TYPE, MPI_COMM_WORLD); 
    Type min = 0.0, max = 0.0, mean = 0.0, std_dev = 0.0, sum = 0.0;
    stats(values, min, max, mean, std_dev, sum);
    return(std::make_tuple( min, max, mean, std_dev, sum));
}

template<typename Type>
void Env::stats(const std::vector<Type> vec,  Type& min, Type& max, Type& mean, Type& std_dev, Type& sum) {
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

uint64_t Env::adjust_nnz(const int32_t leader_tid, const int32_t tid) {
    uint64_t nnz = 0;
    if(tid == leader_tid) {
        for(auto it = Env::threads.begin(); it != Env::threads.end(); it++) {
            nnz += (*it).off_nnz;
        }
        
        uint64_t sum = 0;
        for (auto rit = Env::threads.rbegin(); rit != Env::threads.rend()-1; rit++) {
            sum += (*rit).off_nnz;
            (*rit).off_nnz = nnz - sum;
            (*rit).idx_nnz = (*rit).off_nnz;
        }
        Env::threads[0].idx_nnz = 0;
        Env::threads[0].off_nnz = 0;
    }
    return(nnz);
}

uint64_t Env::adjust_nnz(const std::deque<int32_t> my_threads, const int32_t leader_tid, const int32_t tid) {
    uint64_t nnz = 0;
    if(tid == leader_tid) {
        for(auto t: my_threads) {
            nnz += Env::threads[t].off_nnz;
        }
        
        uint64_t sum = 0;
        for(int32_t i = my_threads.size() - 1; i > 0; i--) {
            int32_t t = my_threads[i];
            sum += Env::threads[t].off_nnz;
            Env::threads[t].off_nnz = nnz - sum;
            Env::threads[t].idx_nnz = Env::threads[t].off_nnz;
        }
        Env::threads[tid].off_nnz = 0;                               
        Env::threads[tid].idx_nnz = 0;
    }
    pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
    return(nnz);
}

void Env::adjust_displacement(const int32_t tid) {
    Env::threads[tid].dis_nnz = (tid == 0) ? 0 : Env::threads[tid].off_nnz - Env::threads[tid-1].idx_nnz;
}

void Env::adjust_displacement(const std::deque<int32_t> my_threads, const int32_t leader_tid, const int32_t tid) {
    if(tid == leader_tid) {
        Env::threads[tid].dis_nnz = 0;
        for(uint32_t i = 1; i < my_threads.size(); i++) {    
            int32_t t_minus_1 = my_threads[i-1];
            int32_t t = my_threads[i];
            Env::threads[t].dis_nnz = Env::threads[t].off_nnz - Env::threads[t_minus_1].idx_nnz;
        }
    }
    pthread_barrier_wait(&Env::thread_barriers[leader_tid]);
}

void Env::init_num_threads(const uint32_t value, const int32_t leader_tid, const int32_t tid) {
    if(tid == leader_tid) {
        Env::num_threads[tid] = value;
    }
}

void Env::increase_num_threads(const uint32_t value, const int32_t leader_tid, const int32_t tid) {
    if(tid == leader_tid) {
        pthread_mutex_lock(&Env::thread_mutexes[tid]); 
        Env::num_threads[tid] += value;
        pthread_mutex_unlock(&Env::thread_mutexes[tid]); 
    }
}

void Env::decrease_num_threads(const uint32_t value, const int32_t leader_tid, const int32_t tid) {
    pthread_mutex_lock(&Env::thread_mutexes[leader_tid]); 
    Env::num_threads[leader_tid] -= value;
    if(Env::num_threads[leader_tid] == 0) {
        pthread_cond_broadcast(&Env::thread_conds[leader_tid]);   
    }
    else {
        pthread_cond_wait(&Env::thread_conds[leader_tid], &Env::thread_mutexes[leader_tid]); 
    }
    pthread_mutex_unlock(&Env::thread_mutexes[leader_tid]);
}
#endif
