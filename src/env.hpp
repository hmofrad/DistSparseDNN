/*
 * env.hpp: MPI runtime environment
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef ENV_HPP
#define ENV_HPP


#include <mpi.h>
#include <omp.h>
#include <numa.h>
#include <thread>
#include <stdarg.h>
#include <unistd.h>

#include "types.hpp"

namespace Env {
    int nranks = 0;
    int rank = 0;
    int nthreads = 0;
    const uint64_t PAGE_SIZE = sysconf(_SC_PAGESIZE);
    //double start_time = 0;
    //double end_time = 0;
    int iteration = 0;
    
    double io_time = 0;
    double spmm_sym_time = 0;
    double spmm_time = 0;
    double memory_time = 0;
    double exec_time = 0;
    double end_to_end_time = 0;
    
    std::vector<uint64_t> nnz_ranks;
    std::vector<uint64_t> nnz_i_ranks;
    std::vector<double>   time_ranks;
    
    std::vector<uint64_t> offset_nnz; /* Thread Offset from the beginning of the compressed format data */
    std::vector<uint64_t> index_nnz;  /* Current index of thread pointing to where the new data will be inserted */
    std::vector<uint32_t> displacement_nnz; /* The part that a thread may skip cuasing some internal fragmentation */  
    std::vector<uint32_t> start_col;
    std::vector<uint32_t> end_col;
    std::vector<double>   checksum;
    std::vector<uint64_t> checkcount;
    
    int init();
    void barrier();
    int finalize();
    void assign_col(uint32_t ncols, int32_t tid);
    uint64_t assign_nnz();
    double clock();
    double tic();
    double toc(double start_time);
    template<typename Type>
    std::tuple<Type, Type, Type, Type, Type> statistics(const Type time);
    template<typename Type>
    void stats(const std::vector<Type> vec, Type& sum, Type& mean, Type& std_dev, Type& min, Type& max);
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
    
    Env::nthreads = omp_get_max_threads(); 

    offset_nnz.resize(Env::nthreads);
    index_nnz.resize(Env::nthreads);
    displacement_nnz.resize(Env::nthreads);
    start_col.resize(Env::nthreads);
    end_col.resize(Env::nthreads);
    checksum.resize(Env::nthreads);
    checkcount.resize(Env::nthreads);
    
    MPI_Barrier(MPI_COMM_WORLD);  
    return(status);
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

void Env::assign_col(uint32_t ncols, int32_t tid) {
    Env::start_col[tid] = (ncols/Env::nthreads) * tid;
    Env::end_col[tid]   = (ncols/Env::nthreads) * (tid+1);
}


template<typename Type>
std::tuple<Type, Type, Type, Type, Type> Env::statistics(const Type time) {
    std::vector<Type> times(Env::nranks);
    MPI_Datatype MPI_TYPE = MPI_Types::get_mpi_data_type<Type>();
    MPI_Allgather(&time, 1, MPI_TYPE, times.data(), 1, MPI_TYPE, MPI_COMM_WORLD); 
    Type sum = 0.0, mean = 0.0, std_dev = 0.0, min = 0.0, max = 0.0;
    stats(times, sum, mean, std_dev, min, max);    
    return(std::make_tuple(sum, mean, std_dev, min, max));
    //Logging::print(Logging::LOG_LEVEL::INFO, "%s time (sec): min | avg +/- std_dev | max: %f | %f +/- %f | %f\n", str.c_str(), min, mean, std_dev, max);
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