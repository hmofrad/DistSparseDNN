/*
 * io.hpp: IO interface
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef IO_HPP
#define IO_HPP

#include <fstream>
#include <sstream>
#include<tuple>

#include "env.hpp"
#include "log.hpp"
#include "triple.hpp"


namespace IO_interface {
    template<typename Weight>
    std::tuple<uint64_t, uint64_t, uint64_t> read_text(std::string inputFile, std::vector<struct Triple<Weight>>& triples);
}

template<typename Weight>
std::tuple<uint64_t, uint64_t, uint64_t> IO_interface::read_text(std::string inputFile, std::vector<struct Triple<Weight>>& triples) {
    uint64_t nrows = 0;
    uint64_t ncols = 0;    
    uint64_t nnz = 0;
    
    std::ifstream fin(inputFile.c_str());
    if(not fin.is_open()) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Opening %s\n", inputFile.c_str());
        std::exit(Env::finalize());
    }
    
    std::string line;
    
    uint64_t nlines = 0;
    while (std::getline(fin, line)) {
        nlines++;
    }
    fin.clear();
    fin.seekg(0, std::ios_base::beg);
    
    uint64_t share = nlines / Env::nranks;
    uint64_t start_line = Env::rank * share;
    uint64_t end_line = (Env::rank != Env::nranks - 1) ? ((Env::rank + 1) * share) : nlines;
    share = (Env::rank == Env::nranks - 1) ? end_line - start_line : share;
    uint64_t curr_line = 0;
    while(curr_line < start_line) {
        std::getline(fin, line);
        curr_line++;
    }
    
    triples.resize(share);
    #pragma omp parallel reduction(max : nrows, ncols)
    {
        int nthreads = Env::nthreads; 
        int tid = omp_get_thread_num();
        
        uint64_t share_t = share / nthreads; 
        uint64_t start_line_t = curr_line + (tid * share_t);
        uint64_t end_line_t = (tid != Env::nthreads - 1) ? curr_line + ((tid + 1) * share_t) : end_line;
        share_t = (tid == Env::nthreads - 1) ? end_line_t - start_line_t : share_t;
        uint64_t curr_line_t = 0;
        std::string line_t;
        std::ifstream fin_t(inputFile.c_str());
        if(not fin_t.is_open()) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Opening %s\n", inputFile.c_str());
            std::exit(Env::finalize());
        }
        fin_t.seekg(fin.tellg(), std::ios_base::beg);
        curr_line_t = curr_line;
        
        while(curr_line_t < start_line_t) {
            std::getline(fin_t, line_t);
            curr_line_t++;
        }
        
        struct Triple<Weight> triple;
        std::istringstream iss_t;
        while (curr_line_t < end_line_t) {
            std::getline(fin_t, line_t);
            iss_t.clear();
            iss_t.str(line_t);
            iss_t >> triple.row >> triple.col >> triple.weight;
            //long int d = (curr_line_t - curr_line);
            //printf("%d %d %lu %lu %lu\n", Env::rank, tid, curr_line, curr_line_t, d);
            triples[curr_line_t - curr_line] = triple;
            
            if(triple.row > nrows)
                nrows = triple.row;
            if(triple.col > ncols)
                ncols = triple.col;

            curr_line_t++;
        }
        
        fin_t.close();
    }
    fin.close();
        
    uint64_t reducer = 0;
    MPI_Allreduce(&ncols, &reducer, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);
    ncols = reducer;
    
    MPI_Allreduce(&nrows, &reducer, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);
    nrows = reducer;
    nnz = triples.size();
    MPI_Allreduce(&nnz, &reducer, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
    nnz = reducer;
    
    if(nlines != nnz) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Invalid number of input features %lu", nnz);
        std::exit(Env::finalize());
    }

    return std::make_tuple(nrows, ncols, nnz);
 }
#endif