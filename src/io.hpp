/*
 * io.hpp: IO interface
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef IO_HPP
#define IO_HPP

#include <fstream>
#include <sstream>
#include <tuple>

#include "env.hpp"
#include "log.hpp"
#include "tile.hpp"
//#include "tiling.hpp"

namespace IO {
    template<typename Weight>
    std::tuple<uint32_t, uint32_t, uint64_t> get_text_info(std::string inputFile);
    //template<typename Weight>
    //std::tuple<uint64_t, uint64_t, uint64_t> read_text(std::string inputFile);
    template<typename Weight>
    void read_text_file(std::string inputFile, std::vector<std::vector<struct Tile<Weight>>>& tiles, uint32_t tile_height, uint32_t tile_width);
}

template<typename Weight>
std::tuple<uint32_t, uint32_t, uint64_t> IO::get_text_info(std::string inputFile) {
    uint32_t nrows = 0;
    uint32_t ncols = 0;    
    uint64_t nnz = 0;
    
    std::ifstream fin(inputFile.c_str());
    if(not fin.is_open()) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Opening %s\n", inputFile.c_str());
        std::exit(Env::finalize());
    }
    
    std::string line;
    struct Triple<Weight> triple;
    std::istringstream iss;
    while (std::getline(fin, line)) {
        iss.clear();
        iss.str(line);
        iss >> triple.row >> triple.col >> triple.weight;
        if(triple.row > nrows)
            nrows = triple.row;
        if(triple.col > ncols)
            ncols = triple.col;
        nnz++;
    }
    fin.close();
    return std::make_tuple(nrows + 1, ncols + 1, nnz);
}

template<typename Weight>
void IO::read_text_file(std::string inputFile, std::vector<std::vector<struct Tile<Weight>>>& tiles, uint32_t tile_height, uint32_t tile_width) {
    Logging::print(Logging::LOG_LEVEL::INFO, "Read text: Start reading the input file %s\n", inputFile.c_str());
    
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
    Logging::print(Logging::LOG_LEVEL::INFO, "Read text: File contains %lu lines\n", nlines);
    
    uint64_t share = nlines / Env::nranks;
    uint64_t start_line = Env::rank * share;
    uint64_t end_line = (Env::rank != Env::nranks - 1) ? ((Env::rank + 1) * share) : nlines;
    share = (Env::rank == Env::nranks - 1) ? end_line - start_line : share;
    uint64_t curr_line = 0;
    while(curr_line < start_line) {
        std::getline(fin, line);
        curr_line++;
    }
    
    //triples.resize(share);
    std::vector<struct Triple<Weight>> triples(share);
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

    for(auto& triple: triples) {
        std::pair pair = std::make_pair((triple.row / tile_height), (triple.col / tile_width));
        tiles[pair.first][pair.second].triples.push_back(triple);
    }
      

    /*  
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
    */
    Logging::print(Logging::LOG_LEVEL::INFO, "Read text: Done reading the input file %s\n", inputFile.c_str());
    Env::barrier();
 }
#endif