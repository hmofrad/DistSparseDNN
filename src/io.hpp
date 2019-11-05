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

enum INPUT_TYPE {_TEXT_, _BINARY_};

namespace IO {
    template<typename Weight>
    std::tuple<uint64_t, uint32_t, uint32_t> text_file_stat(const std::string inputFile);
    template<typename Weight>
    void text_file_read(const std::string inputFile, std::vector<std::vector<struct Tile<Weight>>>& tiles, const uint32_t tile_height, const uint32_t tile_width, bool one_rank);
    void text_file_categories(const std::string inputFile, std::vector<uint32_t>& categories, const uint32_t tile_height);
    
    template<typename Weight>
    std::tuple<uint64_t, uint32_t, uint32_t> binary_file_stat(const std::string inputFile);
    template<typename Weight>
    void binary_file_read(const std::string inputFile, std::vector<std::vector<struct Tile<Weight>>>& tiles, const uint32_t tile_height, const uint32_t tile_width, bool one_rank);
    void binary_file_categories(const std::string inputFile, std::vector<uint32_t>& categories, const uint32_t tile_height);
}

template<typename Weight>
std::tuple<uint64_t, uint32_t, uint32_t> IO::text_file_stat(const std::string inputFile) {
    double start_time = Env::tic();
    Logging::print(Logging::LOG_LEVEL::INFO, "Read text: Start collecting info from the input file %s\n", inputFile.c_str());
    uint64_t nnz = 0;
    uint32_t nrows = 0;
    uint32_t ncols = 0;    

    
    std::ifstream fin(inputFile.c_str(), std::ios_base::in);
    if(not fin.is_open()) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Opening %s\n", inputFile.c_str());
        std::exit(Env::finalize());
    }
    
    struct Triple<Weight> triple;
    std::string line;
    std::istringstream iss;
    while (std::getline(fin, line)) {
        iss.clear();
        iss.str(line);
        iss >> triple.row >> triple.col >> triple.weight;
        nrows = (nrows < triple.row) ? triple.row : nrows;
        ncols = (ncols < triple.col) ? triple.col : ncols;
        nnz++;
    }
    fin.close();
    Logging::print(Logging::LOG_LEVEL::INFO, "Read text: Done  collecting info from the input file %s\n", inputFile.c_str());
    Env::barrier();
    
    Env::io_time += Env::toc(start_time);
    
    return std::make_tuple(nnz, nrows + 1, ncols + 1);
}

template<typename Weight>
void IO::text_file_read(const std::string inputFile, std::vector<std::vector<struct Tile<Weight>>>& tiles, const uint32_t tile_height, const uint32_t tile_width, bool one_rank) {
    double start_time = Env::tic();
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Read text: Start reading the input file %s\n", inputFile.c_str());
    
    std::ifstream fin(inputFile.c_str(), std::ios_base::in);
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
    
    if(one_rank) {
        share = nlines;
        start_line = 0;
        end_line = nlines;
        curr_line = 0;
        fin.clear();
        fin.seekg(0, std::ios_base::beg);
    }
    
    std::vector<struct Triple<Weight>> triples(share);
    #pragma omp parallel
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
            triples[curr_line_t - curr_line] = triple;
            curr_line_t++;
        }
        
        fin_t.close();
    }
    fin.close();

    for(auto& triple: triples) {
        std::pair pair = std::make_pair((triple.row / tile_height), (triple.col / tile_width));
        tiles[pair.first][pair.second].triples.push_back(triple);
    }

    Logging::print(Logging::LOG_LEVEL::INFO, "Read text: Done reading the input file %s\n", inputFile.c_str());
    Env::barrier();
    
    Env::io_time += Env::toc(start_time);
}

void IO::text_file_categories(const std::string inputFile, std::vector<uint32_t>& categories, const uint32_t tile_height) {
    double start_time = Env::tic();
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Read text: Start reading the category file %s\n", inputFile.c_str());
    
    std::ifstream fin(inputFile.c_str(), std::ios_base::in);
    if(not fin.is_open()) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Opening %s\n", inputFile.c_str());
        std::exit(Env::finalize());
    }

    uint32_t nlines = 0;
    std::string line;
    while (std::getline(fin, line)) {
        nlines++;
    }
    fin.clear();
    fin.seekg(0, std::ios_base::beg);
    
    uint32_t start = (!Env::rank) ? 0 : Env::rank * tile_height;
    uint32_t end = (Env::rank + 1) * tile_height;
    
    uint32_t nCategories_local = 0;
    uint32_t category = 0;
    std::istringstream iss;
    categories.resize(tile_height);
    while (std::getline(fin, line)) {
        iss.clear();
        iss.str(line);
        iss >> category;
        if((!Env::rank and (category <  end)) or
            (Env::rank and (category > start)
                       and (category < end))) {
            categories[category % tile_height] = 1;
            nCategories_local++;
        }
    }
    fin.close();

    uint32_t nCategories_global = 0;
    MPI_Allreduce(&nCategories_local, &nCategories_global, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
    Logging::print(Logging::LOG_LEVEL::INFO, "Read text: Total number of categories %d\n", nCategories_global);
    if(nlines != nCategories_global) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Reading %s\n", inputFile.c_str());
        std::exit(Env::finalize());
    }
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Read text: Done  reading the category file %s %d\n", inputFile.c_str());
    Env::barrier();
    
    Env::io_time += Env::toc(start_time);
} 
 
 
template<typename Weight>
std::tuple<uint64_t, uint32_t, uint32_t> IO::binary_file_stat(const std::string inputFile) {
    double start_time = Env::tic();
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Read binary: Start collecting info from the input file %s\n", inputFile.c_str());
    uint64_t nnz = 0;
    uint32_t nrows = 0;
    uint32_t ncols = 0;    
    
    
    std::ifstream fin(inputFile.c_str(), std::ios_base::binary);
    if(not fin.is_open()) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Opening %s\n", inputFile.c_str());
        std::exit(Env::finalize());
    }
    
    uint64_t filesize, offset = 0;
    fin.seekg (0, std::ios_base::end);
    filesize = (uint64_t) fin.tellg();
    fin.seekg(0, std::ios_base::beg);
    
    if(filesize % sizeof(struct Triple<Weight>)) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Reading %s\n", inputFile.c_str());
        std::exit(Env::finalize());
    }
    
    struct Triple<Weight> triple;
    while(offset < filesize) {
        fin.read(reinterpret_cast<char*>(&triple), sizeof(struct Triple<Weight>));
        offset += sizeof(struct Triple<Weight>);
        nrows = (nrows < triple.row) ? triple.row : nrows;
        ncols = (ncols < triple.col) ? triple.col : ncols;
        nnz++;
    }
    fin.close();
    Logging::print(Logging::LOG_LEVEL::INFO, "Read binary: Done  collecting info from the input file %s\n", inputFile.c_str());
    
    Env::io_time += Env::toc(start_time);
    
    Env::barrier();
    return std::make_tuple(nnz, nrows + 1, ncols + 1);
} 
 
template<typename Weight>
void IO::binary_file_read(const std::string inputFile, std::vector<std::vector<struct Tile<Weight>>>& tiles, const uint32_t tile_height, const uint32_t tile_width, bool one_rank) {
    double start_time = Env::tic();
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Read binary: Start reading the input file %s\n", inputFile.c_str());
    
    std::ifstream fin(inputFile.c_str(), std::ios_base::binary);
    if(not fin.is_open()) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Opening %s\n", inputFile.c_str());
        std::exit(Env::finalize());
    }
    
    uint64_t filesize, offset = 0;
    fin.seekg (0, std::ios_base::end);
    filesize = (uint64_t) fin.tellg();
    fin.clear();
    fin.close();
    
    if(filesize % sizeof(struct Triple<Weight>)) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Reading %s\n", inputFile.c_str());
        std::exit(Env::finalize());
    }
    
    uint64_t nTriples = filesize / sizeof(struct Triple<Weight>);
    Logging::print(Logging::LOG_LEVEL::INFO, "Read binary: File size is %lu bytes with %lu triples\n", filesize, nTriples);
    
    uint64_t share = (nTriples / Env::nranks) * sizeof(struct Triple<Weight>);
    
    uint64_t start_offset = Env::rank * share;
    uint64_t end_offset = (Env::rank != Env::nranks - 1) ? ((Env::rank + 1) * share) : filesize;
    share = (Env::rank == Env::nranks - 1) ? end_offset - start_offset : share;
    uint64_t share_tripels = share/sizeof(struct Triple<Weight>);
    
    if(one_rank) {
        share = filesize;
        start_offset = 0;
        end_offset = filesize;
        share_tripels = share/sizeof(struct Triple<Weight>);
    }
    
    std::vector<struct Triple<Weight>> triples(share_tripels);    
    #pragma omp parallel
    {
        int nthreads = Env::nthreads; 
        int tid = omp_get_thread_num();
        uint64_t nTriples_t = share / sizeof(struct Triple<Weight>);
        uint64_t share_t = (nTriples_t / nthreads) * sizeof(struct Triple<Weight>); 
        uint64_t start_offset_t = start_offset + (tid * share_t);
        uint64_t end_offset_t = (tid != Env::nthreads - 1) ? start_offset + ((tid + 1) * share_t) : end_offset;
        share_t = (tid == Env::nthreads - 1) ? end_offset_t - start_offset_t : share_t;
        uint64_t offset_t = start_offset_t;

        std::ifstream fin_t(inputFile.c_str(), std::ios_base::binary);
        if(not fin_t.is_open()) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Opening %s\n", inputFile.c_str());
            std::exit(Env::finalize());
        }
        fin_t.seekg(offset_t, std::ios_base::beg);
        
        struct Triple<Weight> triple;
        uint64_t index = 0;
        while(offset_t < end_offset_t) {
            fin_t.read(reinterpret_cast<char*>(&triple), sizeof(struct Triple<Weight>));
            index = ((offset_t - start_offset) / sizeof(struct Triple<Weight>));
            triples[index] = triple;
            offset_t += sizeof(struct Triple<Weight>);
        }
        fin_t.close();
    }

    for(auto& triple: triples) {
        std::pair pair = std::make_pair((triple.row / tile_height), (triple.col / tile_width));
        tiles[pair.first][pair.second].triples.push_back(triple);
    }

    Logging::print(Logging::LOG_LEVEL::INFO, "Read binary: Done reading the input file %s\n", inputFile.c_str());
    Env::barrier();
    
    Env::io_time += Env::toc(start_time);
}
 
 

void IO::binary_file_categories(const std::string inputFile, std::vector<uint32_t>& categories, const uint32_t tile_height) {
    double start_time = Env::tic();
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Read binary: Start reading the category file %s\n", inputFile.c_str());
    
    std::ifstream fin(inputFile.c_str(), std::ios_base::binary);
    if(not fin.is_open()) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Opening %s\n", inputFile.c_str());
        std::exit(Env::finalize());
    }
    
    uint64_t filesize, offset = 0;
    fin.seekg (0, std::ios_base::end);
    filesize = (uint64_t) fin.tellg();
    fin.seekg(0, std::ios_base::beg);
    
    if(filesize % sizeof(uint32_t)) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Reading %s\n", inputFile.c_str());
        std::exit(Env::finalize());
    }
    
    uint32_t start = (!Env::rank) ? 0 : Env::rank * tile_height;
    uint32_t end = (Env::rank + 1) * tile_height;
    
    uint32_t nCategories_local = 0;
    uint32_t category = 0;
    categories.resize(tile_height);
    while(offset < filesize) {
        fin.read(reinterpret_cast<char*>(&category), sizeof(uint32_t));
        offset += sizeof(uint32_t);
        if((!Env::rank and (category <  end)) or
            (Env::rank and (category > start)
                       and (category < end))) {
            categories[category % tile_height] = 1;
            nCategories_local++;
        }
    }
    fin.close();

    uint32_t nCategories_global = 0;
    MPI_Allreduce(&nCategories_local, &nCategories_global, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
    Logging::print(Logging::LOG_LEVEL::INFO, "Read binary: Total number of categories %d\n", nCategories_global);
    if((filesize / sizeof(uint32_t)) != nCategories_global) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Reading %s\n", inputFile.c_str());
        std::exit(Env::finalize());
    }
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Read binary: Done  reading the category file %s %d\n", inputFile.c_str());
    Env::barrier();
    
    Env::io_time += Env::toc(start_time);
}  
#endif