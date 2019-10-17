/*
 * tiling.hpp: Tiling strategy
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef TILING_HPP
#define TILING_HPP

#include <fstream>
#include <sstream>
#include <numeric>
#include<tuple>

#include "triple.hpp"
//#include "tile.hpp"


template<typename Weight>
struct Tile{
    public:
        Tile() {};
        ~Tile() {};
        
        std::vector<struct Triple<Weight>> triples;
        
        int32_t rank;
};


enum TILING_TYPE {_1D_COL_, _1D_ROW_,_2D_};
const char* TILING_TYPES[] = {"_1D_COL_", "_1D_ROW_", "_2D_"};

template<typename Weight>
class Tiling {
    public:
        Tiling() {};
        ~Tiling() {};
        
        Tiling(TILING_TYPE tiling_type_, uint32_t ntiles_, uint32_t nrowgrps_, uint32_t ncolgrps_, uint32_t nranks_, std::string inputFile);
        //Tiling( TILING_TYPE tiling_type_, uint32_t ntiles_, uint32_t nrowgrps_, uint32_t ncolgrps_, uint32_t nranks_, uint32_t rank_nthreads_);
        
        TILING_TYPE tiling_type;        
        uint32_t ntiles, nrowgrps, ncolgrps;
        uint32_t nranks, rank_ntiles, rank_nrowgrps, rank_ncolgrps;
        uint32_t rowgrp_nranks, colgrp_nranks;
        uint32_t rank_nthreads;
        
        uint32_t nthreads, thread_ntiles, thread_nrowgrps, thread_ncolgrps;
        uint32_t rowgrp_nthreads, colgrp_nthreads;
        
        uint32_t nrows, ncols;
        uint64_t nnz;
        uint32_t tile_height, tile_width;
        
        std::vector<std::vector<struct Tile<Weight>>> tiles;
        void print_tiling(std::string field);
        bool assert_tiling();

        void integer_factorize(uint32_t n, uint32_t& a, uint32_t& b);
        
        std::tuple<uint64_t, uint64_t, uint64_t>  get_text_info(std::string inputFile);
        void read_text_file(std::string inputFile);
        
        void insert_triple(const struct Triple<Weight>& triple);
        
        
};

template<typename Weight>
void Tiling<Weight>::insert_triple(const struct Triple<Weight>& triple) {
    std::pair pair = std::make_pair((triple.row / tile_height), (triple.col / tile_width));
    //struct Triple<Weight> pair;
    //pair.row = (triple.row / tile_height);
    //pair.col = (triple.col / tile_width);
    //if((pair.first > nrowgrps) or (pair.second > ncolgrps) or (pair.first < 0) or (pair.second < 0) ) {
        //if(!Env::rank)
        //printf("rank=%d: Invalid entry for tile[%d][%d]=[%d %d]\n", Env::rank, pair.first, pair.second, triple.row, triple.col);
        //Env::exit(0);
    //}
    tiles[pair.first][pair.second].triples.push_back(triple);
    //tiles[pair.first][pair.second].triples.push_back(triple);
}

/* Process-based tiling based on MPI ranks*/ 
template<typename Weight>
Tiling<Weight>::Tiling(TILING_TYPE tiling_type_, uint32_t ntiles_, uint32_t nrowgrps_, uint32_t ncolgrps_, uint32_t nranks_, std::string inputFile) 
        :  tiling_type(tiling_type_), ntiles(ntiles_) , nrowgrps(nrowgrps_), ncolgrps(ncolgrps_) , nranks(nranks_)
        , rank_ntiles(ntiles_/nranks_){
           
    std::tie(nrows, ncols, nnz) = get_text_info(inputFile);
            
    if((rank_ntiles * nranks != ntiles) or (nrowgrps * ncolgrps != ntiles)) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
        std::exit(Env::finalize()); 
    }
    
    if ((tiling_type == TILING_TYPE::_1D_ROW_)) {
        rowgrp_nranks = 1;
        colgrp_nranks = nranks;
        if(rowgrp_nranks * colgrp_nranks != nranks) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
            std::exit(Env::finalize()); 
        }
    }
    else if(tiling_type == TILING_TYPE::_1D_COL_) {
        rowgrp_nranks = nranks;
        colgrp_nranks = 1;
        if(rowgrp_nranks * colgrp_nranks != nranks) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
            std::exit(Env::finalize()); 
        }
    }
    else if (tiling_type == TILING_TYPE::_2D_) {
        integer_factorize(nranks, rowgrp_nranks, colgrp_nranks);
        if(rowgrp_nranks * colgrp_nranks != nranks) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
            std::exit(Env::finalize()); 
        }
    }
    
    rank_nrowgrps = nrowgrps / colgrp_nranks;
    rank_ncolgrps = ncolgrps / rowgrp_nranks;        
    if(rank_nrowgrps * rank_ncolgrps != rank_ntiles) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
        std::exit(Env::finalize()); 
    }
    
    while(nrows % nrowgrps)
        nrows++;
    
    while(ncols % ncolgrps)
        ncols++;
    
    tile_height = nrows / nrowgrps;
    tile_width  = ncols / ncolgrps;
    
    
    tiles.resize(nrowgrps, std::vector<struct Tile<Weight>>(ncolgrps));
    int32_t gcd_r = std::gcd(rowgrp_nranks, colgrp_nranks);
    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j];
            tile.rank = (((i % colgrp_nranks) * rowgrp_nranks + (j % rowgrp_nranks)) + ((i / (nrowgrps/(gcd_r))) * (rank_nrowgrps))) % nranks;
        }
    }
    
    if(not assert_tiling()) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
        std::exit(Env::finalize()); 
    }
    
    print_tiling("rank");
    
    read_text_file(inputFile);
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: Process-based%s\n", TILING_TYPES[tiling_type]);
    print_tiling("rank");
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: nrowgrps      x ncolgrps      = [%d x %d]\n", nrowgrps, ncolgrps);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: rowgrp_nranks x colgrp_nranks = [%d x %d]\n", rowgrp_nranks, colgrp_nranks);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: rank_nrowgrps x rank_ncolgrps = [%d x %d]\n", rank_nrowgrps, rank_ncolgrps);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: nrows         x ncols         = [%d x %d]\n", nrows, ncols);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: tile_height   x tile_width    = [%d x %d]\n", tile_height, tile_width);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: nnz                           = [%d     ]\n", nnz);
}

template<typename Weight>
bool Tiling<Weight>::assert_tiling() {
    bool success = true;
    std::vector<int32_t> uniques(nranks);
    for(uint32_t i = 0; i < nrowgrps; i++) {
        int32_t r = tiles[i][i].rank;
        uniques[r]++;
        if(uniques[r] > 1) {
            success = false;
            break;
        }
    }
    return(success);
}

template<typename Weight>
void Tiling<Weight>::print_tiling(std::string field) {
    if(!Env::rank) {    
        uint32_t skip = 15;
        for (uint32_t i = 0; i < nrowgrps; i++) {
            for (uint32_t j = 0; j < ncolgrps; j++) {  
                auto& tile = tiles[i][j];   
                if(field.compare("rank") == 0) 
                    printf("%d ", tile.rank);
                
                if(j > skip) {
                    printf("...");
                    break;
                }
            }
            printf("\n");
            if(i > skip) {
                printf(".\n.\n.\n");
                break;
            }
        }
        printf("\n");
    }
}


template<typename Weight>
std::tuple<uint64_t, uint64_t, uint64_t> Tiling<Weight>::get_text_info(std::string inputFile) {
    uint64_t nrows = 0;
    uint64_t ncols = 0;    
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
void Tiling<Weight>::read_text_file(std::string inputFile) {
    Logging::print(Logging::LOG_LEVEL::INFO, "Start reading the input file %s\n", inputFile.c_str());
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
    //printf("Close rank %d\n", Env::rank);
    //if(Env::rank == 1) {
 //   int i = 0;
    for(auto& triple: triples) {
        //printf("%d %d %f\n", t.row, t.col, t.weight);
        //printf("%d\n", i);
        //i++;
        insert_triple(triple);
    }
    //}
    
  //  printf("Done rank %d\n", Env::rank);
    
    //if(Env::rank == 0) {
        //auto& t = triples[0];
        //printf("%d %d %f\n", t.row, t.col, t.weight);
        //(triple.row / tile_height), (triple.col / tile_width);
    //}
    
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
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Done  reading the input file %s\n", inputFile.c_str());
    
    return std::make_tuple(nrows, ncols, nnz);
    */
}



template<typename Weight>
void Tiling<Weight>::integer_factorize(uint32_t n, uint32_t& a, uint32_t& b) {
    a = b = sqrt(n);
    while (a * b != n) {
        b++;
        a = n / b;
    }
    if((a * b) != n) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Factorization failed\n");
        std::exit(Env::finalize()); 
    }
}



/* Thread-based tiling based on MPI ranks*/ 
/*
template<typename Weight>
Tiling::Tiling(TILING_TYPE tiling_type_, uint32_t ntiles_, uint32_t nrowgrps_, uint32_t ncolgrps_, uint32_t nranks_, uint32_t rank_nthreads_) 
       : tiling_type(tiling_type_), ntiles(ntiles_) , nrowgrps(nrowgrps_), ncolgrps(ncolgrps_) , nranks(nranks_), rank_ntiles(ntiles_/nranks_)
       , rank_nthreads(rank_nthreads_) {
    
    if((rank_ntiles * nranks != ntiles) or (nrowgrps * ncolgrps != ntiles)) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
        std::exit(Env::finalize()); 
    }
    
    if ((tiling_type == TILING_TYPE::_1D_ROW_)) {
        rowgrp_nranks = 1;
        colgrp_nranks = nranks;
        if(rowgrp_nranks * colgrp_nranks != nranks) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
            std::exit(Env::finalize()); 
        }
    }
    else if(tiling_type == TILING_TYPE::_1D_COL_) {
        rowgrp_nranks = nranks;
        colgrp_nranks = 1;
        if(rowgrp_nranks * colgrp_nranks != nranks) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
            std::exit(Env::finalize()); 
        }
    }
    else if (tiling_type == TILING_TYPE::_2D_) {
        integer_factorize(nranks, rowgrp_nranks, colgrp_nranks);
        if(rowgrp_nranks * colgrp_nranks != nranks) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
            std::exit(Env::finalize()); 
        }
    }
    
    rank_nrowgrps = nrowgrps / colgrp_nranks;
    rank_ncolgrps = ncolgrps / rowgrp_nranks;        
    if(rank_nrowgrps * rank_ncolgrps != rank_ntiles) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
        std::exit(Env::finalize()); 
    }
    
    rank_nthreads = rank_nthreads_;
    nthreads = nranks * rank_nthreads;   
    thread_ntiles = ntiles / nthreads;
    if((thread_ntiles * nthreads != ntiles) or (nrowgrps * ncolgrps != ntiles)) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
        std::exit(Env::finalize()); 
    }
    
    
    if ((tiling_type == TILING_TYPE::_1D_ROW_)) {
        rowgrp_nthreads = 1;
        colgrp_nthreads = nthreads;
    }
    else if (tiling_type == TILING_TYPE::_1D_COL_) {
        rowgrp_nthreads =  nthreads;
        colgrp_nthreads = 1;
    }
    else if (tiling_type == TILING_TYPE::_2D_) {
        rowgrp_nthreads = rowgrp_nranks;
        colgrp_nthreads = nrowgrps / rowgrp_nthreads;
    }


    if(rowgrp_nthreads * colgrp_nthreads != nthreads) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
        std::exit(Env::finalize()); 
    }
    thread_nrowgrps = nrowgrps / colgrp_nthreads;
    thread_ncolgrps = ncolgrps / rowgrp_nthreads;
    if(thread_nrowgrps * thread_ncolgrps != thread_ntiles) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
        std::exit(Env::finalize()); 
    }
    
        
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: Thread-based%s\n", TILING_TYPES[tiling_type]);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: nrowgrps        x ncolgrps        = [%d x %d]\n", nrowgrps, ncolgrps);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: rowgrp_nranks   x colgrp_nranks   = [%d x %d]\n", rowgrp_nranks, colgrp_nranks);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: rank_nrowgrps   x rank_ncolgrps   = [%d x %d]\n", rank_nrowgrps, rank_ncolgrps);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: rowgrp_nthreads x colgrp_nthreads = [%d x %d]\n", rowgrp_nthreads, colgrp_nthreads);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: thread_nrowgrps x thread_ncolgrps = [%d x %d]\n", thread_nrowgrps, thread_ncolgrps);
}
*/





#endif