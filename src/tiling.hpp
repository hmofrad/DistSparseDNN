/*
 * tiling.hpp: Tiling strategy
 * (c) Mohammad Hasanzadeh Mofrad, 2020
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef TILING_HPP
#define TILING_HPP

#include <fstream>
#include <sstream>
#include <numeric>
#include<tuple>

#include "triple.hpp"
#include "tile.hpp"
#include "io.hpp"
#include "allocator.hpp"
#include "spmat.hpp"

enum TILING_TYPE {_1D_COL_, _1D_ROW_, _2D_};
const char* TILING_TYPES[] = {"_1D_COL_", "_1D_ROW_", "_2D_"};

template<typename Weight>
class Tiling {
    public:
        Tiling() {};
        ~Tiling() {};
        
        Tiling(const uint32_t ntiles_, const uint32_t nrowgrps_, const uint32_t ncolgrps_, const uint32_t nranks_, 
               const uint64_t nnzs_, const uint32_t nrows_, const uint32_t ncols_, 
               const std::string input_file, const FILE_TYPE file_type,
               const TILING_TYPE tiling_type_, const COMPRESSED_FORMAT compression_type, 
               std::shared_ptr<struct TwoDHasher> hasher);

        Tiling(const uint32_t ntiles_, const uint32_t nrowgrps_, const uint32_t ncolgrps_, 
               const uint32_t nranks_, const uint32_t rank_nthreads_, const uint32_t nthreads_,
               const uint64_t nnzs_, const uint32_t nrows_, const uint32_t ncols_, 
               const std::string input_file, const FILE_TYPE file_type, 
               const TILING_TYPE tiling_type_, const COMPRESSED_FORMAT compression_type,
               std::shared_ptr<struct TwoDHasher> hasher);

        Tiling(const uint32_t ntiles_, const uint32_t nrowgrps_, const uint32_t ncolgrps_, const uint32_t nranks_, 
               const uint64_t nnzs_, const uint32_t nrows_, const uint32_t ncols_, 
               const TILING_TYPE tiling_type_, const COMPRESSED_FORMAT compression_type,
               std::shared_ptr<struct TwoDHasher> hasher);
                
        Tiling(const uint32_t ntiles_, const uint32_t nrowgrps_, const uint32_t ncolgrps_, const uint32_t nranks_,
               const uint32_t rank_nthreads_, const uint32_t nthreads_, 
               const uint64_t nnzs_, const uint32_t nrows_, const uint32_t ncols_,
               const TILING_TYPE tiling_type_, const COMPRESSED_FORMAT compression_type,
               std::shared_ptr<struct TwoDHasher> hasher);

        uint32_t ntiles, nrowgrps, ncolgrps;
        uint32_t nranks, rank_ntiles, rank_nrowgrps, rank_ncolgrps;
        uint32_t rowgrp_nranks, colgrp_nranks;
        uint32_t rank_nthreads;
        
        uint32_t threads_ntiles, threads_nrowgrps, threads_ncolgrps;
        uint32_t nthreads, thread_ntiles, thread_nrowgrps, thread_ncolgrps;
        uint32_t rowgrp_nthreads, colgrp_nthreads;
        
        uint64_t nnzs;        
        uint32_t nrows, ncols;
        
        uint32_t tile_height, tile_width;
        TILING_TYPE tiling_type;
        
        std::vector<std::vector<struct Tile<Weight>>> tiles;
        
        bool one_rank = false;
        std::vector<uint32_t> set_thread_index();
        std::vector<std::deque<uint32_t>> set_threads_indices();
        std::deque<uint32_t> set_rank_indices();
        uint64_t get_info(const std::string field);
        uint32_t get_tile_info(const std::string field, const int32_t tid);
        uint32_t get_tile_info_max(const std::string field);
        void     set_tile_info(const std::vector<std::vector<struct Tile<Weight>>> other_tiles); 

    private:
        void integer_factorize(const uint32_t n, uint32_t& a, uint32_t& b);
        void populate_tiling();
        void print_tiling(const std::string field);
        bool assert_tiling();
        
        void exchange_triples();
        void insert_triples(std::vector<struct Triple<Weight>>& triples);
        void delete_triples(std::vector<struct Triple<Weight>>& triples);
        void compress_triples(const COMPRESSED_FORMAT compression_type);
        
        void tile_load(bool one_rank);
        void tile_load_print(const std::vector<uint64_t> nedges_vec, const uint64_t nedges, const uint32_t nedges_divisor, const std::string nedges_type);
};

/* Process-based tiling based on MPI ranks*/ 
template<typename Weight>
Tiling<Weight>::Tiling(const uint32_t ntiles_, const uint32_t nrowgrps_, const uint32_t ncolgrps_, const uint32_t nranks_, 
                       const uint64_t nnzs_, const uint32_t nrows_, const uint32_t ncols_,
                       const std::string input_file, const FILE_TYPE file_type,
                       const TILING_TYPE tiling_type_, 
                       const COMPRESSED_FORMAT compression_type, std::shared_ptr<struct TwoDHasher> hasher)
        : ntiles(ntiles_) , nrowgrps(nrowgrps_), ncolgrps(ncolgrps_), nranks(nranks_), rank_ntiles(ntiles_/nranks_), 
          nnzs(nnzs_), nrows(nrows_), ncols(ncols_), tiling_type(tiling_type_) {
    
    one_rank = ((nranks == 1) and (nranks != (uint32_t) Env::nranks)) ? true : false;
   
    if((rank_ntiles * nranks != ntiles) or (nrowgrps * ncolgrps != ntiles)) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
        std::exit(Env::finalize()); 
    }
    
    if (tiling_type == TILING_TYPE::_1D_ROW_) {
        rowgrp_nranks = 1;
        colgrp_nranks = nranks;
        if(rowgrp_nranks * colgrp_nranks != nranks) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
            std::exit(Env::finalize()); 
        }
        nrows += (nrows % nrowgrps) ? (nrowgrps - (nrows % nrowgrps)) : 0;
    }
    else if(tiling_type == TILING_TYPE::_1D_COL_) {
        rowgrp_nranks = nranks;
        colgrp_nranks = 1;
        if(rowgrp_nranks * colgrp_nranks != nranks) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
            std::exit(Env::finalize()); 
        }
        ncols += (ncols % ncolgrps) ? (ncolgrps - (ncols % ncolgrps)) : 0;
    }

    rank_nrowgrps = nrowgrps / colgrp_nranks;
    rank_ncolgrps = ncolgrps / rowgrp_nranks;        
    if(rank_nrowgrps * rank_ncolgrps != rank_ntiles) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
        std::exit(Env::finalize()); 
    }
    
    nthreads = 1;
    if (tiling_type == TILING_TYPE::_1D_ROW_) {
        rowgrp_nthreads = 1;
        colgrp_nthreads = nthreads;
        
    }
    else if (tiling_type == TILING_TYPE::_1D_COL_) {
        rowgrp_nthreads =  nthreads;
        colgrp_nthreads = 1;
        
    }

    if(rowgrp_nthreads * colgrp_nthreads != nthreads) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
        std::exit(Env::finalize()); 
    }
    
    threads_nrowgrps = nrowgrps;
    threads_ncolgrps = ncolgrps;
    
    thread_ntiles = ntiles/nthreads;
    
    thread_nrowgrps = threads_nrowgrps / colgrp_nthreads;
    thread_ncolgrps = threads_ncolgrps / rowgrp_nthreads;
    if(thread_nrowgrps * thread_ncolgrps != thread_ntiles) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
        std::exit(Env::finalize()); 
    }
    
    tile_height = nrows / nrowgrps;
    tile_width  = ncols / ncolgrps;
    
    tiles.resize(nrowgrps);
    for (uint32_t i = 0; i < nrowgrps; i++) { tiles[i].resize(ncolgrps); }
    int32_t gcd_r = std::gcd(rowgrp_nranks, colgrp_nranks);

    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j];
            tile.rank = (one_rank) ? Env::rank : (((i % colgrp_nranks) * rowgrp_nranks + (j % rowgrp_nranks)) + ((i / (nrowgrps/(gcd_r))) * (rank_nrowgrps))) % nranks;
            tile.thread = 0;
            tile.nrows = nrows;
            tile.ncols = ncols;
            tile.start_row = i*tile_height;
            tile.end_row = (i+1)*tile_height;
            tile.height = tile_height;
            tile.start_col = j*tile_width;
            tile.end_col = (j+1)*tile_width;
            tile.width = tile_width;
        }
    }
    
    if((not one_rank) and (ntiles == nranks *nranks)) {
        if(not assert_tiling()) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed1\n");
            std::exit(Env::finalize()); 
        }
    }
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling information: Process-based%s\n", TILING_TYPES[tiling_type]);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling information: nrowgrps      x ncolgrps      = [%d x %d]\n", nrowgrps, ncolgrps);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling information: rowgrp_nranks x colgrp_nranks = [%d x %d]\n", rowgrp_nranks, colgrp_nranks);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling information: rank_nrowgrps x rank_ncolgrps = [%d x %d]\n", rank_nrowgrps, rank_ncolgrps);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling information: nrows         x ncols         = [%d x %d]\n", nrows, ncols);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling information: tile_height   x tile_width    = [%d x %d]\n", tile_height, tile_width);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling information: nnzs                          = [%d]\n", nnzs);
    
    std::vector<struct Triple<Weight>> triples = IO::read_file_ijw<Weight>(input_file, file_type, hasher, one_rank, nrows, ncols);
    Tiling<Weight>::insert_triples(triples);
    Tiling<Weight>::delete_triples(triples);

    if(not one_rank) exchange_triples();
    tile_load(one_rank);

    print_tiling("rank");
    print_tiling("nedges");
    
    compress_triples(compression_type);    
}

template<typename Weight>
Tiling<Weight>::Tiling(const uint32_t ntiles_, const uint32_t nrowgrps_, const uint32_t ncolgrps_,  const uint32_t nranks_, 
                       const uint32_t rank_nthreads_, const uint32_t nthreads_, 
                       const uint64_t nnzs_, const uint32_t nrows_, const uint32_t ncols_,
                       const std::string input_file, const FILE_TYPE file_type,
                       const TILING_TYPE tiling_type_, const COMPRESSED_FORMAT compression_type,
                       std::shared_ptr<struct TwoDHasher> hasher)
                     : ntiles(ntiles_) , nrowgrps(nrowgrps_), ncolgrps(ncolgrps_), nranks(nranks_), rank_ntiles(ntiles_/nranks_), 
                       rank_nthreads(rank_nthreads_), nthreads(nthreads_),
                       nnzs(nnzs_), nrows(nrows_), ncols(ncols_), tiling_type(tiling_type_) {
    
    one_rank = ((nranks == 1) and (nranks != (uint32_t) Env::nranks)) ? true : false;              
    
    if((rank_ntiles * nranks != ntiles) or (nrowgrps * ncolgrps != ntiles)) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
        std::exit(Env::finalize()); 
    }
    
    if (tiling_type == TILING_TYPE::_1D_ROW_) {
        rowgrp_nranks = 1;
        colgrp_nranks = nranks;
        if(rowgrp_nranks * colgrp_nranks != nranks) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed 1\n");
            std::exit(Env::finalize()); 
        }
        nrows += (nrows % nrowgrps) ? (nrowgrps - (nrows % nrowgrps)) : 0;
    }
    else if(tiling_type == TILING_TYPE::_1D_COL_) {
        rowgrp_nranks = nranks;
        colgrp_nranks = 1;
        if(rowgrp_nranks * colgrp_nranks != nranks) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
            std::exit(Env::finalize()); 
        }
        ncols += (ncols % ncolgrps) ? (ncolgrps - (ncols % ncolgrps)) : 0;
    }

    rank_nrowgrps = nrowgrps / colgrp_nranks;
    rank_ncolgrps = ncolgrps / rowgrp_nranks;        
    if(rank_nrowgrps * rank_ncolgrps != rank_ntiles) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failedd\n");
        std::exit(Env::finalize()); 
    }
    
    threads_ntiles = nthreads;
    thread_ntiles = threads_ntiles/nthreads;
    
    if (tiling_type == TILING_TYPE::_1D_ROW_) {
        threads_nrowgrps = nthreads;
        threads_ncolgrps = 1;
    }
    else if(tiling_type == TILING_TYPE::_1D_COL_) {
        threads_nrowgrps = 1;
        threads_ncolgrps = nthreads;
    }

    if((thread_ntiles * nthreads != threads_ntiles) or (threads_nrowgrps * threads_ncolgrps != threads_ntiles)) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
        std::exit(Env::finalize()); 
    }

    if (tiling_type == TILING_TYPE::_1D_ROW_) {
        rowgrp_nthreads = 1;
        colgrp_nthreads = nthreads;
    }
    else if (tiling_type == TILING_TYPE::_1D_COL_) {
        rowgrp_nthreads =  nthreads;
        colgrp_nthreads = 1;
    }

    if(rowgrp_nthreads * colgrp_nthreads != nthreads) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
        std::exit(Env::finalize()); 
    }
    
    thread_nrowgrps = threads_nrowgrps / colgrp_nthreads;
    thread_ncolgrps = threads_ncolgrps / rowgrp_nthreads;
    if(thread_nrowgrps * thread_ncolgrps != thread_ntiles) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
        std::exit(Env::finalize()); 
    }

    tile_height = nrows / nrowgrps;
    tile_width  = ncols / ncolgrps;
    
    tiles.resize(nrowgrps);
    for (uint32_t i = 0; i < nrowgrps; i++) tiles[i].resize(ncolgrps);
    
    int32_t gcd_r = std::gcd(rowgrp_nranks, colgrp_nranks);
    int32_t gcd_t = std::gcd(rowgrp_nthreads, colgrp_nthreads);

    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j];
            int32_t thread_rank = (((i % colgrp_nthreads) * rowgrp_nthreads + (j % rowgrp_nthreads)) 
                                + ((i / (nrowgrps/gcd_t)) * (thread_nrowgrps))) % (Env::nranks * Env::nthreads);
            tile.rank   = (one_rank) ? Env::rank : thread_rank % Env::nranks;
            tile.thread = thread_rank / Env::nranks;   
            tile.nrows = nrows;
            tile.ncols = ncols;
            tile.start_row = i*tile_height;
            tile.end_row = (i+1)*tile_height;
            tile.height = tile_height;
            tile.start_col = j*tile_width;
            tile.end_col = (j+1)*tile_width;
            tile.width = tile_width;        
        }
    }

    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: Thread-based%s\n", TILING_TYPES[tiling_type]);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling information: nrowgrps      x ncolgrps      = [%d x %d]\n", nrowgrps, ncolgrps);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling information: rowgrp_nranks x colgrp_nranks = [%d x %d]\n", rowgrp_nranks, colgrp_nranks);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling information: rank_nrowgrps x rank_ncolgrps = [%d x %d]\n", rank_nrowgrps, rank_ncolgrps);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: rowgrp_nthreads  x colgrp_nthreads  = [%d x %d]\n", rowgrp_nthreads, colgrp_nthreads);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: thread_nrowgrps  x thread_ncolgrps  = [%d x %d]\n", thread_nrowgrps, thread_ncolgrps);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling information: nrows            x ncols            = [%d x %d]\n", nrows, ncols);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling information: tile_height      x tile_width       = [%d x %d]\n", tile_height, tile_width);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling information: nnzs                                = [%d]\n", nnzs);
    
    std::vector<struct Triple<Weight>> triples = IO::read_file_ijw<Weight>(input_file, file_type, hasher, one_rank, nrows, ncols);
    Tiling<Weight>::insert_triples(triples);
    Tiling<Weight>::delete_triples(triples);

    if(not one_rank) exchange_triples();
    tile_load(one_rank);

    print_tiling("rank");
    print_tiling("thread");
    print_tiling("nedges");

    compress_triples(compression_type);
}

template<typename Weight>
Tiling<Weight>::Tiling(const uint32_t ntiles_, const uint32_t nrowgrps_, const uint32_t ncolgrps_, const uint32_t nranks_, 
                       const uint64_t nnzs_, const uint32_t nrows_, const uint32_t ncols_,
                       const TILING_TYPE tiling_type_, const COMPRESSED_FORMAT compression_type,
                       std::shared_ptr<struct TwoDHasher> hasher)
                     : ntiles(ntiles_), nrowgrps(nrowgrps_), ncolgrps(ncolgrps_), nranks(nranks_), rank_ntiles(ntiles_/nranks_), 
                       nnzs(nnzs_), nrows(nrows_), ncols(ncols_), tile_height(nrows / nrowgrps), tile_width(ncols / ncolgrps), tiling_type(tiling_type_) {
                           
    if((rank_ntiles * nranks != ntiles) or (nrowgrps * ncolgrps != ntiles)) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
        std::exit(Env::finalize()); 
    }
    
    if (tiling_type == TILING_TYPE::_1D_ROW_) {
        rowgrp_nranks = 1;
        colgrp_nranks = nranks;
        if(rowgrp_nranks * colgrp_nranks != nranks) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
            std::exit(Env::finalize()); 
        }
    }
    else {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
        std::exit(Env::finalize()); 
    }
    
    rank_nrowgrps = nrowgrps / colgrp_nranks;
    rank_ncolgrps = ncolgrps / rowgrp_nranks;        
    if(rank_nrowgrps * rank_ncolgrps != rank_ntiles) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
        std::exit(Env::finalize()); 
    }
    
    tiles.resize(nrowgrps);
    for (uint32_t i = 0; i < nrowgrps; i++) tiles[i].resize(ncolgrps);

    int32_t gcd_r = std::gcd(rowgrp_nranks, colgrp_nranks);
    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j];
            tile.rank = (((i % colgrp_nranks) * rowgrp_nranks + (j % rowgrp_nranks)) + ((i / (nrowgrps/(gcd_r))) * (rank_nrowgrps))) % nranks;
            tile.nrows = nrows;
            tile.ncols = ncols;
            tile.start_row = i*tile_height;
            tile.end_row = (i+1)*tile_height;
            tile.height = tile_height;
            tile.start_col = j*tile_width;
            tile.end_col = (j+1)*tile_width;
            tile.width = tile_width;
            tile.nedges = 0;
            tile.thread = 0;
        }
    }
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: Thread-based%s\n", TILING_TYPES[tiling_type]);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling information: nrowgrps      x ncolgrps      = [%d x %d]\n", nrowgrps, ncolgrps);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling information: rowgrp_nranks x colgrp_nranks = [%d x %d]\n", rowgrp_nranks, colgrp_nranks);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling information: rank_nrowgrps x rank_ncolgrps = [%d x %d]\n", rank_nrowgrps, rank_ncolgrps);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling information: nrows            x ncols            = [%d x %d]\n", nrows, ncols);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling information: tile_height      x tile_width       = [%d x %d]\n", tile_height, tile_width);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling information: nnzs                                = [%d]\n", nnzs);
    
    print_tiling("rank");
    print_tiling("nedges");

    compress_triples(compression_type);
}


template<typename Weight>
Tiling<Weight>::Tiling(const uint32_t ntiles_, const uint32_t nrowgrps_, const uint32_t ncolgrps_, const uint32_t nranks_, 
                       const uint32_t rank_nthreads_, const uint32_t nthreads_, 
                       const uint64_t nnzs_, const uint32_t nrows_, const uint32_t ncols_,
                       const TILING_TYPE tiling_type_, const COMPRESSED_FORMAT compression_type,
                       std::shared_ptr<struct TwoDHasher> hasher)
                     : ntiles(ntiles_), nrowgrps(nrowgrps_), ncolgrps(ncolgrps_), nranks(nranks_), rank_ntiles(ntiles_/nranks_), 
                       nthreads(nthreads_),
                       nnzs(nnzs_), nrows(nrows_), ncols(ncols_), tile_height(nrows / nrowgrps), tile_width(ncols / ncolgrps), tiling_type(tiling_type_) {
    if((rank_ntiles * nranks != ntiles) or (nrowgrps * ncolgrps != ntiles)) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
        std::exit(Env::finalize()); 
    }
    
    if (tiling_type == TILING_TYPE::_1D_ROW_) {
        rowgrp_nranks = 1;
        colgrp_nranks = nranks;
        if(rowgrp_nranks * colgrp_nranks != nranks) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
            std::exit(Env::finalize()); 
        }
    }
    else {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
        std::exit(Env::finalize()); 
    }
    
    rank_nrowgrps = nrowgrps / colgrp_nranks;
    rank_ncolgrps = ncolgrps / rowgrp_nranks;        
    if(rank_nrowgrps * rank_ncolgrps != rank_ntiles) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
        std::exit(Env::finalize()); 
    }

    threads_ntiles = nthreads;
    thread_ntiles = threads_ntiles/nthreads;
    
    if (tiling_type == TILING_TYPE::_1D_ROW_) {
        threads_nrowgrps = nthreads;
        threads_ncolgrps = 1;
    }
    else if(tiling_type == TILING_TYPE::_1D_COL_) {
        threads_nrowgrps = 1;
        threads_ncolgrps = nthreads;
    }

    if((thread_ntiles * nthreads != threads_ntiles) or (threads_nrowgrps * threads_ncolgrps != threads_ntiles)) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
        std::exit(Env::finalize()); 
    }

    if (tiling_type == TILING_TYPE::_1D_ROW_) {
        rowgrp_nthreads = 1;
        colgrp_nthreads = nthreads;
    }
    else if (tiling_type == TILING_TYPE::_1D_COL_) {
        rowgrp_nthreads =  nthreads;
        colgrp_nthreads = 1;
    }

    if(rowgrp_nthreads * colgrp_nthreads != nthreads) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
        std::exit(Env::finalize()); 
    }
    thread_nrowgrps = threads_nrowgrps / colgrp_nthreads;
    thread_ncolgrps = threads_ncolgrps / rowgrp_nthreads;
    if(thread_nrowgrps * thread_ncolgrps != thread_ntiles) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tiling failed\n");
        std::exit(Env::finalize()); 
    }
    
    tiles.resize(nrowgrps);
    for (uint32_t i = 0; i < nrowgrps; i++) tiles[i].resize(ncolgrps);
    
    int32_t gcd_r = std::gcd(rowgrp_nranks, colgrp_nranks);
    int32_t gcd_t = std::gcd(rowgrp_nthreads, colgrp_nthreads);
    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j];
            int32_t thread_rank = (((i % colgrp_nthreads) * rowgrp_nthreads + (j % rowgrp_nthreads)) 
                                + ((i / (nrowgrps/gcd_t)) * (thread_nrowgrps))) % (Env::nranks * Env::nthreads);
            tile.rank   = thread_rank % Env::nranks;
            tile.thread = thread_rank / Env::nranks;  
            tile.nrows = nrows;
            tile.ncols = ncols;
            tile.start_row = i*tile_height;
            tile.end_row = (i+1)*tile_height;
            tile.height = tile_height;
            tile.start_col = j*tile_width;
            tile.end_col = (j+1)*tile_width;
            tile.width = tile_width; 
            tile.nedges = 0;            
        }
    }
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: Thread-based%s\n", TILING_TYPES[tiling_type]);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling information: nrowgrps      x ncolgrps      = [%d x %d]\n", nrowgrps, ncolgrps);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling information: rowgrp_nranks x colgrp_nranks = [%d x %d]\n", rowgrp_nranks, colgrp_nranks);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling information: rank_nrowgrps x rank_ncolgrps = [%d x %d]\n", rank_nrowgrps, rank_ncolgrps);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: rowgrp_nthreads  x colgrp_nthreads  = [%d x %d]\n", rowgrp_nthreads, colgrp_nthreads);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: thread_nrowgrps  x thread_ncolgrps  = [%d x %d]\n", thread_nrowgrps, thread_ncolgrps);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling information: nrows            x ncols            = [%d x %d]\n", nrows, ncols);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling information: tile_height      x tile_width       = [%d x %d]\n", tile_height, tile_width);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling information: nnzs                                = [%d]\n", nnzs);

    print_tiling("rank");
    print_tiling("thread");
    print_tiling("nedges");

    compress_triples(compression_type);
}

template<typename Weight>
std::vector<uint32_t> Tiling<Weight>::set_thread_index() {
    std::vector<uint32_t> threads_rg(Env::nthreads);
    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j];
            if(tile.rank == Env::rank) threads_rg[tile.thread] = i;
        }
    } 
    return threads_rg;
}

template<typename Weight>
std::vector<std::deque<uint32_t>> Tiling<Weight>::set_threads_indices() {
    std::vector<std::deque<uint32_t>> threads_rgs(Env::nthreads);
    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j];
            if(tile.rank == Env::rank) threads_rgs[tile.thread].push_back(i);
        }
    } 
    return threads_rgs;
}

template<typename Weight>
std::deque<uint32_t> Tiling<Weight>::set_rank_indices() {
    std::deque<uint32_t> rank_rgs;
    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j];
            if(tile.rank == Env::rank) rank_rgs.push_back(i);
        }
    } 
    return rank_rgs;
}

/*
// Faster one
template<typename Weight>
void Tiling<Weight>::integer_factorize(const uint32_t n, uint32_t& a, uint32_t& b) {
    int d = sqrt(n);
    while(n%d != 0) d--;
    a=d, b=n/d;
}
*/
template<typename Weight>
void Tiling<Weight>::integer_factorize(const uint32_t n, uint32_t& a, uint32_t& b) {
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

template<typename Weight>
bool Tiling<Weight>::assert_tiling() {
    bool success = true;
    std::vector<int32_t> uniques(nranks);
    for(uint32_t i = 0; i < nrowgrps; i++) {
        int32_t r = tiles[i][i].rank;
        uniques[r]++;
        if(uniques[r] > 1) { success = false; break; }
    }
    return(success);
}

template<typename Weight>
void Tiling<Weight>::print_tiling(const std::string field) {
    Logging::print(Logging::LOG_LEVEL::INFO, "%s:\n", field.c_str());
    const uint32_t skip = 15;
    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = 0; j < ncolgrps; j++) {  
            auto& tile = tiles[i][j];   
            if(field.compare("rank")        == 0) Logging::print(Logging::LOG_LEVEL::VOID, "%d ", tile.rank);
            else if(field.compare("thread") == 0) Logging::print(Logging::LOG_LEVEL::VOID, "%d ", tile.thread);
            else if(field.compare("nedges") == 0) Logging::print(Logging::LOG_LEVEL::VOID, "%d ", tile.nedges);
            else if(field.compare("height") == 0) Logging::print(Logging::LOG_LEVEL::VOID, "%d ", tile.height);
            else if(field.compare("width")  == 0) Logging::print(Logging::LOG_LEVEL::VOID, "%d ", tile.width);
            if(j > skip) { Logging::print(Logging::LOG_LEVEL::VOID, "..."); break; }
        }
        Logging::print(Logging::LOG_LEVEL::VOID, "\n");
        if(i > skip) { Logging::print(Logging::LOG_LEVEL::VOID, ".\n.\n.\n"); break; }
    }
    Logging::print(Logging::LOG_LEVEL::VOID, "\n");
}

template<typename Weight>
void Tiling<Weight>::exchange_triples() {
    Env::barrier();
    Logging::print(Logging::LOG_LEVEL::INFO, "Tile exchange: Start exchanging tiles...\n", nranks);
    
    // Sanity check for the number of edges 
    uint64_t nedges_start_local  = 0;
    uint64_t nedges_end_local    = 0;
    uint64_t nedges_start_global = 0;
    uint64_t nedges_end_global   = 0;

    std::vector<MPI_Request> out_requests;
    std::vector<MPI_Request> in_requests;
    
    MPI_Request request;     
    
    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j];
            auto& triples = tile.triples;
            nedges_start_local +=  (triples.empty()) ? 0 : triples.size();
        }
    }
      
    std::vector<std::vector<Triple<Weight>>> outboxes(nranks);
    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = 0; j < ncolgrps; j++)   {
            auto& tile = tiles[i][j];
            if(tile.rank != Env::rank) {
                auto& outbox = outboxes[tile.rank];
                outbox.insert(outbox.end(), tile.triples.begin(), tile.triples.end());
                tile.triples.clear();
                tile.triples.shrink_to_fit();
            }
        }
    }
    
    MPI_Datatype MANY_TRIPLES;
    MPI_Type_contiguous(sizeof(Triple<Weight>), MPI_BYTE, &MANY_TRIPLES);
    MPI_Type_commit(&MANY_TRIPLES);
    
    std::vector<std::vector<Triple<Weight>>> inboxes(nranks);
    std::vector<uint32_t> inbox_sizes(nranks);    
    for (uint32_t r = 0; r < nranks; r++) {
        if (r != (uint32_t) Env::rank) {
            auto& outbox = outboxes[r];
            uint32_t outbox_size = outbox.size();
            MPI_Sendrecv(&outbox_size, 1, MPI_UNSIGNED, r, 0, &inbox_sizes[r], 1, MPI_UNSIGNED,
                                                        r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            auto& inbox = inboxes[r];
            inbox.resize(inbox_sizes[r]);
            MPI_Sendrecv(outbox.data(), outbox.size(), MANY_TRIPLES, r, 0, inbox.data(), inbox.size(), MANY_TRIPLES,
                                                                     r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    auto retval = MPI_Type_free(&MANY_TRIPLES);
    if(retval != MPI_SUCCESS) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tile exchanging failed!\n");
        std::exit(Env::finalize()); 
    }
    
    // Insert exchanged triples
    uint64_t exchange_size_local = 0;
    for (uint32_t r = 0; r < nranks; r++) {
        if (r != (uint32_t) Env::rank) {
            auto& inbox = inboxes[r];
            if(not inbox.empty()) {
                insert_triples(inbox);
                delete_triples(inbox);
                exchange_size_local += inbox.size();
            }
        }
    }
    
    // Finzalize sanity check 
    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j];
            if(tile.rank == Env::rank) nedges_end_local += tile.triples.size();
        }
    }    
    MPI_Allreduce(&nedges_start_local, &nedges_start_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(  &nedges_end_local,   &nedges_end_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
    if(nedges_start_global != nedges_end_global) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tile exchange failed\n");
        std::exit(Env::finalize()); 
    }
    
    uint64_t exchange_size_global = 0;
    MPI_Allreduce(&exchange_size_local, &exchange_size_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tile exchange: Exchanged %lu edges.\n", exchange_size_global);
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Tile exchange: Done  exchange tiles.\n");
    Env::barrier();
}


template<typename Weight>
void Tiling<Weight>::tile_load(bool one_rank) {
    Env::barrier();
    Logging::print(Logging::LOG_LEVEL::INFO, "Tile load: Start calculating load...\n");
    if(one_rank) {
        for (uint32_t i = 0; i < nrowgrps; i++) {
            for (uint32_t j = 0; j < ncolgrps; j++) {
                auto& tile = tiles[i][j];
                if(tile.rank == Env::rank) tile.nedges = tile.triples.size();
            }
        }
    }
    else {
        std::vector<std::vector<uint64_t>> nedges_grid(nranks, std::vector<uint64_t>(rank_ntiles));
        std::vector<uint64_t> rank_nedges(nranks);
        std::vector<uint64_t> rowgrp_nedges(nrowgrps);
        std::vector<uint64_t> colgrp_nedges(ncolgrps);
        
        uint32_t k = 0;
        for(uint32_t i = 0; i < nrowgrps; i++) {
            for(uint32_t j = 0; j < ncolgrps; j++) {
                auto& tile = tiles[i][j];
                if(Env::rank == tile.rank){
                    nedges_grid[Env::rank][k] = (tile.nedges) ? tile.nedges : tile.triples.size();
                    k++;
                }
            }
        }
        
        for(uint32_t r = 0; r < nranks; r++) {
            if(r != (uint32_t) Env::rank) {
                auto& out_edges = nedges_grid[Env::rank];
                auto& in_edges = nedges_grid[r];
                MPI_Sendrecv(out_edges.data(), out_edges.size(), MPI_UNSIGNED_LONG, r, Env::rank, 
                              in_edges.data(),  in_edges.size(), MPI_UNSIGNED_LONG, r, r, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        uint64_t nedges = 0;
        std::vector<uint32_t> kth(nranks);
        for(uint32_t i = 0; i < nrowgrps; i++) {
            for(uint32_t j = 0; j < ncolgrps; j++) {
                auto& tile = tiles[i][j];
                uint32_t& k = kth[tile.rank];
                uint64_t e = nedges_grid[tile.rank][k];
                tile.nedges = e;
                rank_nedges[tile.rank] += e;
                rowgrp_nedges[i] += e;
                colgrp_nedges[j] += e;
                nedges += e;
                k++;
            }
        }
    }
    Logging::print(Logging::LOG_LEVEL::INFO, "Tile load: Done calculating load.\n");
    Env::barrier();
}

template<typename Weight>
void Tiling<Weight>::insert_triples(std::vector<struct Triple<Weight>>& triples){
    for(auto triple: triples) {
        std::pair pair = std::make_pair((triple.row / tile_height), (triple.col / tile_width));
        tiles[pair.first][pair.second].triples.push_back(triple);
    }
}

template<typename Weight>
void Tiling<Weight>::delete_triples(std::vector<struct Triple<Weight>>& triples){
    triples.clear();
    triples.shrink_to_fit();
}

template<typename Weight>
uint32_t Tiling<Weight>::get_tile_info_max(const std::string field) {
    uint32_t max = 0;
    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j];
            if(tile.rank == Env::rank) {
                if(field.compare("height") == 0) max = (tile.height > max) ? tile.height : max;
                else if(field.compare("width") == 0) max = (tile.width > max) ? tile.width : max;
            }
        }
    }
    return(max);
}

template<typename Weight>
uint32_t Tiling<Weight>::get_tile_info(const std::string field, const int32_t tid) {
    uint32_t ret = 0;
    bool b = false;
    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j];
            if((tile.rank == Env::rank) and (tile.thread == tid)) {
                if(field.compare("start_row") == 0)   { ret = tile.start_row; b = true;}
                else if(field.compare("height") == 0) { ret = tile.height; b = true;}
                else if(field.compare("width") == 0)  { ret = tile.height; b = true;}
            }
            if(b) break;
        }
        if(b) break;
    }
    
    return(ret);
}

template<typename Weight>
void Tiling<Weight>::set_tile_info(const std::vector<std::vector<struct Tile<Weight>>> other_tiles) {
    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j];
            auto other = other_tiles[i][j];
            tile.nrows = other.nrows;
            tile.ncols = other.ncols;
            tile.start_row = other.start_row;
            tile.end_row = other.end_row;
            tile.start_col = other.start_col;
            tile.end_col = other.end_col;
            tile.height = other.height;
            tile.width = other.width;
        }
    }
    print_tiling("height");
}

template<typename Weight>
void Tiling<Weight>::compress_triples(const COMPRESSED_FORMAT compression_type) {
    Env::barrier();
    Logging::print(Logging::LOG_LEVEL::INFO, "Tile compression: Start compressing tile using %s\n", COMPRESSED_FORMATS[compression_type]);

    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j];
            if(tile.rank == Env::rank) tile.compress(compression_type, one_rank, Env::threads_socket_id[tile.thread]);
        }
    }    
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Tile compression: Done compressing tiles.\n");
    Env::barrier();
}

#endif