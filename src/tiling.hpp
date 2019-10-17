/*
 * tiling.hpp: Tiling strategy
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef TILING_HPP
#define TILING_HPP

enum TILING_TYPE {_1D_COL_, _1D_ROW_,_2D_};
const char* TILING_TYPES[] = {"_1D_COL_", "_1D_ROW_", "_2D_"};

class Tiling {
    public:
        Tiling() {};
        ~Tiling() {};
        
        Tiling(TILING_TYPE tiling_type_, uint32_t ntiles_, uint32_t nrowgrps_, uint32_t ncolgrps_, uint32_t nranks_, uint32_t nrows_, uint32_t ncols_, uint64_t nnz_);
        Tiling( TILING_TYPE tiling_type_, uint32_t ntiles_, uint32_t nrowgrps_, uint32_t ncolgrps_, uint32_t nranks_, uint32_t rank_nthreads_);
        
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

        void integer_factorize(uint32_t n, uint32_t& a, uint32_t& b);
};

Tiling::Tiling(TILING_TYPE tiling_type_, uint32_t ntiles_, uint32_t nrowgrps_, uint32_t ncolgrps_, uint32_t nranks_, uint32_t nrows_, uint32_t ncols_, uint64_t nnz_) 
        :  tiling_type(tiling_type_), ntiles(ntiles_) , nrowgrps(nrowgrps_), ncolgrps(ncolgrps_) , nranks(nranks_)
        , rank_ntiles(ntiles_/nranks_), nrows(nrows_), ncols(ncols_), nnz(nnz_) {
    /* Process-based tiling based on MPI ranks*/ 
    /*
    tiling_type = tiling_type_;
    ntiles = ntiles_;
    nrowgrps = nrowgrps_;
    ncolgrps = ncolgrps_;
    nranks = nranks_;
    rank_ntiles = ntiles / nranks;
    */
    
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
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: Process-based%s\n", TILING_TYPES[tiling_type]);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: nrowgrps      x ncolgrps      = [%d x %d]\n", nrowgrps, ncolgrps);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: rowgrp_nranks x colgrp_nranks = [%d x %d]\n", rowgrp_nranks, colgrp_nranks);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: rank_nrowgrps x rank_ncolgrps = [%d x %d]\n", rank_nrowgrps, rank_ncolgrps);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: nrows         x ncols         = [%d x %d]\n", nrows, ncols);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: tile_height   x tile_width    = [%d x %d]\n", tile_height, tile_width);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: nnz                           = [%d     ]\n", nnz);
}

Tiling::Tiling(TILING_TYPE tiling_type_, uint32_t ntiles_, uint32_t nrowgrps_, uint32_t ncolgrps_, uint32_t nranks_, uint32_t rank_nthreads_) 
       : tiling_type(tiling_type_), ntiles(ntiles_) , nrowgrps(nrowgrps_), ncolgrps(ncolgrps_) , nranks(nranks_), rank_ntiles(ntiles_/nranks_)
       , rank_nthreads(rank_nthreads_) {
    /* Thread-based tiling based on MPI ranks*/ 
    /*
    tiling_type = tiling_type_;
    ntiles = ntiles_;
    nrowgrps = nrowgrps_;
    ncolgrps = ncolgrps_;
    nranks = nranks_;
    rank_ntiles = ntiles / nranks;
    rank_nthreads = rank_nthreads_;
    */
    
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


void Tiling::integer_factorize(uint32_t n, uint32_t& a, uint32_t& b) {
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

#endif