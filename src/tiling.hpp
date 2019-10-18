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
#include "tile.hpp"
#include "io.hpp"


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
        
        private:
            void integer_factorize(uint32_t n, uint32_t& a, uint32_t& b);
            void print_tiling(std::string field);
            bool assert_tiling();
            void insert_triple(struct Triple<Weight> triple);
            void tile_exchange();
            void tile_sort();
            
            
};


/* Process-based tiling based on MPI ranks*/ 
template<typename Weight>
Tiling<Weight>::Tiling(TILING_TYPE tiling_type_, uint32_t ntiles_, uint32_t nrowgrps_, uint32_t ncolgrps_, uint32_t nranks_, std::string inputFile) 
        :  tiling_type(tiling_type_), ntiles(ntiles_) , nrowgrps(nrowgrps_), ncolgrps(ncolgrps_) , nranks(nranks_)
        , rank_ntiles(ntiles_/nranks_){
           
    //std::tie(nrows, ncols, nnz) = get_text_info(inputFile);
    std::tie(nrows, ncols, nnz) = IO::get_text_info<Weight>(inputFile);
            
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
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: Process-based%s\n", TILING_TYPES[tiling_type]);
    print_tiling("rank");
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: nrowgrps      x ncolgrps      = [%d x %d]\n", nrowgrps, ncolgrps);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: rowgrp_nranks x colgrp_nranks = [%d x %d]\n", rowgrp_nranks, colgrp_nranks);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: rank_nrowgrps x rank_ncolgrps = [%d x %d]\n", rank_nrowgrps, rank_ncolgrps);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: nrows         x ncols         = [%d x %d]\n", nrows, ncols);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: tile_height   x tile_width    = [%d x %d]\n", tile_height, tile_width);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tiling Information: nnz                           = [%d     ]\n", nnz);
    
    IO::read_text_file<Weight>(inputFile, tiles, tile_height, tile_width);
    tile_exchange();
    tile_sort();

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
        //printf("\n");
    }
}

template<typename Weight>
void Tiling<Weight>::insert_triple(struct Triple<Weight> triple) {
    std::pair pair = std::make_pair((triple.row / tile_height), (triple.col / tile_width));
    tiles[pair.first][pair.second].triples.push_back(triple);
}

template<typename Weight>
void Tiling<Weight>::tile_exchange() {
    MPI_Barrier(MPI_COMM_WORLD);
    Logging::print(Logging::LOG_LEVEL::INFO, "Tile exchange: Start exchanging tiles ...\n", Env::nranks);
    
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
            nedges_start_local +=  (tile.triples.size() > 0) ? tile.triples.size() : 0;
        }
    }
      
    std::vector<std::vector<Triple<Weight>>> outboxes(Env::nranks);
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
    
    std::vector<std::vector<Triple<Weight>>> inboxes(Env::nranks);
    std::vector<uint32_t> inbox_sizes(Env::nranks);    
    for (int32_t r = 0; r < Env::nranks; r++) {
        if (r != Env::rank) {
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
    for (int32_t r = 0; r < Env::nranks; r++) {
        if (r != Env::rank) {
            auto& inbox = inboxes[r];
            for(auto& triple: inbox) {
                insert_triple(triple);
            }
            inbox.clear();
            inbox.shrink_to_fit();
        }
    }
    
    // Finzalize sanity check 
    for (uint32_t i = 0; i < nrowgrps; i++) {
        for (uint32_t j = 0; j < ncolgrps; j++) {
            auto& tile = tiles[i][j];
            if(tile.rank == Env::rank) {
                nedges_end_local += tile.triples.size();
            }
        }
    }    
    MPI_Allreduce(&nedges_start_local, &nedges_start_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(  &nedges_end_local,   &nedges_end_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
    if(nedges_start_global != nedges_end_global) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Tile exchange failed\n");
        std::exit(Env::finalize()); 
    }
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Tile exchange: Done exchange tiles!\n");
    MPI_Barrier(MPI_COMM_WORLD);
}


template<typename Weight>
void Tiling<Weight>::tile_sort() {
    ColSort<Weight> f_col;
    
    auto f_comp = [] (const Triple<Weight>& a, const Triple<Weight>& b)
                  {return (a.row == b.row and a.col == b.col);};
                  
    std::vector<struct Triple<Weight>> nums = {{3,2,10}, {2,3,5}, {2,3,1}, {4,0,0}};              
    std::sort(nums.begin(), nums.end(), f_col);
    if(!Env::rank) {
    for(auto& n: nums)
        printf("%d %d %f\n", n.row, n.col, n.weight);
    }
  
  /*
    for(uint32_t t: local_tiles_row_order) {
        auto pair = tile_of_local_tile(t);
        auto& tile = tiles[pair.row][pair.col];
        std::vector<struct Triple<Weight, Integer_Type>>& triples = tile.triples;
        if(triples.size()) {
            std::sort(triples.begin(), triples.end(), f_col);
            /// remove parallel edges (duplicates), necessary for triangle couting 
            if(not parallel_edges) {
                auto last = std::unique(triples.begin(), triples.end(), f_comp);
                triples.erase(last, triples.end());
            }
        }
        tile.nedges = tile.triples.size();
    }
    */
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