/*
 * tile.hpp: Tile data structure 
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef TILE_HPP
#define TILE_HPP

#include "triple.hpp"
#include "spmat.hpp"

enum REFINE_TYPE {_REFINE_NONE_, _REFINE_ROWS_, _REFINE_COLS_, _REFINE_BOTH_};
const char* REFINE_TYPES[] = {"_REFINE_NONE_", "_REFINE_ROWS_", "_REFINE_COLS_", "_REFINE_BOTH_"};

template<typename Weight>
struct Tile{
    public:
        Tile() {};
        ~Tile() {};
        
        void sort(const RowSort<Weight> f_row, const ColSort<Weight> f_col, const COMPRESSED_FORMAT compression_type);
        void compress(const uint64_t nnz, const uint32_t nrows, const uint32_t ncols, const uint32_t tile_height, const uint32_t tile_width,
                      const COMPRESSED_FORMAT compression_type, const REFINE_TYPE refine_type, const bool one_rank);
        
        std::vector<struct Triple<Weight>> triples;
        std::shared_ptr<struct Compressed_Format<Weight>> spmat = nullptr;
        
        int32_t rank;
        int32_t thread;
        uint64_t nedges = 0;
        //uint32_t start_col;
        //uint32_t end_col;
};

template<typename Weight>
void Tile<Weight>::sort(const RowSort<Weight> f_row, const ColSort<Weight> f_col, const COMPRESSED_FORMAT compression_type) {
    if((compression_type == COMPRESSED_FORMAT::_CSR_)  or
       (compression_type == COMPRESSED_FORMAT::_DCSR_) or
       (compression_type == COMPRESSED_FORMAT::_TCSR_)) {
            if(not triples.empty()) {
                std::sort(triples.begin(), triples.end(), f_row);    	
            }
    }
    else if((compression_type == COMPRESSED_FORMAT::_CSC_)  or
       (compression_type == COMPRESSED_FORMAT::_DCSC_) or
       (compression_type == COMPRESSED_FORMAT::_TCSC_)) {
            if(not triples.empty()) {
                std::sort(triples.begin(), triples.end(), f_col);    	
            }
    }
}


template<typename Weight>
void Tile<Weight>::compress(const uint64_t nnz, const uint32_t nrows, const uint32_t ncols, const uint32_t tile_height, const uint32_t tile_width,
                            const COMPRESSED_FORMAT compression_type, const REFINE_TYPE refine_type, const bool one_rank) {  
//printf("cooooooooooooo %d\n", cocompress);
    if(not triples.empty() and (nnz == triples.size())){
        /*
        if(compression_type == COMPRESSED_FORMAT::_CSR_) {
            spmat = std::make_shared<CSR<Weight>>(triples.size(), tile_height, tile_width);
            spmat->populate(triples, tile_height, tile_width);
            //spmat->walk();
            triples.clear();
            triples.shrink_to_fit();
        }
        else if(compression_type == COMPRESSED_FORMAT::_DCSR_) {
            spmat = std::make_shared<DCSR<Weight>>();
        }
        else if(compression_type == COMPRESSED_FORMAT::_TCSR_) {
            spmat = std::make_shared<TCSR<Weight>>();
        }
        */
        //else 
        if(compression_type == COMPRESSED_FORMAT::_CSC_) {
            spmat = std::make_shared<CSC<Weight>>(nnz, tile_height, tile_width, one_rank);
            //if(not cocompress) {
            //printf("populate nnz=%lu nrows=%d ncols=%d height=%d width=%d %lu\n", nnz, nrows, ncols, tile_height, tile_width, triples.size());
             
            spmat->populate(triples, tile_height, tile_width);
            //if(refine) spmat->refine(nrows);
            
            if(refine_type == REFINE_TYPE::_REFINE_ROWS_) {
                spmat->refine_rows(nrows);
            }
            else if(refine_type == REFINE_TYPE::_REFINE_COLS_) {
                spmat->refine_cols();
            }
            else if(refine_type == REFINE_TYPE::_REFINE_BOTH_) {
                spmat->refine_both(nrows);
            }
            
            
            /*
            if(ncols != tile_height) {
                spmat->refine_cols();
            }
            else {
                spmat->refine_rows(nrows);
                //spmat->walk();
                std::exit(0);
            }
            */
            //spmat->walk();
            triples.clear();
            triples.shrink_to_fit();
        } 
        /*    
        else if(compression_type == COMPRESSED_FORMAT::_DCSC_) {
            spmat = std::make_shared<DCSC<Weight>>();
        }
        else if(compression_type == COMPRESSED_FORMAT::_TCSC_) {
            spmat = std::make_shared<TCSC<Weight>>();
        }
        */
    }
    else if(nnz) {
        /*
        if(compression_type == COMPRESSED_FORMAT::_CSR_) {
            spmat = std::make_shared<CSR<Weight>>(nnz, tile_height, tile_width);
            //spmat->populate(tile_height, tile_width);
        }
        else
        */            
        if(compression_type == COMPRESSED_FORMAT::_CSC_) {
            spmat = std::make_shared<CSC<Weight>>(nnz, tile_height, tile_width, one_rank);
            //spmat->populate(tile_height, tile_width);
        } 
        
    }
    else {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Compression failed\n");
        std::exit(Env::finalize()); 
    }
}






#endif