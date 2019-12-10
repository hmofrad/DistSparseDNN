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
    if(not triples.empty() and (nnz == triples.size())){
        if(compression_type == COMPRESSED_FORMAT::_CSC_) {
            spmat = std::make_shared<CSC<Weight>>(nnz, tile_height, tile_width, one_rank);
            spmat->populate(triples, tile_height, tile_width);
            
            if(refine_type == REFINE_TYPE::_REFINE_ROWS_) {
                spmat->refine_rows(nrows);
            }
            else if(refine_type == REFINE_TYPE::_REFINE_COLS_) {
                spmat->refine_cols();
            }
            else if(refine_type == REFINE_TYPE::_REFINE_BOTH_) {
                spmat->refine_both(nrows);
            }

            //spmat->walk();
            triples.clear();
            triples.shrink_to_fit();
        }
    }
    else {// if(nnz) {
        if(compression_type == COMPRESSED_FORMAT::_CSC_) {
            spmat = std::make_shared<CSC<Weight>>(nnz, tile_height, tile_width, one_rank);
        }
    }
    //else {
    //    Logging::print(Logging::LOG_LEVEL::ERROR, "Compression failed\n");
    //    std::exit(Env::finalize()); 
    //}
}
#endif