/*
 * tile.hpp: Tile data structure 
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef TILE_HPP
#define TILE_HPP

#include "triple.hpp"
#include "spmat.hpp"

template<typename Weight>
struct Tile{
    public:
        Tile() {};
        ~Tile() {};
        
        void sort(const COMPRESSED_FORMAT compression_type, const RowSort<Weight> f_row, const ColSort<Weight> f_col);
        void compress(const COMPRESSED_FORMAT compression_type, uint64_t nnz, uint32_t tile_height, uint32_t tile_width);
        
        std::vector<struct Triple<Weight>> triples;
        std::shared_ptr<struct Compressed_Format<Weight>> spmat = nullptr;
        //std::unique_ptr<struct Compressed_Format<Weight>> spmat;
        
        int32_t rank;
        uint64_t nedges = 0;
};

template<typename Weight>
void Tile<Weight>::sort(const COMPRESSED_FORMAT compression_type, const RowSort<Weight> f_row, const ColSort<Weight> f_col) {
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
void Tile<Weight>::compress(const COMPRESSED_FORMAT compression_type, uint64_t nnz, uint32_t tile_height, uint32_t tile_width) {  
    if(not triples.empty()) {
        if(compression_type == COMPRESSED_FORMAT::_CSR_) {
            spmat = std::make_shared<CSR<Weight>>(triples.size(), tile_height, tile_width);
            spmat->populate(triples, tile_height, tile_width);
            spmat->walk();
            triples.clear();
            triples.shrink_to_fit();
        }
        /*
        else if(compression_type == COMPRESSED_FORMAT::_DCSR_) {
            spmat = std::make_shared<DCSR<Weight>>();
        }
        else if(compression_type == COMPRESSED_FORMAT::_TCSR_) {
            spmat = std::make_shared<TCSR<Weight>>();
        }
        */
        else if(compression_type == COMPRESSED_FORMAT::_CSC_) {
            spmat = std::make_shared<CSC<Weight>>(triples.size(), tile_height, tile_width);
            spmat->populate(triples, tile_height, tile_width);
            spmat->walk();
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
        if(compression_type == COMPRESSED_FORMAT::_CSR_) {
            spmat = std::make_shared<CSR<Weight>>(nnz, tile_height, tile_width);
            //spmat->populate(tile_height, tile_width);
        }
        else if(compression_type == COMPRESSED_FORMAT::_CSC_) {
            spmat = std::make_shared<CSC<Weight>>(nnz, tile_height, tile_width);
            //spmat->populate(tile_height, tile_width);
        } 
        
    }
    else {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Compression failed\n");
        std::exit(Env::finalize()); 
    }
}






#endif