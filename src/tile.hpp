/*
 * tile.hpp: Tile data structure 
 * (c) Mohammad Hasanzadeh Mofrad, 2020
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

        void compress(const COMPRESSED_FORMAT compression_type_, const bool one_rank, const int32_t socket_id);
        std::vector<struct Triple<Weight>> triples;
        std::shared_ptr<struct Compressed_Format<Weight>> spmat = nullptr;
        COMPRESSED_FORMAT compression_type;
        
        int32_t rank;
        int32_t thread;
        uint64_t nedges    = 0;
        uint32_t start_row = 0;
        uint32_t end_row   = 0;
        uint32_t start_col = 0;
        uint32_t end_col   = 0;
        uint32_t height    = 0;
        uint32_t width     = 0;
};

template<typename Weight>
void Tile<Weight>::compress(const COMPRESSED_FORMAT compression_type_, const bool one_rank, const int32_t socket_id) {  
    compression_type = compression_type_;

    if(compression_type == COMPRESSED_FORMAT::_CSC_) spmat = std::make_shared<struct CSC<Weight>>(triples.size(), height, width, socket_id);
    else if(compression_type == COMPRESSED_FORMAT::_CSR_) spmat = std::make_shared<struct CSR<Weight>>(triples.size(), height, width, socket_id);
    else {
        Logging::print(Logging::LOG_LEVEL::ERROR, "%s compression not implemented\n", COMPRESSED_FORMATS[compression_type]);
        std::exit(Env::finalize());
    }

    if(not triples.empty()){
        spmat->populate(triples, height, width);
        //spmat->walk_dxm(one_rank, 0, 0);
        triples.clear();
        triples.shrink_to_fit();
    }
}
#endif