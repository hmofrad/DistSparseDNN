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

        void compress(const bool one_rank);
        
        std::vector<struct Triple<Weight>> triples;
        std::shared_ptr<struct CSC<Weight>> spmat = nullptr;
        
        int32_t rank;
        int32_t thread;
        uint64_t nedges = 0;
        uint32_t start_row = 0;
        uint32_t end_row = 0;
        uint32_t start_col = 0;
        uint32_t end_col = 0;
        uint32_t height = 0;
        uint32_t width = 0;
};

template<typename Weight>
void Tile<Weight>::compress(const bool one_rank) {  
                            
    if(not triples.empty()){        
        const ColSort<Weight> f_col;
        std::sort(triples.begin(), triples.end(), f_col);   
        spmat = std::make_shared<CSC<Weight>>(triples.size(), height, width);
        spmat->populate(triples, start_row, end_row, start_col, end_col);
        spmat->walk(one_rank);
        triples.clear();
        triples.shrink_to_fit();
    }
   else {
        spmat = std::make_shared<CSC<Weight>>(0, height, width);
    }
}
#endif