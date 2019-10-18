/*
 * tile.hpp: Tile data structure 
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef TILE_HPP
#define TILE_HPP

#include "triple.hpp"

template<typename Weight>
struct Tile{
    public:
        Tile() {};
        ~Tile() {};
        
        std::vector<struct Triple<Weight>> triples;
        
        int32_t rank;
};

#endif