/*
 * triple.hpp: Triple data structure 
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef TRIPLE_HPP
#define TRIPLE_HPP

template<typename Weight>
struct Triple {
    uint32_t row;
    uint32_t col;
    Weight weight;
};

struct Empty {};
using MTY = Empty;

template<>
struct Triple<Empty> {
    uint32_t row;
    uint32_t col;
};

#endif