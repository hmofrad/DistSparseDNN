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

template <typename Weight>
struct ColSort {
    bool operator()(const struct Triple<Weight>& a, const struct Triple<Weight>& b) {
        return((a.col == b.col)  ?
               ((a.row == b.row) ?  
               ((a.weight == b.weight) ? false : a.weight < b.weight ) 
                                               : a.row < b.row)
                                               : (a.col < b.col));
    }
};

template <typename Weight>
struct RowSort {
    bool operator()(const struct Triple<Weight>& a, const struct Triple<Weight>& b) {
        return((a.row == b.row) ?
              ((a.col == b.col) ?  
              ((a.weight == b.weight) ? false : a.weight < b.weight ) 
                                              : a.col < b.col) 
                                              : (a.row < b.row));
    }
};

/*
template <>
struct ColSort {
    return((a.col == b.col) ? (a.row < b.row) : (a.col < b.col));
};
*/

#endif