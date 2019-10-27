/*
 * spops.h: Sparse Matrix operations helper
 * Sparse Matrix - Sparse Matrix (SpMM)
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com
 */
  
#ifndef SPOPS_H
#define SPOPS_H

#include "env.hpp"

template<typename Weight>
inline void spmm_sym();
inline void spmm();

#endif