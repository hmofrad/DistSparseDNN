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
        Tile() {};//printf("CREATING Tile = %d %p\n", spmat != nullptr, spmat);};
        ~Tile() {
       //     printf("1.~tile: Delete\n"); 
     //       std::cout << spmat.use_count() << std::endl;
   //spmat.reset();
   //std::cout << spmat.use_count() << std::endl;
            /*
            if(spmat) {
                delete spmat;
                printf("2.~tile: Delete %d %p\n", spmat != nullptr, spmat); 
                spmat = nullptr;
            }
            */
            //printf("3.~tile: Delete %d %p\n", spmat != nullptr, spmat); 
        };
        
        std::vector<struct Triple<Weight>> triples;
        ////struct Compressed_Block<Weight>* spmat = nullptr;
        //std::unique_ptr<struct Compressed_Block<Weight>> spmat1;
        std::shared_ptr<struct Compressed_Block<Weight>> spmat = nullptr;
        
        int32_t rank;
        uint64_t nedges;
};

#endif