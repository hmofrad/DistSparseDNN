/*
 * bitmap.hpp: Bitmap implementation 
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com
 */
 
#ifndef BITMAP_H
#define BITMAP_H

#define BYTE_N_BITS 8

struct Bitmap {
    public:
        Bitmap() {};
        ~Bitmap() {};
        Bitmap(uint32_t size_);
        
        void set_bit(uint32_t index);
        bool get_bit(uint32_t index);
        void clear_bit(uint32_t index);
        uint64_t count_and_clear();
        
    private:    
        std::shared_ptr<struct Data_Block<unsigned char>> bitmap_blk; 
        uint32_t size = 0;
        

};

Bitmap::Bitmap(uint32_t size_) : size(size_) {
    uint32_t size_raw = ((size_ + BYTE_N_BITS - 1)/BYTE_N_BITS);
    bitmap_blk = std::move(std::make_shared<struct Data_Block<unsigned char>>(size_raw));
}

void Bitmap::set_bit(uint32_t index) {
    unsigned char* bitmap = bitmap_blk->ptr;
    uint32_t byte  =  index / BYTE_N_BITS;
    uint32_t bit   =  index % BYTE_N_BITS;
    unsigned char mask = 1 << bit;
    bitmap[byte] |= mask;
}

bool Bitmap::get_bit(uint32_t index) {
    unsigned char* bitmap = bitmap_blk->ptr;
    uint32_t byte  =  index / BYTE_N_BITS;
    uint32_t bit   =  index % BYTE_N_BITS;
    unsigned char mask = 1 << bit;
    return((bitmap[byte] & mask) != 0);
}

void Bitmap::clear_bit(uint32_t index) {
    unsigned char* bitmap = bitmap_blk->ptr;
    uint32_t byte  =  index / BYTE_N_BITS;
    uint32_t bit   =  index % BYTE_N_BITS;
    unsigned char mask = ~(1 << bit);
    bitmap[byte] &= mask;
}

uint64_t Bitmap::count_and_clear() {
    uint64_t count = 0; 
    unsigned char* bitmap = bitmap_blk->ptr;
    /*
    for(uint32_t i = size - 1; i+BYTE_N_BITS >= BYTE_N_BITS; i-=BYTE_N_BITS) {
        uint32_t byte  =  i / BYTE_N_BITS;
        for(uint32_t j = 0; j < BYTE_N_BITS; j++) {
            uint32_t bit =  (i - j) % BYTE_N_BITS;
            unsigned char mask = 1 << bit;
            count += bitmap[byte] & mask;
            printf("%d %d\n", i, j, bit);
        }
        break;
        //if(i < 59000) break;
    }
    */
    
    for(uint32_t i = size - 1; i+1 >= 1; i--) {
        if(get_bit(i)) {
            clear_bit(i);
            count++;
        }
    }
    
    
    //unsigned int byte  =  index / BYTE_N_BITS;
    //unsigned int bit   =  index % BYTE_N_BITS;
    
    //bitmap[byte] &= (bitmap[byte] - 1);
    
    //count += n & 1; 
   // n >>= 1; 
    
    //unsigned char mask = 1 << bit;
    
    /*
    for(i = BITMAP_COUNT - 1; i+1 >= 1; i--) {
        j = get_bit(i);
        printf("Block[%5d]=%d\n", i, j);
    }
    */
    /*
    while (n) { 
        n &= (n - 1); 
        count++; 
    } 
    */
    return count;
}

#endif