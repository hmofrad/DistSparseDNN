/*
 * allocator.hpp: Allocate/deallocate/reallocate 
 * contiguous region of memory using mmap/mremap
 * To keep the realloced memory valid, 
 * we always return the new virtual address
 *
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com
 */

#ifndef ALLOCATOR_HPP
#define ALLOCATOR_HPP

#include <sys/mman.h>
#include <unistd.h>
#include <cstring> 

template<typename Data_Type>
struct Data_Block {
    public:
        Data_Block();
        Data_Block(const uint64_t nitems_);
        ~Data_Block();
        void allocate();
        void deallocate();
        void reallocate(const uint64_t nitems_);
        void clear();
        
        uint64_t nitems;
        uint64_t nbytes;
        Data_Type* ptr;
};

template<typename Data_Type>
Data_Block<Data_Type>::Data_Block(const uint64_t nitems_) : nitems(nitems_), nbytes(nitems_ * sizeof(Data_Type)), ptr(nullptr) {
    allocate();
}

template<typename Data_Type>
Data_Block<Data_Type>::Data_Block() : nitems(0), nbytes(0), ptr(nullptr) {}

template<typename Data_Type>
Data_Block<Data_Type>::~Data_Block() {
    deallocate();
}

template<typename Data_Type>
void Data_Block<Data_Type>::allocate() {
    if(nbytes) {
        nbytes += (nbytes % Env::PAGE_SIZE) ? (Env::PAGE_SIZE - (nbytes % Env::PAGE_SIZE)) : 0;
        if((ptr = (Data_Type*) mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {  
            Logging::print(Logging::LOG_LEVEL::ERROR, "Cannot map memory\n");
            std::exit(Env::finalize()); 
        }
        memset(ptr, 0,  nbytes); 
    }
}

template<typename Data_Type>
void Data_Block<Data_Type>::deallocate() {
    if(ptr and nbytes) {
        if((munmap(ptr, nbytes)) == -1) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Cannot unmap memory\n");
            std::exit(Env::finalize()); 
        }
        ptr = nullptr;
    }
}

template<typename Data_Type>
void Data_Block<Data_Type>::reallocate(const uint64_t nitems_) {
    if(nbytes) {
        uint64_t old_nbytes = nbytes;
        uint64_t new_nbytes = nitems_ * sizeof(Data_Type);
        new_nbytes += (new_nbytes % Env::PAGE_SIZE) ? (Env::PAGE_SIZE - (new_nbytes % Env::PAGE_SIZE)) : 0;

        if(old_nbytes != new_nbytes) {
            if((ptr = (Data_Type*) mremap(ptr, old_nbytes, new_nbytes, MREMAP_MAYMOVE)) == (void*) -1) { 
                Logging::print(Logging::LOG_LEVEL::ERROR, "Cannot unmap memory\n");
                std::exit(Env::finalize()); 
            }
            if(new_nbytes > old_nbytes) {
                memset(ptr + nitems, 0, new_nbytes - old_nbytes); // If grow, zeros the added memory
            }
            nitems = nitems_;
            nbytes = new_nbytes;
        }
    }
}

template<typename Data_Type>
void Data_Block<Data_Type>::clear() {
    memset(ptr, 0,  nbytes); 
}
#endif