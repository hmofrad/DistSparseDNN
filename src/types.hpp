/*
 * Types.hpp: Get MPI data Type from a templated class 
    Borrowed from https://github.com/thu-pacman/GeminiGraph/blob/master/core/mpi.hpp 
 */
#ifndef TYPES_HPP
#define TYPES_HPP

namespace MPI_Types {
    template<typename Type>    
    MPI_Datatype get_mpi_data_type();
}

template<typename Type>
MPI_Datatype MPI_Types::get_mpi_data_type() {
    if(std::is_same<Type, char>::value){
        return MPI_CHAR;
    }
    else if(std::is_same<Type, unsigned char>::value){
        return MPI_UNSIGNED_CHAR;
    }
    else if(std::is_same<Type, int>::value){
        return MPI_INT;
    }
    else if(std::is_same<Type, unsigned int>::value){
        return MPI_UNSIGNED;
    }
    else if(std::is_same<Type, long>::value){
        return MPI_UNSIGNED_LONG;
    }
    else if(std::is_same<Type, unsigned long>::value){
        return MPI_UNSIGNED_LONG;
    }
    else if(std::is_same<Type, float>::value){
        return MPI_FLOAT;
    }
    else if(std::is_same<Type, double>::value){
        return MPI_DOUBLE;
    }
    else {
        fprintf(stderr, "Type not supported\n");
        return MPI_DATATYPE_NULL;
    }
}
#endif