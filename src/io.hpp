/*
 * io.hpp: IO interface
 * (c) Mohammad Hasanzadeh Mofrad, 2020
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef IO_HPP
#define IO_HPP

#include <fstream>
#include <sstream>
#include <tuple>

#include "env.hpp"
#include "log.hpp"
#include "tile.hpp"
#include "hashers.hpp"
enum INPUT_TYPE {_TEXT_, _BINARY_};
enum VALUE_TYPE {_CONSTANT_, _NONZERO_INSTANCES_ONLY_, _INSTANCE_AND_VALUE_PAIRS_};

namespace IO {
    template<typename Weight>
    uint64_t get_nnzs(const std::string input_file, const INPUT_TYPE input_type, const std::shared_ptr<struct TwoDHasher> hasher, const uint32_t nrows);
    template<typename Weight>
    std::vector<struct Triple<Weight>> read_file_ijw(const std::string input_file, const INPUT_TYPE input_type, std::shared_ptr<struct TwoDHasher> hasher, bool one_rank, const uint32_t nrows, const uint32_t ncols);
    template<typename Weight>
    uint32_t read_file_iv(const std::string input_file, const INPUT_TYPE input_type, const std::shared_ptr<struct TwoDHasher> hasher, const bool dimension, const VALUE_TYPE value_type, std::vector<Weight>& values, const uint32_t nrows);
}

template<typename Weight>
uint64_t IO::get_nnzs(const std::string input_file, const INPUT_TYPE input_type, const std::shared_ptr<struct TwoDHasher> hasher, const uint32_t nrows) {
    Logging::print(Logging::LOG_LEVEL::INFO, "Get nnzs: Start counting nnzs from file %s\n", input_file.c_str());
    uint64_t ninput_nnzs = 0;
    
    if(input_type == INPUT_TYPE::_TEXT_) {
        std::ifstream fin(input_file.c_str(), std::ios_base::in);
        if(not fin.is_open()) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Opening %s\n", input_file.c_str());
            std::exit(Env::finalize());
        }
        
        struct Triple<Weight> triple;
        std::string line;
        std::istringstream iss;
        while (std::getline(fin, line)) {
            iss.clear();
            iss.str(line);
            iss >> triple.row >> triple.col >> triple.weight;
            triple.row = hasher->hasher_r->hash(triple.row);
            ninput_nnzs += (triple.row < nrows) ? 1 : 0;
        }
        fin.close();
    }
    else if(input_type == INPUT_TYPE::_BINARY_) {
        std::ifstream fin(input_file.c_str(), std::ios_base::binary);
        if(not fin.is_open()) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Opening %s\n", input_file.c_str());
            std::exit(Env::finalize());
        }
        
        uint64_t file_size, offset = 0;
        fin.seekg (0, std::ios_base::end);
        file_size = (uint64_t) fin.tellg();
        fin.seekg(0, std::ios_base::beg);
        
        if(file_size % sizeof(struct Triple<Weight>)) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Reading %s\n", input_file.c_str());
            std::exit(Env::finalize());
        }
        
        struct Triple<Weight> triple;
        while(offset < file_size) {
            fin.read(reinterpret_cast<char*>(&triple), sizeof(struct Triple<Weight>));
            offset += sizeof(struct Triple<Weight>);
            triple.row = hasher->hasher_r->hash(triple.row);
            ninput_nnzs += (triple.row < nrows) ? 1 : 0;
        }
        fin.close();
    }
    Logging::print(Logging::LOG_LEVEL::INFO, "Get nnzs: Done counting nnzs from file %s, nnzs=%lu\n", input_file.c_str(), ninput_nnzs);
    Env::barrier();
    
    return ninput_nnzs;
}

template<typename Weight>
std::vector<struct Triple<Weight>> IO::read_file_ijw(const std::string input_file, const INPUT_TYPE input_type, std::shared_ptr<struct TwoDHasher> hasher, bool one_rank, const uint32_t nrows,  const uint32_t ncols) {
    Logging::print(Logging::LOG_LEVEL::INFO, "Read file: Start reading the input file %s\n", input_file.c_str());
    std::vector<struct Triple<Weight>> triples;
    std::vector<std::vector<struct Triple<Weight>>> triples1(Env::nthreads);
    if(input_type == INPUT_TYPE::_TEXT_) {
        std::ifstream fin(input_file.c_str(), std::ios_base::in);
        if(not fin.is_open()) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Opening %s\n", input_file.c_str());
            std::exit(Env::finalize());
        }
        
        std::string line;
        uint64_t nlines = 0;
        while (std::getline(fin, line)) {
            nlines++;
        }
        fin.clear();
        fin.seekg(0, std::ios_base::beg);
        Logging::print(Logging::LOG_LEVEL::INFO, "Read file: File contains %lu lines\n", nlines);
        
        uint64_t share = nlines / Env::nranks;
        uint64_t start_line = Env::rank * share;
        uint64_t end_line = (Env::rank != Env::nranks - 1) ? ((Env::rank + 1) * share) : nlines;
        share = (Env::rank == Env::nranks - 1) ? end_line - start_line : share;
        uint64_t curr_line = 0;
        while(curr_line < start_line) {
            std::getline(fin, line);
            curr_line++;
        }
        
        if(one_rank) {
            share = nlines;
            start_line = 0;
            end_line = nlines;
            curr_line = 0;
            fin.clear();
            fin.seekg(0, std::ios_base::beg);
        }
        
        #pragma omp parallel
        {
            int nthreads = Env::nthreads; 
            int tid = omp_get_thread_num();
            
            uint64_t share_t = share / nthreads; 
            uint64_t start_line_t = curr_line + (tid * share_t);
            uint64_t end_line_t = (tid != Env::nthreads - 1) ? curr_line + ((tid + 1) * share_t) : end_line;
            share_t = (tid == Env::nthreads - 1) ? end_line_t - start_line_t : share_t;
            uint64_t curr_line_t = 0;
            std::string line_t;
            std::ifstream fin_t(input_file.c_str());
            if(not fin_t.is_open()) {
                Logging::print(Logging::LOG_LEVEL::ERROR, "Opening %s\n", input_file.c_str());
                std::exit(Env::finalize());
            }
            fin_t.seekg(fin.tellg(), std::ios_base::beg);
            curr_line_t = curr_line;
            
            while(curr_line_t < start_line_t) {
                std::getline(fin_t, line_t);
                curr_line_t++;
            }
            
            struct Triple<Weight> triple;
            std::istringstream iss_t;
            while (curr_line_t < end_line_t) {
                std::getline(fin_t, line_t);
                iss_t.clear();
                iss_t.str(line_t);
                iss_t >> triple.row >> triple.col >> triple.weight;
                triple.row = hasher->hasher_r->hash(triple.row);
                triple.col = hasher->hasher_c->hash(triple.col);
                if(triple.col >= ncols) {
                    Logging::print(Logging::LOG_LEVEL::ERROR, "Incorrect file dimensions [%dx%d]\n", nrows, ncols); 
                    std::exit(Env::finalize());
                }
                if(triple.row < nrows) triples1[tid].push_back(triple);
                curr_line_t++;
            }
            
            fin_t.close();
        }
        fin.close();
    }
    else if(input_type == INPUT_TYPE::_BINARY_) {
        std::ifstream fin(input_file.c_str(), std::ios_base::binary);
        if(not fin.is_open()) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Opening %s\n", input_file.c_str());
            std::exit(Env::finalize());
        }
        
        uint64_t file_size, offset = 0;
        fin.seekg (0, std::ios_base::end);
        file_size = (uint64_t) fin.tellg();
        fin.clear();
        fin.close();
        
        if(file_size % sizeof(struct Triple<Weight>)) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Reading %s\n", input_file.c_str());
            std::exit(Env::finalize());
        }
        
        uint64_t nTriples = file_size / sizeof(struct Triple<Weight>);
        Logging::print(Logging::LOG_LEVEL::INFO, "Read file: File size is %lu bytes with %lu triples\n", file_size, nTriples);
        
        uint64_t share = (nTriples / Env::nranks) * sizeof(struct Triple<Weight>);
        
        uint64_t start_offset = Env::rank * share;
        uint64_t end_offset = (Env::rank != Env::nranks - 1) ? ((Env::rank + 1) * share) : file_size;
        share = (Env::rank == Env::nranks - 1) ? end_offset - start_offset : share;
        uint64_t share_tripels = share/sizeof(struct Triple<Weight>);

        if(one_rank) {
            share = file_size;
            start_offset = 0;
            end_offset = file_size;
            share_tripels = share/sizeof(struct Triple<Weight>);
        }
        
        #pragma omp parallel
        {
            int nthreads = Env::nthreads; 
            int tid = omp_get_thread_num();
            uint64_t nTriples_t = share / sizeof(struct Triple<Weight>);
            uint64_t share_t = (nTriples_t / nthreads) * sizeof(struct Triple<Weight>); 
            uint64_t start_offset_t = start_offset + (tid * share_t);
            uint64_t end_offset_t = (tid != Env::nthreads - 1) ? start_offset + ((tid + 1) * share_t) : end_offset;
            share_t = (tid == Env::nthreads - 1) ? end_offset_t - start_offset_t : share_t;
            uint64_t offset_t = start_offset_t;

            std::ifstream fin_t(input_file.c_str(), std::ios_base::binary);
            if(not fin_t.is_open()) {
                Logging::print(Logging::LOG_LEVEL::ERROR, "Opening %s\n", input_file.c_str());
                std::exit(Env::finalize());
            }
            fin_t.seekg(offset_t, std::ios_base::beg);
            
            struct Triple<Weight> triple;
            while(offset_t < end_offset_t) {
                fin_t.read(reinterpret_cast<char*>(&triple), sizeof(struct Triple<Weight>));
                triple.row = hasher->hasher_r->hash(triple.row);
                triple.col = hasher->hasher_c->hash(triple.col);
                if(triple.col >= ncols) {
                    Logging::print(Logging::LOG_LEVEL::ERROR, "Incorret file dimensions [%dx%d]\n", nrows, ncols); 
                    std::exit(Env::finalize());
                }
                if(triple.row < nrows) triples1[tid].push_back(triple);
                offset_t += sizeof(struct Triple<Weight>);
            }
            fin_t.close();
        }
    }
    for(auto& triple1: triples1) triples.insert(triples.end(), triple1.begin(), triple1.end());
    Logging::print(Logging::LOG_LEVEL::INFO, "Read file: Done reading the input file %s\n", input_file.c_str());
    Env::barrier(); 
    
    return(triples);
}

template<typename Weight>
uint32_t IO::read_file_iv(const std::string input_file, const INPUT_TYPE input_type, const std::shared_ptr<struct TwoDHasher> hasher, const bool dimension, const VALUE_TYPE value_type, std::vector<Weight>& values, const uint32_t nrows) {
    Logging::print(Logging::LOG_LEVEL::INFO, "Read file: Start reading file %s\n", input_file.c_str());
    values.resize(nrows);
    uint32_t ninstances = 0;
    uint32_t instance = 0;
    Weight value = 0;
    if(input_type == INPUT_TYPE::_TEXT_) {
        std::ifstream fin(input_file.c_str(), std::ios_base::in);
        if(not fin.is_open()) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Opening %s\n", input_file.c_str());
            std::exit(Env::finalize());
        }

        uint32_t nlines = 0;
        std::string line;
        while (std::getline(fin, line)) {
            nlines++;
        }
        fin.clear();
        fin.seekg(0, std::ios_base::beg);

        std::istringstream iss;
        while(std::getline(fin, line)) {
            iss.clear();
            iss.str(line);
            if(value_type == VALUE_TYPE::_NONZERO_INSTANCES_ONLY_) {
                iss >> instance;
                if(dimension) { instance = hasher->hasher_r->hash(instance); }
                else { instance = hasher->hasher_c->hash(instance); }
                if(instance < nrows) {
                    values[instance] = 1;
                    ninstances++;
                }
            }
            else if(value_type == VALUE_TYPE::_INSTANCE_AND_VALUE_PAIRS_) {
                iss >> instance >>  value;
                if(dimension) { instance = hasher->hasher_r->hash(instance); }
                else { instance = hasher->hasher_c->hash(instance); }
                if(instance < nrows) {
                    values[instance] = value;
                    ninstances++;
                }
            }
            nlines--;
        }
        fin.close();

        if(nlines) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Reading %s\n", input_file.c_str());
            std::exit(Env::finalize());
        }
        
    }    
    else if(input_type == INPUT_TYPE::_BINARY_) {
        std::ifstream fin(input_file.c_str(), std::ios_base::binary);
        if(not fin.is_open()) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Opening %s\n", input_file.c_str());
            std::exit(Env::finalize());
        }
        
        uint64_t file_size, offset = 0;
        fin.seekg (0, std::ios_base::end);
        file_size = (uint64_t) fin.tellg();
        fin.seekg(0, std::ios_base::beg);

        if(file_size % sizeof(uint32_t)) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Reading %s\n", input_file.c_str());
            std::exit(Env::finalize());
        }
        uint32_t loop_count = 0;        
        while(offset < file_size) {
            if(value_type == VALUE_TYPE::_NONZERO_INSTANCES_ONLY_) {
                fin.read(reinterpret_cast<char*>(&instance), sizeof(uint32_t));
                offset += sizeof(uint32_t);
                if(dimension) { instance = hasher->hasher_r->hash(instance); }
                else { instance = hasher->hasher_c->hash(instance); }
                if(instance < nrows) {
                    values[instance] = 1;
                    ninstances++;
                }
            }
            else if(value_type == VALUE_TYPE::_INSTANCE_AND_VALUE_PAIRS_) {
                fin.read(reinterpret_cast<char*>(&instance), sizeof(uint32_t));
                offset += sizeof(uint32_t);
                fin.read(reinterpret_cast<char*>(&value), sizeof(Weight));
                offset += sizeof(Weight);
                if(dimension) { instance = hasher->hasher_r->hash(instance); }
                else { instance = hasher->hasher_c->hash(instance); }
                if(instance < nrows) {
                    values[instance] = value;
                    ninstances++;
                }
            }
            loop_count++;
        }
        fin.close();
        
        uint64_t loop_size = (value_type == VALUE_TYPE::_NONZERO_INSTANCES_ONLY_) ? (loop_count * sizeof(uint32_t)) :
                                                                                    (loop_count * 2 * sizeof(uint32_t));
        if(loop_size != file_size) {
            Logging::print(Logging::LOG_LEVEL::ERROR, "Reading %s\n", input_file.c_str());
            std::exit(Env::finalize());
        }
    }    
    if(value_type == VALUE_TYPE::_INSTANCE_AND_VALUE_PAIRS_) ninstances=std::max(ninstances, nrows); // Take account of zero
    Logging::print(Logging::LOG_LEVEL::INFO, "Read file: Total number of instances %d\n", ninstances);
    Logging::print(Logging::LOG_LEVEL::INFO, "Read file: Done reading file %s\n", input_file.c_str());
    Env::barrier();
    return(ninstances);    
}  
#endif