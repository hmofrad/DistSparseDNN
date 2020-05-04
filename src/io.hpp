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
enum CATEGORY_TYPE {_NONZERO_INSTANCES_ONLY_1, _INSTANCE_AND_PREDICTED_CLASS_PAIRS_1};
enum BIAS_TYPE {_CONSTANTS_, _VECTORS_};

namespace IO {
	template<typename Weight>
    uint64_t get_nnzs(const std::string input_file, const INPUT_TYPE input_type, const std::shared_ptr<struct TwoDHasher> hasher, const uint32_t nrows);
	template<typename Weight>
    std::vector<struct Triple<Weight>> read_file_ijw(const std::string input_file, const INPUT_TYPE input_type, std::shared_ptr<struct TwoDHasher> hasher, bool one_rank, const uint32_t nrows, const uint32_t ncols);
	template<typename Weight>
    uint32_t read_file_iv(const std::string input_file, const INPUT_TYPE input_type, const std::shared_ptr<struct TwoDHasher> hasher, const VALUE_TYPE value_type, std::vector<Weight>& values, const uint32_t nrows);
    
	
    template<typename Weight>
    std::tuple<uint64_t, uint32_t, uint32_t> text_file_stat(const std::string input_file);
    template<typename Weight>
    std::vector<struct Triple<Weight>> text_file_read(const std::string input_file, bool one_rank);
    uint32_t text_file_categories(const std::string input_file, std::vector<uint32_t>& categories, const uint32_t tile_height, const CATEGORY_TYPE category_type, const std::shared_ptr<struct TwoDHasher> hasher);
    
    template<typename Weight>
    std::tuple<uint64_t, uint32_t, uint32_t> binary_file_stat(const std::string input_file);
    //template<typename Weight>
    //std::vector<struct Triple<Weight>> binary_file_read(const std::string input_file, bool one_rank, std::shared_ptr<struct TwoDHasher> hasher);
    int32_t binary_file_categories(const std::string input_file, std::vector<uint32_t>& categories, const uint32_t tile_height, const CATEGORY_TYPE category_type, const std::shared_ptr<struct TwoDHasher> hasher);
    
    template<typename Weight>
    std::vector<struct Triple<Weight>> binary_file_read(const std::string input_file, bool one_rank, std::shared_ptr<struct TwoDHasher> hasher, const uint32_t nrows, const uint32_t ncols);
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
		
		//std::vector<struct Triple<Weight>> triples(share);
		//std::vector<std::vector<struct Triple<Weight>>> triples1(Env::nthreads);
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
				//triples[curr_line_t - curr_line] = triple;
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
uint32_t IO::read_file_iv(const std::string input_file, const INPUT_TYPE input_type, const std::shared_ptr<struct TwoDHasher> hasher, const VALUE_TYPE value_type, std::vector<Weight>& values, const uint32_t nrows) {
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
				instance = hasher->hasher_r->hash(instance);
				if(instance < nrows) {
					values[instance] = 1;
					ninstances++;
				}
			}
			else if(value_type == VALUE_TYPE::_INSTANCE_AND_VALUE_PAIRS_) {
				iss >> instance >>  value;
				instance = hasher->hasher_r->hash(instance);
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
				instance = hasher->hasher_r->hash(instance);
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
				instance = hasher->hasher_r->hash(instance);
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
    Logging::print(Logging::LOG_LEVEL::INFO, "Read file: Total number of instances %d\n", ninstances);
    Logging::print(Logging::LOG_LEVEL::INFO, "Read file: Done reading file %s\n", input_file.c_str());
    Env::barrier();
    return(ninstances);	
}

template<typename Weight>
std::tuple<uint64_t, uint32_t, uint32_t> IO::text_file_stat(const std::string input_file) {
    Logging::print(Logging::LOG_LEVEL::INFO, "Read text: Start collecting info from the input file %s\n", input_file.c_str());
    uint64_t nnz = 0;
    uint32_t nrows = 0;
    uint32_t ncols = 0;    
    
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
        nrows = (nrows < triple.row) ? triple.row : nrows;
        ncols = (ncols < triple.col) ? triple.col : ncols;
        nnz++;
    }
    fin.close();
    Logging::print(Logging::LOG_LEVEL::INFO, "Read text: Done collecting info from the input file %s\n", input_file.c_str());
    Env::barrier();
    
    return std::make_tuple(nnz, nrows + 1, ncols + 1);
}

template<typename Weight>
std::vector<struct Triple<Weight>> IO::text_file_read(const std::string input_file, bool one_rank) {
    Logging::print(Logging::LOG_LEVEL::INFO, "Read text: Start reading the input file %s\n", input_file.c_str());
    
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
    Logging::print(Logging::LOG_LEVEL::INFO, "Read text: File contains %lu lines\n", nlines);
    
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
    
    std::vector<struct Triple<Weight>> triples(share);
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
            triples[curr_line_t - curr_line] = triple;
            curr_line_t++;
        }
        
        fin_t.close();
    }
    fin.close();

    Logging::print(Logging::LOG_LEVEL::INFO, "Read text: Done reading the input file %s\n", input_file.c_str());
    Env::barrier();
    
    return(triples);
}

uint32_t IO::text_file_categories(const std::string input_file, std::vector<uint32_t>& categories, const uint32_t tile_height, const CATEGORY_TYPE category_type, std::shared_ptr<struct TwoDHasher> hasher) {
    Logging::print(Logging::LOG_LEVEL::INFO, "Read text: Start reading the predicted category file %s\n", input_file.c_str());
    
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

    uint32_t ninstances = 0;
	uint32_t instance = 0;
    uint32_t category = 0;
    std::istringstream iss;
    categories.resize(tile_height);
    while (std::getline(fin, line)) {
        iss.clear();
        iss.str(line);
		if(category_type == CATEGORY_TYPE::_NONZERO_INSTANCES_ONLY_1) {
			iss >> instance;
			instance = hasher->hasher_r->hash(instance);
			if(instance < tile_height) {
				categories[instance] = 1;
				ninstances++;
			}
		}
		else if(category_type == CATEGORY_TYPE::_INSTANCE_AND_PREDICTED_CLASS_PAIRS_1) {
			iss >> instance >>  category;
			instance = hasher->hasher_r->hash(instance);
			if(instance < tile_height) {
				categories[instance] = category;
				ninstances++;
			}
		}
		nlines--;
    }
    fin.close();

    Logging::print(Logging::LOG_LEVEL::INFO, "Read text: Total number of instances %d %d\n", ninstances, nlines);
    if(nlines) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Reading %s\n", input_file.c_str());
        std::exit(Env::finalize());
    }
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Read text: Done  reading the predicted category file %s\n", input_file.c_str());
    Env::barrier();
    return(ninstances);
} 
 
 
template<typename Weight>
std::tuple<uint64_t, uint32_t, uint32_t> IO::binary_file_stat(const std::string input_file) {
    Logging::print(Logging::LOG_LEVEL::INFO, "Read binary: Start collecting info from the input file %s\n", input_file.c_str());
    uint64_t nnz = 0;
    uint32_t nrows = 0;
    uint32_t ncols = 0;    
    
    
    std::ifstream fin(input_file.c_str(), std::ios_base::binary);
    if(not fin.is_open()) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Opening %s\n", input_file.c_str());
        std::exit(Env::finalize());
    }
    
    uint64_t filesize, offset = 0;
    fin.seekg (0, std::ios_base::end);
    filesize = (uint64_t) fin.tellg();
    fin.seekg(0, std::ios_base::beg);
    
    if(filesize % sizeof(struct Triple<Weight>)) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Reading %s\n", input_file.c_str());
        std::exit(Env::finalize());
    }
    
    struct Triple<Weight> triple;
    while(offset < filesize) {
        fin.read(reinterpret_cast<char*>(&triple), sizeof(struct Triple<Weight>));
        offset += sizeof(struct Triple<Weight>);
        nrows = (nrows < triple.row) ? triple.row : nrows;
        ncols = (ncols < triple.col) ? triple.col : ncols;
        nnz++;
    }
    fin.close();
    Logging::print(Logging::LOG_LEVEL::INFO, "Read binary: Done  collecting info from the input file %s\n", input_file.c_str());
    
    Env::barrier();
    return std::make_tuple(nnz, nrows + 1, ncols + 1);
} 
/*
template<typename Weight>
std::vector<struct Triple<Weight>> IO::binary_file_read(const std::string input_file, bool one_rank, std::shared_ptr<struct TwoDHasher> hasher) {
    Logging::print(Logging::LOG_LEVEL::INFO, "Read binary: Start reading the input file %s\n", input_file.c_str());
    
    std::ifstream fin(input_file.c_str(), std::ios_base::binary);
    if(not fin.is_open()) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Opening %s\n", input_file.c_str());
        std::exit(Env::finalize());
    }
    
    uint64_t filesize, offset = 0;
    fin.seekg (0, std::ios_base::end);
    filesize = (uint64_t) fin.tellg();
    fin.clear();
    fin.close();
    
    if(filesize % sizeof(struct Triple<Weight>)) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Reading %s\n", input_file.c_str());
        std::exit(Env::finalize());
    }
    
    uint64_t nTriples = filesize / sizeof(struct Triple<Weight>);
    Logging::print(Logging::LOG_LEVEL::INFO, "Read binary: File size is %lu bytes with %lu triples\n", filesize, nTriples);
    
    uint64_t share = (nTriples / Env::nranks) * sizeof(struct Triple<Weight>);
    
    uint64_t start_offset = Env::rank * share;
    uint64_t end_offset = (Env::rank != Env::nranks - 1) ? ((Env::rank + 1) * share) : filesize;
    share = (Env::rank == Env::nranks - 1) ? end_offset - start_offset : share;
    uint64_t share_tripels = share/sizeof(struct Triple<Weight>);

    if(one_rank) {
        share = filesize;
        start_offset = 0;
        end_offset = filesize;
        share_tripels = share/sizeof(struct Triple<Weight>);
    }

    std::vector<struct Triple<Weight>> triples(share_tripels);    
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
        uint64_t index = 0;
        while(offset_t < end_offset_t) {
            fin_t.read(reinterpret_cast<char*>(&triple), sizeof(struct Triple<Weight>));
            index = ((offset_t - start_offset) / sizeof(struct Triple<Weight>));
            triple.row = hasher->hasher_r->hash(triple.row);
            triple.col = hasher->hasher_c->hash(triple.col);
            triples[index] = triple;
            offset_t += sizeof(struct Triple<Weight>);
        }
        fin_t.close();
    }
    Logging::print(Logging::LOG_LEVEL::INFO, "Read binary: Done reading the input file %s\n", input_file.c_str());
    Env::barrier(); 

    return(triples);
}
*/

template<typename Weight>
std::vector<struct Triple<Weight>> IO::binary_file_read(const std::string input_file, bool one_rank, std::shared_ptr<struct TwoDHasher> hasher, const uint32_t nrows, const uint32_t ncols) {
    Logging::print(Logging::LOG_LEVEL::INFO, "Read binary: Start reading the input file %s\n", input_file.c_str());
    
    std::ifstream fin(input_file.c_str(), std::ios_base::binary);
    if(not fin.is_open()) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Opening %s\n", input_file.c_str());
        std::exit(Env::finalize());
    }
    
    uint64_t filesize, offset = 0;
    fin.seekg (0, std::ios_base::end);
    filesize = (uint64_t) fin.tellg();
    fin.clear();
    fin.close();
    
    if(filesize % sizeof(struct Triple<Weight>)) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Reading %s\n", input_file.c_str());
        std::exit(Env::finalize());
    }
    
    uint64_t nTriples = filesize / sizeof(struct Triple<Weight>);
    Logging::print(Logging::LOG_LEVEL::INFO, "Read binary: File size is %lu bytes with %lu triples\n", filesize, nTriples);
    
    uint64_t share = (nTriples / Env::nranks) * sizeof(struct Triple<Weight>);
    
    uint64_t start_offset = Env::rank * share;
    uint64_t end_offset = (Env::rank != Env::nranks - 1) ? ((Env::rank + 1) * share) : filesize;
    share = (Env::rank == Env::nranks - 1) ? end_offset - start_offset : share;
    uint64_t share_tripels = share/sizeof(struct Triple<Weight>);

    if(one_rank) {
        share = filesize;
        start_offset = 0;
        end_offset = filesize;
        share_tripels = share/sizeof(struct Triple<Weight>);
    }
    
    //std::vector<struct Triple<Weight>> triples(share_tripels);    
    std::vector<struct Triple<Weight>> triples;
    std::vector<std::vector<struct Triple<Weight>>> triples1(Env::nthreads);
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
        uint64_t index = 0;
        while(offset_t < end_offset_t) {
            fin_t.read(reinterpret_cast<char*>(&triple), sizeof(struct Triple<Weight>));
            index = ((offset_t - start_offset) / sizeof(struct Triple<Weight>));
            
            //if(Env::print) {
            //    printf("[%d %d]->[%d %d]\n", triple.row, triple.col, (int) hasher->hasher_r->hash(triple.row), (int) hasher->hasher_c->hash(triple.col));
            //}
            //printf("%lu\n", hasher->hasher_c->hash(2));
            //std::exit(0);
            triple.row = hasher->hasher_r->hash(triple.row);
            triple.col = hasher->hasher_c->hash(triple.col);
            
            if((triple.row < nrows) and (triple.col < ncols))
                triples1[tid].push_back(triple);
            
            //triples[index] = triple;
            offset_t += sizeof(struct Triple<Weight>);
        }
        fin_t.close();
    }
    Logging::print(Logging::LOG_LEVEL::INFO, "Read binary: Done reading the input file %s\n", input_file.c_str());
    Env::barrier(); 
    
    for(auto triple1: triples1) {
        triples.insert(triples.end(), triple1.begin(), triple1.end());
    }

    return(triples);
}


int32_t IO::binary_file_categories(const std::string input_file, std::vector<uint32_t>& categories, const uint32_t tile_height, const CATEGORY_TYPE category_type, std::shared_ptr<struct TwoDHasher> hasher) {
    Logging::print(Logging::LOG_LEVEL::INFO, "Read binary: Start reading the predicted category file %s\n", input_file.c_str());
    
    std::ifstream fin(input_file.c_str(), std::ios_base::binary);
    if(not fin.is_open()) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Opening %s\n", input_file.c_str());
        std::exit(Env::finalize());
    }
    
    uint64_t filesize, offset = 0;
    fin.seekg (0, std::ios_base::end);
    filesize = (uint64_t) fin.tellg();
    fin.seekg(0, std::ios_base::beg);
    
    if(filesize % sizeof(uint32_t)) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Reading %s\n", input_file.c_str());
        std::exit(Env::finalize());
    }

    uint32_t ninstances = 0;
	uint32_t instance = 0;
    uint32_t category = 0;
    categories.resize(tile_height);
    while(offset < filesize) {
		if(category_type == CATEGORY_TYPE::_NONZERO_INSTANCES_ONLY_1) {
			fin.read(reinterpret_cast<char*>(&instance), sizeof(uint32_t));
			offset += sizeof(uint32_t);
			instance = hasher->hasher_r->hash(instance);
			if(instance < tile_height) {
				categories[instance] = 1;
				ninstances++;
			}
		}
		else if(category_type == CATEGORY_TYPE::_INSTANCE_AND_PREDICTED_CLASS_PAIRS_1) {
			fin.read(reinterpret_cast<char*>(&instance), sizeof(uint32_t));
			offset += sizeof(uint32_t);
			fin.read(reinterpret_cast<char*>(&category), sizeof(uint32_t));
			offset += sizeof(uint32_t);
			instance = hasher->hasher_r->hash(instance);
			if(instance < tile_height) {
				categories[instance] = category;
				ninstances++;
			}
		}
    }
    fin.close();
	
	Logging::print(Logging::LOG_LEVEL::INFO, "Read text: Total number of instances %d\n", ninstances);
    
    if((filesize /(ninstances * sizeof(uint32_t))) != ninstances) {
        Logging::print(Logging::LOG_LEVEL::ERROR, "Reading %s\n", input_file.c_str());
        std::exit(Env::finalize());
    }
    
    Logging::print(Logging::LOG_LEVEL::INFO, "Read binary: Done reading the predicted category file %s\n", input_file.c_str());
    Env::barrier();
    return(ninstances);
}  
#endif