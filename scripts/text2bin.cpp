/*
 * text2bin.cpp: text2bin converter (Plaintext weighted edge list --> binary weighted edge list)
 * Compile and run: g++ -o text2bin text2bin.cpp -std=c++14 -DNDEBUG -O3 -flto -fwhole-program -march=native
 * (c) Mohammad Hasanzadeh Mofrad, 2020
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */
 
#include <iostream> 
#include <fstream>
#include <sstream>

/* mode: 1 = Single column input text file
         2 = Two columns input text file
         3 = Three columns input text file (3rd column as double/float weights)
		 4 = Two columns input text file
 */
 
using WGT = float; 

int main(int argc, char **argv) {
    
	if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " input.txt output.bin mode [1-3]"  << std::endl;
    	std::exit(1);
	}
    
    std::string filepath_in  = argv[1];
    std::string filepath_out = argv[2];
    int mode = atoi(argv[3]);
    
    std::ifstream fin(filepath_in.c_str(),   std::ios_base::in);
    if(!fin.is_open()) {
        fprintf(stderr, "Unable to open input file %s\n", filepath_in.c_str());
        std::exit(1); 
    }
    
    std::ofstream fout(filepath_out.c_str(), std::ios_base::out);
    if(!fout.is_open()) {
        fprintf(stderr, "Unable to open output file %s\n", filepath_out.c_str());
        std::exit(1); 
    }
    
    uint64_t num_edges = 0;
    uint32_t num_rows = 0;
    uint32_t num_cols = 0;
    uint32_t i = 0, j = 0;
    WGT w = .0;
    std::string line;
    std::istringstream iss;
    while (std::getline(fin, line)) {
        iss.clear();
        iss.str(line);

        if(mode == 1) {
            iss >> i;
            fout.write(reinterpret_cast<const char*>(&i), sizeof(uint32_t));
            //std::cout << "i=" << i << std::endl;
        }
        else if(mode == 2) {
            iss >> i >> j;
            fout.write(reinterpret_cast<const char*>(&i), sizeof(uint32_t));
            fout.write(reinterpret_cast<const char*>(&j), sizeof(uint32_t));
            //std::cout << "i=" << i << " j=" << j << std::endl;
        }
        else if(mode == 3) {
            iss >> i >> j >> w;
            fout.write(reinterpret_cast<const char*>(&i), sizeof(uint32_t));
            fout.write(reinterpret_cast<const char*>(&j), sizeof(uint32_t));
            fout.write(reinterpret_cast<const char*>(&w), sizeof(WGT));
            //std::cout << "i=" << i << " j=" << j << " w=" << w << std::endl;
        }
		else if(mode == 4) {
            iss >> i >> w;
            fout.write(reinterpret_cast<const char*>(&i), sizeof(uint32_t));
            fout.write(reinterpret_cast<const char*>(&w), sizeof(WGT));
            //std::cout << "i=" << i << " j=" << j << std::endl;
        }
        
        num_edges++;
        num_rows = (num_rows < i) ? i : num_rows;
        num_cols = (num_cols < j) ? j : num_cols;
    }
    fout.close();
    fin.close();
	
    std::cout << "File \"" << filepath_in << "\": [" << num_rows+1 << " x " << num_cols+1 << "]" << ", nnz=" <<  num_edges << " convertd into File \"" << filepath_out << "\"." << std::endl;

	return(0);
}
