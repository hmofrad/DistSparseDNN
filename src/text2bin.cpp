/*
 * text2bin.cpp: text2bin converter (Plaintext weighted edge list --> binary weighted edge list)
 * Compile and run: g++ -o text2bin text2bin.cpp  -std=c++14 -DNDEBUG -O3 -flto -fwhole-program -march=native
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */
 
#include <iostream> 
#include <fstream>
#include <sstream>

int main(int argc, char **argv) {
    
	if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " input.txt output.bin"  << std::endl;
    	std::exit(1);
	}
    
    std::string filepath_in  = argv[1];
    std::string filepath_out = argv[2];
    
    std::ifstream fin(filepath_in.c_str(),   std::ios_base::in);
    if(!fin.is_open()) {
        fprintf(stderr, "Unable to open input file\n");
        std::exit(1); 
    }
    
    std::ofstream fout(filepath_out.c_str(), std::ios_base::out);
    if(!fout.is_open()) {
        fprintf(stderr, "Unable to open output file\n");
        std::exit(1); 
    }
    
    uint64_t num_edges = 0;
    uint32_t num_rows = 0;
    uint32_t num_cols = 0;
    uint32_t i, j;
    double w;
    std::string line;
    std::istringstream iss;
    while (std::getline(fin, line)) {
        iss.clear();
        iss.str(line);
        //std::cout << "i=" << i << "j=" << j << std::endl;
        iss >> i >> j >> w;
        fout.write(reinterpret_cast<const char*>(&i), sizeof(uint32_t));
        fout.write(reinterpret_cast<const char*>(&j), sizeof(uint32_t));
        fout.write(reinterpret_cast<const char*>(&w), sizeof(double));
            
            num_edges++;
            num_rows = (num_rows < i) ? i : num_rows;
            num_cols = (num_cols < j) ? j : num_cols;
        
    }
    fout.close();
    fin.close();
	
    std::cout << "File \"" << filepath_in << "\": [" << num_rows << " x " << num_cols << "]" << ", nnz=" <<  num_edges << " convertd into" << std::endl;
    std::cout << "File \"" << filepath_out << "\"." << std::endl;

	return(0);
}