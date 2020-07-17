# DistSparseDNN

Distributed Sparse Deep Neural Network Inference


# Dependencies

- C++17 or higher
- Python3 (optional)
- Pthread, OpenMP, and NUMActl (optional)

## Install

make

## Unistall

make clean

## Test

    bin/./radixnet -i <input_ninstances input_nfeatures ncategories input_path> 
                   -n <nneurons nmax_layers layers_path> 
                   -c <compression_type[0-4]>
                   -m <multiplication_type[0-4]> 
                   -p <parallelism_type[0-4]>
                   -h <hashing_type[0-3]>

## Supported Features
- compression_type
	0. Uncompressed Dense Column (UDC)
	1. Compressed Sparse Column (CSC)
	2. Doubly Compressed Sparse Column (DCSC) (not implemented yet)
	3. Triply Compressed Sparse Column (TCSC) (not implemented yet)
	4. Compressed Sparse Row (CSR)

- multiplication_type
	0. Dense matrix by dense matrix
	1. Dense matrix by compressed matrix
	2. Compressed matrix by compressed matrix
	3. Compressed matrix by doubly compressed matrix (not implemented yet)
	4. Compressed matrix by triply compressed matrix (not implemented yet)
- parallelism_type
	0. Model parallelism
	1. Data parallelism 
	3. Data-then-model parallelism
	4. manage-worker parallelism
	5. work-stealing parallelism

- hashing_type
	0. No hashing
	1. Input hashing
	2. Layer hashing
	3. Input and layer hashing	

## Datasets

- Radixnet Sparse DNN: Download the MNIST and DNN files from http://graphchallenge.mit.edu/data-sets 
- Customized Sparse DNNs: Generate the sparse network using the sparse dnn generator program located under scripts/sparse_dnn_generator.py
	- Currently supported input datasets are MNIST, fashion MNIST, CIFAR-10, CIFAR-100, and IMDB.
- 


## Example Commands

For Radixnet Sparse DNN, e.g., for the smallest DNN use

    mpirun -np 4 bin/./radixnet -i 60000 1024 0 data/radixnet/bin/MNIST -n 1024 120 data/radixnet/bin/DNN -c 1 -m 2 -p 0 -h 3

For other datasets e.g., for inferring fashion MNIST on a sparse DNN with 30 layers each with 2048 neurons use

    python3 scripts/sparse_dnn_generator.py fashion_mnist # Parameters are hardcoded
    mpirun -np 4 bin/./fashion_mnist -i 60000 784 10 data/fashion_mnist/bin/ -n 2048 30 data/fashion_mnist/bin/ -c 1 -m 2 -p 0 -h 3
