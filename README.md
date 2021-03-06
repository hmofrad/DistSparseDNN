# DistSparseDNN

Distributed Sparse Deep Neural Network Inference


# Dependencies

- C++17 or higher
- Pthread, OpenMP, and NUMActl (optional)
- Python3 and TensorFlow2 (only for creating customized sparse DNN) 

## Install

    make

## Uninstall

    make clean

## Test

    bin/./app -i <input_ninstances input_nfeatures ncategories input_path> 
                   -n <nneurons nmax_layers layers_path> 
                   -c <input_compression_type[0-4] layer_compression_type[0-4]>
                   -p <parallelism_type[0-4]>
                   -h <hashing_type[0-3]>

## Supported Features

### Compression Types
<ol start="0">
  <li>Uncompressed Dense Column (UDC)</li>
  <li>Compressed Sparse Column (CSC)</li>
  <li>Doubly Compressed Sparse Column (DCSC) (not implemented yet)</li>
  <li>Triply Compressed Sparse Column (TCSC) (not implemented yet) </li>
  <li>Compressed Sparse Row (CSR)</li>
</ol>

### Parallelism Types
<ol start="0">
  <li>Model parallelism (supported UDC and CSC)</li>
  <li>Data parallelism (supported UDC, CSC, and CSR)</li>
  <li>Data-then-model parallelism (supported CSC and CSR)</li>
  <li>Manage-worker parallelism (supported CSC and CSR)</li>
  <li>Work-stealing parallelism (supported CSC and CSR)</li>
</ol>

|                        |     Col-major Formats    |            |             |             |     Row-major Formats    |            |             |             |
|------------------------|--------------------------|------------|-------------|-------------|--------------------------|------------|-------------|-------------|
|     Parallelism        |            UDC           |     CSC    |     DCSC    |     TCSC    |            UDR           |     CSR    |     DCSR    |     TCSR    |
|     Data               |            Yes           |     Yes    |             |             |                          |     Yes    |             |             |
|     Model              |            Yes           |     Yes    |             |             |                          |            |             |             |
|     Data-then-model    |                          |     Yes    |             |             |                          |     Yes    |             |             |
|     Manager-worker     |                          |     Yes    |             |             |                          |     Yes    |             |             |
|     Work-stealing      |                          |     Yes    |             |             |                          |     Yes    |             |             |

### Hashing Types
<ol start="0">
  <li>No hashing</li>
  <li>Input hashing</li>
  <li>Layer hashing</li>
  <li>Input and layer hashing</li>
</ol>

|                      |     Input   |             |     Layer   |             |
|:--------------------:|:-----------:|:-----------:|:-----------:|:-----------:|
|     Hashing Type     |     Rows    |     Cols    |     Rows    |     Cols    |
|     None             |      No     |      No     |      No     |      No     |
|     Input            |      Yes    |      No     |      No     |      No     |
|     Layer            |      No     |      Yes    |      Yes    |      Yes    |
|     Input & Layer    |      Yes    |      Yes    |      Yes    |      Yes    |

## Supported Multiplication Types
<ol start="0">
  <li>Dense matrix by dense matrix</li>
  <li>Dense matrix by compressed matrix</li>
  <li>Compressed matrix by dense matrix</li>
  <li>Compressed matrix by compressed matrix</li>
  <li>Compressed matrix by doubly compressed matrix (not implemented yet)</li>
  <li>Compressed matrix by triply compressed matrix (not implemented yet)</li>
</ol>

|                                            |     Col-major Formats    |                     |     Row-major Formats    |                     |
|--------------------------------------------|:------------------------:|:-------------------:|:------------------------:|:-------------------:|
|     Multiplication Type (Input x Layer)    |        Input Format      |     Layer Format    |        Input Format      |     Layer Format    |
|     Dense x Dense                          |            UDC           |          UDC        |                          |                     |
|     Dense x Compressed                     |            UDC           |          CSC        |                          |                     |
|     Compressed x Dense                     |            CSC           |          UDC        |                          |                     |
|     Compressed x Compressed                |            CSC           |          CSC        |            CSR           |          CSR        |
|     Dense x Doubly Compressed              |                          |                     |                          |                     |
|     Dense x Triply Compressed              |                          |                     |                          |                     |
|     Compressed x Doubly Compressed         |                          |                     |                          |                     |
|     Compressed x Triply Compressed         |                          |                     |                          |                     |



## Datasets

- Radixnet Sparse DNN: Download the MNIST and DNN files from http://graphchallenge.mit.edu/data-sets 
- **Under Development** Customized Sparse DNNs: Generate the sparse network using the sparse dnn generator program located under scripts/sparse_dnn_generator.py
	- Currently supported input datasets are MNIST, fashion MNIST, CIFAR-10, CIFAR-100, and IMDB. 


## Example Commands

For Radixnet Sparse DNN, e.g., for the smallest DNN use

    mpirun -np 4 bin/./radixnet -i 60000 1024 0 data/radixnet/bin/MNIST -n 1024 120 data/radixnet/bin/DNN -c 1 1 -p 0 -h 3

For other datasets e.g., for inferring fashion MNIST on a sparse DNN with 30 layers each with 2048 neurons use

    python3 scripts/sparse_dnn_generator.py fashion_mnist # Parameters are hardcoded
    mpirun -np 4 bin/./fashion_mnist -i 60000 784 10 data/fashion_mnist/bin/ -n 2048 30 data/fashion_mnist/bin/ -c 1 1 -p 0 -h 3

## Papers

- Mohammad Hasanzadeh Mofrad, Rami Melhem, Yousuf Ahmad and Mohammad Hammoud. [“Accelerating Distributed Inference of Sparse Deep Neural Networks via Mitigating the Straggler Effect.”](http://people.cs.pitt.edu/~moh18/files/papers/PID6571125.pdf) In proceedings of IEEE High Performance Extreme Computing (HPEC), Waltham, MA USA, 2020.
- Mohammad Hasanzadeh Mofrad, Rami Melhem, Yousuf Ahmad and Mohammad Hammoud. [“Studying the Effects of Hashing of Sparse Deep Neural Networks on Data and Model Parallelisms.”](http://people.cs.pitt.edu/~moh18/files/papers/PID6577535.pdf) In proceedings of IEEE High Performance Extreme Computing (HPEC), Waltham, MA USA, 2020.

## Contact

Mohammad Hasnzadeh Mofrad (m.hasanzadeh.mofrad@gmail.com)
