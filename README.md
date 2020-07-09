# DistSparseDNN (Distributed Sparse Deep Neural Network Inference)
### Install
    make
### Uninstall
    make clean
### Run
    bin/./radixnet -m <#input instances #input features>
                   -n <#neurons>
                   -l <#layers>
                   -c <#categories>
                   <input path[text|binary]>
                   <DNN   path[text|binary]>
                   -p <parallelism type[data|model|data_then_model]>

### Description

For Radixnet Sparse DNN, first download the MNIST and DNN files from http://graphchallenge.mit.edu/data-sets and then, e.g., for the smallest DNN run

mpirun -np 4 bin/./radixnet -m 60000 1024 -n 1024 -l 120 -c 0 data/radixnet/bin/MNIST data/radixnet/bin/DNN -p 0

For other datasets including MNIST, fashion MNIST, CIFAR-10, CIFAR-100, and IMDB generate the input dataset and sparse DNN using scripts/sparse_dnn_generator.py and then run them; e.g., run fashion MNIST with 30 layers each with 2048 neurons as follow

python3 scripts/sparse_dnn_generator.py fashion_mnist # set parameters internally

mpirun -np 4 bin/./fashion_mnist -m 60000 784 -n 2048 -l 30 -c 10 data/fashion_mnist/bin/ data/fashion_mnist/bin/ -p 0

