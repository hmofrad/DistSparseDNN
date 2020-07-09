# DistSparseDNN (Distributed Sparse Deep Neural Network Inference)
## Install
    make
## Uninstall
    make clean
## Run Radixnet (http://graphchallenge.mit.edu/)
    mpirun -np 4 bin/./radixnet -m 60000 1024 -n 1024 -l 120 -c 0 data/radixnet/bin/MNIST data/radixnet/bin/DNN -p 0
## Other datasets (scripts/sparse_dnn_generator.py)
   MNIST, fashion MNIST, CIFAR-10, CIFAR-100, IMDB
