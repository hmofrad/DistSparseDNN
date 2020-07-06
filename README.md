# DistSparseDNN (Distributed Sparse Deep Neural Network Inference)
## Install
    make
## Uninstall
    make clean
## Run
    mpirun -np 4 bin/./radixnet -m 60000 1024 -n 1024 -l 120 -c 0 data/radixnet/bin/MNIST data/radixnet/bin/DNN -p 0
