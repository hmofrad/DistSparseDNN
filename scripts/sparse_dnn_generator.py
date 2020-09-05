# sparse_dataset_generator.py: Python TensorFlow script for generating deep sparse networks
# (c) Mohammad Hasanzadeh Mofrad, 2020
# (e) m.hasanzadeh.mofrad@gmail.com
# Datasets: MNIST, Fashion MNIST, CIFAR10, CIFAR100, IMDB
# Run: python sparse_dataset_generator.py directory dataset

import sys
import time
import subprocess
import tensorflow as tf
import pathlib as path
import numpy as np
import scipy as sp
from scipy.stats import rankdata
from numpy import linalg as LA
import tensorflow_model_optimization as tfmot

if(len(sys.argv) != 3):
    print("USAGE: python %s <directory> <dataset [mnist|fashion_mnist|cifar10|cifar100|imdb]>\n" %(sys.argv[0]))
    sys.exit()
    
binary=sys.argv[0]
directory=sys.argv[1]
dataset=sys.argv[2]

def read(binary, dataset):
    if(dataset == "mnist" or dataset == "fashion_mnist"):
        return mnist(dataset)
    elif(dataset == "cifar10" or dataset == "cifar100"):
        return cifar(dataset)
    elif(dataset == "imdb"):
        return imdb()
    else:
        print("USAGE: python %s <directory> <dataset [mnist|cifar|imdb]>\n" %(binary))
        sys.exit()

def mnist(dataset):
    if(dataset == "mnist"):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif(dataset == "fashion_mnist"):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    npixels = x_train.shape[1] * x_train.shape[2]
    x_train = x_train.reshape(x_train.shape[0], npixels)
    x_test = x_test.reshape(x_test.shape[0], npixels)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    x_train = np.array(x_train);
    x_test = np.array(x_test);
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    return (x_train, y_train), (x_test, y_test)

def cifar(dataset):    
    if(dataset == "cifar10"):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif(dataset == "cifar100"):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    x_train = tf.image.rgb_to_grayscale(x_train)
    x_test = tf.image.rgb_to_grayscale(x_test)
    x_train = tf.squeeze(x_train)
    x_test = tf.squeeze(x_test)
    npixels = x_train.shape[1] * x_train.shape[2]
    x_train = tf.reshape(x_train, [x_train.shape[0], npixels])
    x_test = tf.reshape(x_test, [x_test.shape[0], npixels])
    x_train = tf.cast(x_train,"float32")
    x_test = tf.cast(x_test,"float32")
    x_train /= 255
    x_test /= 255
    x_train = np.array(x_train);
    x_test = np.array(x_test);
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)  
    return (x_train, y_train), (x_test, y_test)    

def imdb():    
    num_words=10000
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=num_words)
    temp = np.zeros((len(x_train), num_words))
    for i, x in enumerate(x_train):
        temp[i, x] = 1.
    x_train = temp;
    temp = np.zeros((len(x_test), num_words))
    for i, x in enumerate(x_test):
        temp[i, x] = 1.
    x_test = temp;
    y_train = np.asarray(y_train).astype("float32")
    y_test = np.asarray(y_test).astype("float32")
    return (x_train, y_train), (x_test, y_test)    

def write_text(model, nclasses, x_train, directory, dataset, time, score):
    PATH = directory+"/"+dataset+"/text/"
    path.Path(PATH).mkdir(parents=True, exist_ok=True)
    save_model(model, PATH)
    save_predictions(model, nclasses, x_train, PATH)
    save_network(model, PATH)
    save_input(x_train, PATH)
    save_metadata(model, x_train, PATH, time, score)

def save_model(model, PATH):
    model_file = PATH+dataset+".h5"
    model.save(model_file, include_optimizer=False)    

def save_predictions(model, nclasses, x_train, PATH):
    labels = np.array(range(1,x_train.shape[0]+1))
    if(nclasses==2):
        predictions = tf.round(model.predict(x_train))
        predictions = tf.squeeze(predictions)
        predictions = np.array(predictions).astype(int)
    else:
        predictions = (np.argmax(model.predict(x_train), axis=-1)).astype(int)
    C = np.stack([labels,predictions]).T
    predictions_file = PATH+"predictions.txt";
    np.savetxt(predictions_file, C, fmt='%d %d')

def save_network(model, PATH):
    weights = model.get_weights()
    weights = np.array(weights)
    for l in range(0,weights.shape[0]):
        nonzeros = np.count_nonzero(weights[l])
        k = 0
        if(l%2==0):
            W = np.zeros(shape=(nonzeros, 3))
            for i in range(0,weights[l].shape[0]):
                for j in range(0,weights[l].shape[1]):
                    v = weights[l][i][j]
                    if(v != 0):
                        W[k][0] = i
                        W[k][1] = j
                        W[k][2] = v
                        k += 1
            weight_file = PATH+"weights"+str(l//2)+".txt"
            np.savetxt(weight_file,W,"%d %d %f")
            
        else:
            B = np.zeros(shape=(nonzeros, 2))
            for i in range(0,weights[l].shape[0]):
                v = weights[l][i]
                if(v != 0):
                    B[k][0] = i
                    B[k][1] = v
                    k += 1
            bias_file = PATH+"bias"+str(l//2)+".txt"
            np.savetxt(bias_file,B,"%d %f")

def save_input(x_train, PATH):
    ninstances = x_train.shape[0]
    nonzeros = np.count_nonzero(x_train[:ninstances])
    X = np.zeros(shape=(nonzeros, 3))
    k = 0
    for i in range(ninstances):
        for j in range(x_train[i].shape[0]):
            v = x_train[i][j]
            if(v != 0):
                X[k][0] = i+1
                X[k][1] = j
                X[k][2] = v
                k += 1
    input_file = PATH+'input.txt'
    np.savetxt(input_file, X,"%d %d %f")

def save_metadata(model, x_train, PATH, time, score):
    save_metadata_dense(model, x_train, PATH)
    save_metadata_sparse(model, x_train, PATH)
    save_metadata_sparse1(time, score, PATH)
    
def save_metadata_dense(model, x_train, PATH):
    weights = model.get_weights()
    weights = np.array(weights);
    ninstances = x_train.shape[0]
    total_dense_parameters = x_train.shape[0]*x_train.shape[1]
    for l in range(0,weights.shape[0]):
        total_dense_parameters += weights[l].size;

    file_name = PATH+"metadata.txt"
    f = open(file_name, "w")
    f.write("Dense Network:\n")
    S = "Dense Input: ["+str(x_train.shape[0])+"x"+str(x_train.shape[1])+"]("+str(x_train.size)+")\n"
    f.write(S);
    for l in range(0,weights.shape[0]):
        if(l%2==0):
            S = "Dense Layer"+str(l//2)+": Weights"+str(l//2)+"["+str(weights[l].shape[0])+"x"+str(weights[l].shape[1])+"]("+str(weights[l].size)+")"
        else:
            S = ", Dense Bias"+str(l//2)+"["+str(weights[l].shape[0])+"x1]("+str(weights[l].size)+")\n"
        f.write(S);
    S = "Number of parameters="+str(total_dense_parameters)+"\n"
    f.write(S)
    f.close()
    
def save_metadata_sparse(model, x_train, PATH):
    weights = model.get_weights()
    weights = np.array(weights);
    ninstances = x_train.shape[0]
    total_sparse_parameters = np.count_nonzero(x_train[:ninstances])
    for l in range(0,weights.shape[0]):
        total_sparse_parameters += np.count_nonzero(weights[l])
        
    file_name = PATH+"metadata.txt"
    f = open(file_name, "a")
    f.write("\n\nSparse Network:\n")
    nonzeros = np.count_nonzero(x_train)
    S="Sparse Input: ["+str(x_train.shape[0])+"x"+str(x_train.shape[1])+"]("+str(nonzeros)+")\n"
    f.write(S);
    for l in range(0,weights.shape[0]):
        nonzeros = np.count_nonzero(weights[l])
        if(l%2==0):
            S = "Sparse Layer"+str(l//2)+": Weights"+str(l//2)+"["+str(weights[l].shape[0])+"x"+str(weights[l].shape[1])+"]("+str(nonzeros)+")"
        else:
            S = ", Dense Bias"+str(l//2)+"["+str(weights[l].shape[0])+"x1]("+str(weights[l].size)+")\n"
        f.write(S)
        
    S = "Number of parameters="+str(total_sparse_parameters)+"\n"
    f.write(S)
    f.close()

def save_metadata_sparse1(time, score, PATH):
    file_name = PATH+"metadata.txt"
    f = open(file_name, "a")
    S = "Training time (s)="+str(time)+"\n"
    f.write(S)
    S = "Training score(%)="+str(score*100)+"\n"
    f.write(S)
    f.close()
    
def write_binary(directory, dataset, nlayers):
    cmd=" chmod +x ./text2bin.sh"
    print("CMD=" + cmd + "\n")
    process=subprocess.Popen([cmd], shell=True) 
    cmd="./text2bin.sh " +directory+"/"+dataset+"/ "+str(nlayers)
    print("CMD=" + cmd + "\n")
    process=subprocess.Popen([cmd], shell=True) 
    
def generate_dense_model(nfeatures, nneurons, nclasses, nlayers):
    dense_model = tf.keras.Sequential()
    for l in range(0, nlayers):
        if(l==0):
            dense_model.add(tf.keras.layers.Dense(nneurons, input_dim=nfeatures, kernel_initializer="normal", activation="relu"))
        elif(l<nlayers-1):
            dense_model.add(tf.keras.layers.Dense(nneurons, input_dim=nneurons, kernel_initializer="normal", activation="relu"))
        else:
            if(nclasses==2):
                dense_model.add(tf.keras.layers.Dense(1, kernel_initializer="normal", activation="sigmoid"))
            else:
                dense_model.add(tf.keras.layers.Dense(nclasses, kernel_initializer="normal", activation="softmax"))
    return dense_model

def train_sparse_model(dense_model, nclasses, batch_size, nepochs, weight_sparsity, begin_step, height, width):
    pruning_params = {'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(weight_sparsity, begin_step), "block_size": (height, width), 'block_pooling_type': 'AVG'}
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    sparse_model = tfmot.sparsity.keras.prune_low_magnitude(dense_model, **pruning_params)
    sparse_model.summary()
    if(nclasses==2):
        sparse_model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer="adam", metrics=["binary_accuracy"])
        sparse_model.fit(x_train, y_train, batch_size=batch_size, epochs=nepochs, verbose=1, validation_data=(x_test, y_test), callbacks=callbacks)
        sparse_model = tfmot.sparsity.keras.strip_pruning(sparse_model)
        sparse_model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer="adam", metrics=["binary_accuracy"])
    else:
        sparse_model.compile(optimizer="adam", loss=tf.keras.losses.categorical_crossentropy, metrics=["accuracy"])
        sparse_model.fit(x_train, y_train, batch_size=batch_size, epochs=nepochs, verbose=1, validation_data=(x_test, y_test), callbacks=callbacks)
        sparse_model = tfmot.sparsity.keras.strip_pruning(sparse_model)
        sparse_model.compile(optimizer="adam", loss=tf.keras.losses.categorical_crossentropy, metrics=["accuracy"])
    return sparse_model


def neuron_pruning(sparse_model, neuron_sparsity):
    weights = sparse_model.get_weights()
    weights = np.array(weights)
    for l in range(0,weights.shape[0]):
        k = 0
        if(l%2==0):
            norm = LA.norm(weights[l],axis=0)
            norm = np.tile(norm,(weights[l].shape[0],1))
            rank = (rankdata(norm,method='dense') - 1).astype(int).reshape(norm.shape)
            lower_bound_rank = np.ceil(np.max(rank)*neuron_sparsity).astype(int)
            rank[rank<=lower_bound_rank] = 0
            rank[rank>lower_bound_rank] = 1
            weights[l] = weights[l]*rank
    
    sparse_model.set_weights(weights)
    return sparse_model

# Everything starts from here
tic=time.time()
    
(x_train, y_train), (x_test, y_test) = read(binary, dataset)

nfeatures = x_train.shape[1]
if(y_test.ndim==1):
    nclasses = 2
elif(y_test.ndim==2):
    nclasses = y_test.shape[1]
nneurons = 1024
nlayers = 60
dense_model = generate_dense_model(nfeatures, nneurons, nclasses, nlayers)

batch_size = 128
nepochs = 30
begin_step = nepochs//3;
weight_sparsity = 0.75; 
height, width=1, 1
sparse_model = train_sparse_model(dense_model, nclasses, batch_size, nepochs, weight_sparsity, begin_step, height, width)
sparse_score = sparse_model.evaluate(x_test, y_test, verbose=0)
print("Sparse Network accuracy:", sparse_score[1])

neuron_sparsity = 0.25; 
sparse_model = neuron_pruning(sparse_model, neuron_sparsity)
sparse_score = sparse_model.evaluate(x_test, y_test, verbose=0)
print("Sparse Network accuracy (after neuron pruning):", sparse_score[1])

elapsed_time = time.time()-tic
write_text(sparse_model, nclasses, x_train, directory, dataset, elapsed_time, sparse_score[1])
write_binary(directory, dataset, nlayers)
# And end here
