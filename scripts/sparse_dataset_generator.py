# sparse_dataset_generator.py: Python TensorFlow script for generating deep sparse networks
# (c) Mohammad Hasanzadeh Mofrad, 2020
# (e) m.hasanzadeh.mofrad@gmail.com
# Datasets include
#     MNIST:   [http://yann.lecun.com/exdb/mnist/]
#     CIFAR10: [https://www.cs.toronto.edu/~kriz/cifar.html]
# Run: python sparse_dataset_generator.py (mnist|cifar10)

import sys
import tensorflow as tf
import pathlib as path
import numpy as np
import scipy as sp
from scipy.stats import rankdata
import tensorflow_model_optimization as tfmot

if(len(sys.argv) != 2):
    print("USAGE: python %s dataset (mnist|cifar10)\n" %(sys.argv[0]))
    sys.exit(0)
dataset=sys.argv[1]

def mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    npixels = x_train.shape[1] * x_train.shape[2]
    x_train = x_train.reshape(x_train.shape[0], npixels)
    x_test = x_test.reshape(x_test.shape[0], npixels)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    return (x_train, y_train), (x_test, y_test)

def cifar10():    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train=tf.image.rgb_to_grayscale(x_train)
    x_test=tf.image.rgb_to_grayscale(x_test)
    x_train=tf.squeeze(x_train)
    x_test=tf.squeeze(x_test)
    npixels = x_train.shape[1] * x_train.shape[2]
    x_train = tf.reshape(x_train, [x_train.shape[0], npixels])
    x_test = tf.reshape(x_test, [x_test.shape[0], npixels])
    x_train = tf.cast(x_train,"float32")
    x_test = tf.cast(x_test,"float32")
    x_train /= 255
    x_test /= 255
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)  
    return (x_train, y_train), (x_test, y_test)    

if(dataset == "mnist"):
    print("Preparing mnist dataset\n")
    (x_train, y_train), (x_test, y_test) = mnist()
elif(dataset == "cifar10"):
    print("Preparing cifar10 dataset\n")
    (x_train, y_train), (x_test, y_test) = cifar10()
else:
    print("USAGE: python %s dataset (mnist|cifar10)\n" %(sys.argv[0]))
    sys.exit(0)

nfeatures = x_train.shape[1]
nclasses = y_test.shape[1]
batch_size = 128
epochs = 1
nlayers = 3
nneurons = 1024

dense_model = tf.keras.Sequential()
for l in range(0, nlayers):
    if(l==0):
        dense_model.add(tf.keras.layers.Dense(nneurons, input_dim=nfeatures, kernel_initializer="normal", activation="relu"))
    elif(l<nlayers-1):
        dense_model.add(tf.keras.layers.Dense(nneurons, input_dim=nneurons, kernel_initializer="normal", activation="relu"))
    else:
        dense_model.add(tf.keras.layers.Dense(nclasses, kernel_initializer="normal", activation="softmax"))
dense_model.summary()
dense_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer="adam", metrics=["accuracy"])

# Train a sparse model using the dense model
sparsity=0.75
begin_step=0
height, weight = 1,1
pruning_params = {'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(sparsity, begin_step), "block_size": (height, weight), 'block_pooling_type': 'AVG'}
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
sparse_model = tfmot.sparsity.keras.prune_low_magnitude(dense_model, **pruning_params)
sparse_model.summary()
sparse_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
sparse_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=callbacks)
sparse_model=tfmot.sparsity.keras.strip_pruning(sparse_model)
sparse_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
sparse_score = sparse_model.evaluate(x_test, y_test, verbose=0)
print("Sparse Network accuracy:", sparse_score[1])

#Save the sparse model into text files
BASE="../data/"
PATH=BASE+"/"+dataset+"/text/"
path.Path(PATH).mkdir(parents=True, exist_ok=True)
sparse_model_file=PATH+dataset+".h5"
sparse_model.save(sparse_model_file, include_optimizer=False)

labels=np.array(range(1,x_train.shape[0]+1))
predictions = (np.argmax(sparse_model.predict(x_train), axis=-1)).astype(int)
C=np.stack([labels,predictions]).T
np.savetxt(PATH+"predictions.txt", C, fmt='%d %d')

weights=sparse_model.get_weights()
weights=np.array(weights);
for l in range(0,weights.shape[0]):
    nonzeros=np.count_nonzero(weights[l])
    rows=np.empty(nonzeros).astype(int)
    vals=np.empty(nonzeros).astype(float)
    k=0
    if(l%2==0):
        cols=np.empty(nonzeros).astype(int)
        for i in range(0,weights[l].shape[0]):
            for j in range(0,weights[l].shape[1]):
                v=weights[l][i][j]
                if(v != 0):
                    rows[k]=i
                    cols[k]=j
                    vals[k]=v
                    k+=1
        name=PATH+"weights"+str(l//2)+".txt"
        W=np.stack([rows,cols,vals]).T
        np.savetxt(name,W,"%d %d %f")
        
    else:
        for i in range(0,weights[l].shape[0]):
            v=weights[l][i]
            if(v != 0):
                rows[k]=i
                vals[k]=v
                k+=1
        name=PATH+"bias"+str(l//2)+".txt"
        B=np.stack([rows,vals]).T
        np.savetxt(name,B,"%d %f")

ninstances=x_train.shape[0]
nonzeros=np.count_nonzero(x_train[:ninstances])
rows=np.empty(nonzeros).astype(int)
cols=np.empty(nonzeros).astype(int)
vals=np.empty(nonzeros).astype(float)
k=0
for i in range(ninstances):
    for j in range(x_train[i].shape[0]):
        v=x_train[i][j]
        if(v != 0):
            rows[k]=i+1
            cols[k]=j
            vals[k]=v
            k+=1
X=np.stack([rows, cols, vals]).T
np.savetxt(PATH+'input.txt', X,"%d %d %f")


total_dense_parameters=x_train.shape[0]*x_train.shape[1]
total_sparse_parameters=np.count_nonzero(x_train[:ninstances])
for l in range(0,weights.shape[0]):
    total_dense_parameters+=weights[l].size;
    total_sparse_parameters+=np.count_nonzero(weights[l])

f = open(PATH+"metadata.txt", "w")
f.write("Dense Network:\n")
S="Dense Input: ["+str(x_train.shape[0])+"x"+str(x_train.shape[1])+"]("+str(x_train.size)+")\n"
f.write(S);
for l in range(0,weights.shape[0]):
    if(l%2==0):
        S="Dense Layer"+str(l//2)+": Weights"+str(l//2)+"["+str(weights[l].shape[0])+"x"+str(weights[l].shape[1])+"]("+str(weights[l].size)+")"
    else:
        S=", Dense Bias"+str(l//2)+"["+str(weights[l].shape[0])+"x1]("+str(weights[l].size)+")\n"
    f.write(S);
S="Number of parameters="+str(total_dense_parameters)+"\n"
f.write(S)

f.write("\n\nSparse Network:\n")
nonzeros=np.count_nonzero(x_train)
S="Sparse Input: ["+str(x_train.shape[0])+"x"+str(x_train.shape[1])+"]("+str(nonzeros)+")\n"
f.write(S);
for l in range(0,weights.shape[0]):
    nonzeros=np.count_nonzero(weights[l])
    if(l%2==0):
        S="Sparse Layer"+str(l//2)+": Weights"+str(l//2)+"["+str(weights[l].shape[0])+"x"+str(weights[l].shape[1])+"]("+str(nonzeros)+")"
    else:
        S=", Dense Bias"+str(l//2)+"["+str(weights[l].shape[0])+"x1]("+str(weights[l].size)+")\n"
    f.write(S)
    
S="Number of parameters="+str(total_sparse_parameters)+"\n"
f.write(S)
S="Network Accuracy="+str(sparse_score[1])+"\n"
f.write(S)
f.close()

