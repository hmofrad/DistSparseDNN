# sparse_mnist.py: Python TensorFlow script for generating a deep sparse network for mnist dataset
# [http://yann.lecun.com/exdb/mnist/]
# Train a dense model and then using that train the sparse model
# Alternatively you can comment the dense train part (dense_model.fit) and only train the sparse network
# (c) Mohammad Hasanzadeh Mofrad, 2020
# (e) m.hasanzadeh.mofrad@gmail.com
# python sparse_mnist.py

import tensorflow as tf
import pathlib as path
import numpy as np
import scipy as sp
from scipy.stats import rankdata
import tensorflow_model_optimization as tfmot

#Train a dense model
PATH="../data/sparse_mnist/data/"
path.Path(PATH).mkdir(parents=True, exist_ok=True)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], num_pixels)
x_test = x_test.reshape(x_test.shape[0], num_pixels)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
num_classes = y_test.shape[1]
batch_size = 128
epochs = 50
nlayers = 30
nneurons = 1024

dense_model = tf.keras.Sequential()
for l in range(0, nlayers):
    if(l==0):
        dense_model.add(tf.keras.layers.Dense(nneurons, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    elif(l<nlayers-1):
        dense_model.add(tf.keras.layers.Dense(nneurons, input_dim=nneurons, kernel_initializer='normal', activation='relu'))
    else:
        dense_model.add(tf.keras.layers.Dense(num_classes, kernel_initializer='normal', activation='softmax'))
dense_model.summary()
dense_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

ACCURACY_THRESHOLD=.97
class myCallback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('accuracy') > ACCURACY_THRESHOLD): 
            print("\nReached %2.2f%% accuracy, early stopping!!" %(ACCURACY_THRESHOLD*100)) 
            self.model.stop_training = True
callbacks = [myCallback()]

dense_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=callbacks)
dense_score = dense_model.evaluate(x_test, y_test, verbose=0)
print("Dense Network accuracy:", dense_score[1])
sparse_model_file=PATH+"dense_model.h5"
sparse_model.save(sparse_model_file, include_optimizer=False)

#dense_model_file=PATH+"dense_model.h5"
#dense_model = tf.keras.models.load_model(dense_model_file, compile=False)
#dense_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
#dense_score = dense_model.evaluate(x_test, y_test, verbose=0)
#print("Dense Network accuracy:", dense_score[1])

# Train a sparse model using the dense model
pruning_params = {'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.75, 0), 'block_size': (1, 1), 'block_pooling_type': 'AVG'}
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
sparse_model = tfmot.sparsity.keras.prune_low_magnitude(dense_model, **pruning_params)
sparse_model.summary()
sparse_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
sparse_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=callbacks)
sparse_model=tfmot.sparsity.keras.strip_pruning(sparse_model)
sparse_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
sparse_score = sparse_model.evaluate(x_test, y_test, verbose=0)
print("Sparse Network accuracy:", sparse_score[1])
sparse_model_file=PATH+"sparse_model.h5"
sparse_model.save(sparse_model_file, include_optimizer=False)

#Save the sparse model into text files
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
            rows[k]=i
            cols[k]=j
            vals[k]=v
            k+=1
X=np.stack([rows, cols, vals]).T
np.savetxt(PATH+'input.txt', X,"%d %d %f")


total_dense_parameters=0
total_sparse_parameters=0
for l in range(0,weights.shape[0]):
    total_dense_parameters+=weights[l].size;
    total_sparse_parameters+=np.count_nonzero(weights[l])

f = open(PATH+"metadata.txt", "w")
f.write("Dense MNIST Network:\n")
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
S="Network Accuracy="+str(dense_score[1])+"\n"
f.write(S)

f.write("\n\nSparse MNIST Network:\n")
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

