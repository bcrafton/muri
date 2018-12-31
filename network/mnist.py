
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from FullyConnected import FullyConnected

#####################################

TRAIN_EXAMPLES = 60000
TEST_EXAMPLES = 10000

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(TRAIN_EXAMPLES, 784)
x_train = x_train.astype('float32')

# print (np.shape(np.mean(x_train, axis=0, keepdims=True)))
# print (np.shape(np.std(x_train, axis=0, keepdims=True)))

# we cant actually subtract the mean dumb ass.
# mean = np.mean(x_train, axis=0, keepdims=True)
# x_train = x_train - mean

# std = np.std(x_train, axis=0, keepdims=True)
# x_train = x_train / std
x_train /= 255.

y_train = keras.utils.to_categorical(y_train, 10)

#####################################

l1 = FullyConnected((784, 10))

#####################################

total = 0
correct = 0

lastW = 1. / l1.weights.R
W = 1. / l1.weights.R

for jj in range(TRAIN_EXAMPLES):
    # print (jj)
    
    Y = l1.forward(x_train[jj])
    E = Y - y_train[jj]
    DO = l1.backward(x_train[jj], E)

    W = 1. / l1.weights.R
    dW = W - lastW
    lastW = W
    
    # print (np.max(dR) / RON, np.min(dR) / RON)
    # print (np.max(dR) / ROFF, np.min(dR) / ROFF)
    
    # print (np.std(dW))

    if (np.argmax(Y) == np.argmax(y_train[jj])):
        correct += 1
        
    total += 1
    
    if (total % 100 == 0):
        print (1.0 * correct / total)

plt.hist(np.reshape(W, (-1)), bins=np.linspace(1./ROFF, 1./RON, 100))
plt.show()
    
    
    
    
    
    
    
