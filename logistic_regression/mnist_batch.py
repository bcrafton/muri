
import numpy as np
import math
import gzip
import time
import pickle
import argparse
import keras
from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#######################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--alpha', type=float, default=1e-4)
parser.add_argument('--scale', type=float, default=2.0)
parser.add_argument('--low', type=float, default=1e-2)
parser.add_argument('--pca', type=float, default=784)
args = parser.parse_args()

#######################################

TRAIN_EXAMPLES = 60000
TEST_EXAMPLES = 10000
NUM_CLASSES = 10

#######################################

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#######################################

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)

x_train = x_train.reshape(TRAIN_EXAMPLES, 784)
x_train = x_train.astype('float32')
# x_train /= 255
# x_train = x_train - np.average(x_train, axis=0)

#######################################

y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

x_test = x_test.reshape(TEST_EXAMPLES, 784)
x_test = x_test.astype('float32')
# x_test /= 255
# x_test = x_test - np.average(x_test, axis=0)

#######################################

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#######################################

pca = PCA(n_components=args.pca)
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

#######################################

def sigmoid(x):
  return 1. / (1. + np.exp(-x))
  
def dsigmoid(x):
  # USE A NOT Z
  return x * (1. - x)

def tanh(x):
  return np.tanh(x)
  
def dtanh(x):
  # USE A NOT Z
  return (1. - (x ** 2))
  
def relu(x):
  return np.maximum(x, 0, x)
  
def drelu(x):
  # USE A NOT Z
  return x > 0
  
def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

#######################################

print (args.epochs, args.alpha, args.scale, args.low)

LAYER1 = 784
LAYER2 = 10

# remember there is a sigmoid ... so it makes things a little nicer
# maybe we should use softmax, but sigmoid seems to work fine.
# trying a bunch of learning rates also a good idea
low = args.low
high = args.low * args.scale

weights = np.random.uniform(low, high, size=(LAYER1, LAYER2))
bias = np.zeros(shape=(LAYER2))

#######################################
    acc = 1.0 * correct / TEST_EXAMPLES
    accs.append(acc)
        weights = np.clip(weights, low, high)
    print (np.min(weights), np.max(weights), np.average(weights), np.std(weights))
accs = []

for epoch in range(args.epochs):
    print "epoch: " + str(epoch + 1) + "/" + str(args.epochs)

    ### TRAIN ###
    for ex in range(0, TRAIN_EXAMPLES, args.batch_size): 

        if (TRAIN_EXAMPLES < ex + args.batch_size):
            num = ex + args.batch_size - TRAIN_EXAMPLES
        else:
            num = args.batch_size
            
        start = ex
        end = ex + num
            
        A1 = x_train[start:end]
        Z2 = np.dot(A1, weights1) + bias1
        A2 = tanh(Z2)
        Z3 = np.dot(A2, weights2) + bias2
        A3 = softmax(Z3)
        
        ANS = y_train[start:end]
                
        D3 = A3 - ANS
        D2 = np.dot(D3, np.transpose(B)) * dtanh(A2)
                
        DW3 = np.dot(np.transpose(A2), D3)
        DB3 = np.sum(D3, axis=0)
        
        DW2 = np.dot(np.transpose(A1), D2)
        DB2 = np.sum(D2, axis=0)

        weights2 -= args.alpha * DW3    
        bias2 -= args.alpha * DB3    
        weights1 -= args.alpha * DW2
        bias1 -= args.alpha * DB2
        
    ### TEST ###
    A1 = x_test[start:end]
    Z2 = np.dot(A1, weights1) + bias1
    A2 = tanh(Z2)
    Z3 = np.dot(A2, weights2) + bias2
    A3 = softmax(Z3)

    correct = np.sum( np.argmax(A3, axis=1) == np.argmax(y_test[start:end], axis=1) )
    acc = 1.0 * correct / TEST_EXAMPLES
    accs.append(acc)
    print "accuracy: " + str(acc)

name = "./results/epochs_%d_alpha_%f_scale_%f_low_%f.npy"% (args.epochs, args.alpha, args.scale, args.low)
np.save(name, accs)

    
    
    
    
