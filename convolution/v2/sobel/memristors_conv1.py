
import numpy as np
import pylab as plt
from keras.datasets import mnist
from keras.datasets import cifar10

#######################

# stride 1.
fh = 30
fw = 30

#######################

class Memristors:
    def __init__(self, size, init):
        self.size = size
        self.U = 1e-16
        self.D = 10e-9
        self.W0 = self.D * init
        self.RON = 4e3
        self.ROFF = 10e3
        self.P = 5            
        self.W = np.ones(shape=self.size) * self.W0

    def step(self, V, dt):
        self.R = self.RON * (self.W / self.D) + self.ROFF * (1 - (self.W / self.D))
        I = V / self.R
        I = np.sum(I)
        return I

#######################

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
img = x_train[12] / 255.
# img = img[0, 0:30, 0:30]
img = img[0, :, :]

#######################
'''
(x_train, y_train), (x_test, y_test) = mnist.load_data()
img = x_train[12] / 255.
# we want this to be 30x30 ... because we arnt striding.
img = np.pad(img, 2, mode='constant')
'''
#######################

kernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])

#######################

steps = 1500
T = 20 * 1e-3
dt = T / steps
Ts = np.linspace(0, T, steps)

Ms = [[None] * fh] * fw
for ii in range(fw):
    for jj in range(fh):
        Ms[ii][jj] = Memristors(size=(3, 3), init=kernel)
    
#######################

Is = np.zeros(shape=(fw, fh))

kernel_size = 3
for ii in range(fw):
    for jj in range(fh):
        
        print (ii, jj)
        
        x1 = ii 
        x2 = (ii + kernel_size) 
        
        y1 = jj 
        y2 = (jj + kernel_size)
        
        V = img[x1:x2, y1:y2]
        I = Ms[ii][jj].step(V, dt)
        Is[ii][jj] = I

plt.imshow(Is, cmap=plt.cm.gray)
plt.show()
















