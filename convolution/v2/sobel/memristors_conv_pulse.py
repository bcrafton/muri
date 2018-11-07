
import numpy as np
import pylab as plt
from keras.datasets import mnist
from collections import deque

#######################

# 28x28 -> pad 1 -> 30x30 -> 3x3 filters no stride -> 10x10
rows = 10
cols = 10

#######################

class VMemristors:
    def __init__(self, size):
        self.size = size
        self.U = 1e-16
        self.D = 10e-9
        self.W0 = 5e-9
        self.RON = 4e3
        self.ROFF = 10e3
        self.P = 5            
        self.W = np.ones(shape=(self.size)) * self.W0
        self.R = self.RON * (self.W / self.D) + self.ROFF * (1 - (self.W / self.D))
        
    # def step(self, V, dt):
    def step(self, I, dt):
        self.R = self.RON * (self.W / self.D) + self.ROFF * (1 - (self.W / self.D))
        # I = V / self.R
        V = I * self.R
        F = 1 - (2 * (self.W / self.D) - 1) ** (2 * self.P)
        dwdt = ((self.U * self.RON * I) / self.D) * F
        dwdt += 0.1 * (self.W0 - self.W)
        self.W += dwdt * dt
        return I

class NVMemristors:
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

(x_train, y_train), (x_test, y_test) = mnist.load_data()
img = x_train[12] / 255.
# we want this to be 30x30 ... because we arnt striding.
# print (np.shape(img))
img = np.pad(img, 1, mode='constant')

# kernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])

# kernel = np.array([[2.,1.,0.],[4.,2.,0.],[2.,1.,0.]])
# kernel = kernel / 4.

kernel = np.array([[0.,1.,2.],[0.,2.,4.],[0.,1.,2.]])
kernel = kernel / 4.

#######################

steps = 1500
T = 1.
dt = T / steps
Ts = np.linspace(0, T, steps)

filters = [[None for x in range(cols)] for y in range(rows)] 
for ii in range(rows):
    for jj in range(cols):
        filters[ii][jj] = NVMemristors(size=(3, 3), init=kernel)
        
sums = [[None for x in range(cols)] for y in range(rows)] 
for ii in range(rows):
    for jj in range(cols):
        sums[ii][jj] = VMemristors(size=(1))
    
#######################

rate = 5 # 25 Hz max
spks = deque(maxlen=10)
Is = np.zeros(shape=(rows, cols))

for t in Ts:
    spk = np.random.rand(30, 30) < img * rate * dt
    spks.append(spk)
    # sum or max?
    spk_img = np.max(spks, axis=0)
    # make sure its (30, 30)
    assert(np.shape(spk_img) == (30, 30))
    
    for ii in range(rows):
        for jj in range(cols):
            
            x1 = ii * 3
            x2 = (ii + 1) * 3
            
            y1 = jj * 3
            y2 = (jj + 1) * 3
            
            V = spk_img[x1:x2, y1:y2] * 1.
            I = filters[ii][jj].step(V, dt)
            Is[ii][jj] = I

            sums[ii][jj].step(I, dt)
            
    print (Is)

#######################

Rs = np.zeros(shape=(rows, cols))
for ii in range(rows):
    for jj in range(cols):
        Rs[ii][jj] = sums[ii][jj].R

#######################

print (Rs)

plt.imshow(Rs, cmap=plt.cm.gray)
plt.show()
















