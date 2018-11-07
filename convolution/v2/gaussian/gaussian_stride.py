
import numpy as np
import pylab as plt
from keras.datasets import mnist
from keras.datasets import cifar10
import sympy
from PIL import Image

#######################

digit = True

#######################

# stride 1.
fh = 30
fw = 30

#######################

V1, V2, V3, V4, V5, V6, V7, V8, V9 = sympy.symbols('V1, V2, V3, V4, V5, V6, V7, V8, V9')
R1, R2, R3, R4, R5, R6, R7, R8, R9 = sympy.symbols('R1, R2, R3, R4, R5, R6, R7, R8, R9')
I = sympy.symbols('I')
Vx = sympy.symbols('Vx')
Rx = sympy.symbols('Rx')

unknowns = [I, Vx]
eq1 = ((V1 - Vx) / R1) + ((V2 - Vx) / R2) + ((V3 - Vx) / R3) + ((V4 - Vx) / R4) + ((V5 - Vx) / R5) + ((V6 - Vx) / R6) + ((V7 - Vx) / R7) + ((V8 - Vx) / R8) + ((V9 - Vx) / R9) - I
      
eq2 = (Vx / Rx - I)
solution = sympy.solve([eq1, eq2], unknowns) 

'''
print (solution)

subs = {V1:1., V2:1., V3:1., V4:1., V5:1., V6:1., \
        R1:1., R2:1., R3:1., R4:1., R5:1., R6:1., \
        Rx:1.
        }

print (solution[Vx].evalf(subs=subs))
print (solution[Vy].evalf(subs=subs))
print (solution[I].evalf(subs=subs))
'''
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

if digit:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    img = x_train[12] / 255.
    # we want this to be 30x30 ... because we arnt striding.
    img = np.pad(img, 2, mode='constant')

else:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    img = x_train[12] / 255.
    # img = img[0, 0:30, 0:30]
    img = img[0, :, :]

#######################

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

kernel_size = 3
stride = 2

_Is = np.zeros(shape=(int(fw / stride), int(fh / stride)))

for ii in range(0, fw, stride):
    for jj in range(0, fw, stride):
    
        print (ii, jj)
        
        x1 = ii
        x2 = (ii + kernel_size) 
        
        y1 = jj
        y2 = (jj + kernel_size)
        
        V = img[x1:x2, y1:y2]
        print (np.shape(V))
        # I = Ms[ii][jj].step(V, dt)
        
        subs = {V1:V[0][0], V2:V[1][0], V3:V[2][0],       \
                V4:V[0][1], V5:V[1][1], V6:V[2][1],       \
                V7:V[0][2], V8:V[1][2], V9:V[2][2],       \
                R1:1, R2:2, R3:1,                         \
                R4:2, R5:4, R6:2,                         \
                R7:1, R8:2, R9:1,                         \
                Rx:1.                                     \
                }
                
        _I = solution[I].evalf(subs=subs)
        _Is[ii][jj] = _I

# plt.imshow(_Is, cmap=plt.cm.gray)
# plt.show()

if digit:
    plt.imsave("mnist.png", img, cmap=plt.cm.gray) 
    img = Image.open('mnist.png')
    basewidth = 300
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save('mnist.png')

    plt.imsave("mnist_conv.png", _Is, cmap=plt.cm.gray) 
    img = Image.open('mnist_conv.png')
    basewidth = 300
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save('mnist_conv.png')

else:
    plt.imsave("horse.png", img, cmap=plt.cm.gray) 
    img = Image.open('horse.png')
    basewidth = 300
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save('horse.png')

    plt.imsave("horse_conv.png", _Is, cmap=plt.cm.gray) 
    img = Image.open('horse_conv.png')
    basewidth = 300
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save('horse_conv.png')













