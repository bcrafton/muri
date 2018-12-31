
import numpy as np
import pylab as plt

class Memristors:
    
    def __init__(self, N, M):
        self.U = 5e-14
        self.D = 10e-9
        self.W0 = 5e-9
        self.RON = 1e3
        self.ROFF = 5e4
        self.P = 5            
        self.W = np.ones(shape=(N, M)) * self.W0
        
        self.Rs = []
        
    def step(self, V, dt):
        R = self.RON * (self.W / self.D) + self.ROFF * (1 - (self.W / self.D))
        I = V / R
        # when W = D, F = 0 and we cannot go anywhere bc dwdt has F in product.
        F = 1 - (2 * (self.W / self.D) - 1) ** (2 * self.P)
        dwdt = ((self.U * self.RON * I) / self.D) * F
        # self.W = np.clip(self.W + dwdt * dt, 0., self.D)
        self.W = np.clip(self.W + dwdt * dt, 1e-9, 9e-9)
        
        # print ('V: %f I: %f W: %f DW: %f F: %f' % (V[0, 0], I[0, 0], self.W[0, 0], dwdt[0, 0] * dt, F[0, 0]))
        print ('V: %0.6f W: %0.10f DW: %0.12f' % (V[0, 0], self.W[0, 0], dwdt[0, 0]))
        
        self.Rs.append(R[0, 0])
        
        return I
      
      
# we are trying to make 100us and 400 pulses go from high to low
# bc that will losely model their memristor
        
# i think U, D, W all have to be adjusted together
# cannot just change U and expect it to work.
# eh actually: self.RON * I ... is just V.
        
dt = 100e-6
steps = 400
T = steps * dt
Ts = np.linspace(0, T, steps)

M = Memristors(10, 10)

Vs = []
Is = []

for t in Ts:
    sign = np.sin(2 * np.pi * t / T) > 0
    if sign:
        V = np.ones(shape=(10, 10))
    else: 
        V = -np.ones(shape=(10, 10))
    
    # V = np.sin(2 * np.pi * t / T) * np.ones(shape=(10, 10))
    I = M.step(V, dt)
    
    # print (V[0, 0], I[0, 0])
    
    Vs.append(V[0, 0])
    Is.append(I[0, 0])
    
plt.subplot(2, 1, 1)
plt.plot(Vs, Is)
plt.subplot(2, 1, 2)
plt.plot(Ts, M.Rs)

plt.show()
