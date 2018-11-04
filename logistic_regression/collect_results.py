
import numpy as np
import os
import threading

################################################

epochs = [10]
alphas = [0.0005] # alphas = [0.0005, 0.001, 0.005]
scales = [2., 4., 8., 16., 32.]
lows = np.linspace(0.005, 0.1, 10)
pcas = [100, 150, 200, 250, 300, 350, 400]

################################################

table = []
for epoch in epochs:
    for scale in scales:
        for low in lows:
            for alpha in alphas:
                for pca in pcas:
                    name = "./results/epochs_%d_alpha_%f_scale_%f_low_%f_pca_%d.npy"% (epoch, alpha, scale, low, pca)
                    acc = np.load(name)
                    acc = np.max(acc)
                    table.append( (epoch, alpha, scale, low, pca, acc) )
                    print ("epochs %d alpha %f scale %f low %f pca %d acc %f" % (epoch, alpha, scale, low, pca, acc))
     

header = "epochs,alpha,scale,low,pca,acc"
np.savetxt(fname='results.csv', X=table, delimiter=',', header=header)

################################################

scale_idx = {}
for ii in range(len(scales)):
    scale = scales[ii]
    scale_idx[scale] = ii
    
pca_idx = {}
for ii in range(len(pcas)):
    pca = pcas[ii]
    pca_idx[pca] = ii

print (scale_idx, pca_idx)

shape = (len(scales), len(pcas))
mat = np.zeros(shape=shape)

for tup in table:
    scale = scale_idx[int(tup[2])]
    pca = pca_idx[int(tup[4])]
    acc = tup[5]
    
    mat[scale][pca] = max(acc, mat[scale][pca])

np.savetxt(fname='results.csv', X=mat, delimiter=',')

################################################

'''
scale_idx = 2 
pca_idx = 4
acc_idx = 5

shape = (len(scales), len(pcas))
mat = {}

for tup in table:
    scale = int(tup[scale_idx])
    pca = int(tup[pca_idx])
    acc = tup[acc_idx]
    if (scale, pca) in mat.keys():
        mat[(scale, pca)] = max(acc, mat[(scale, pca)])
    else:
        mat[(scale, pca)] = acc
    
print (mat)
'''

################################################
