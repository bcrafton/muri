
import numpy as np
import os
import threading

def run_command(epoch, scale, low, alpha, pca):
    cmd = "python mnist.py --epochs %d --scale %f --low %f --alpha %f --pca %d" % (epoch, scale, low, alpha, pca)
    os.system(cmd)
    return

################################################

epochs = [10]
alphas = [0.0005] # alphas = [0.0005, 0.001, 0.005]
scales = [2., 4., 8., 16., 32.]
lows = np.linspace(0.005, 0.1, 10)
pcas = [100, 150, 200, 250, 300, 350, 400]

print ( "num tests %d" % (len(epochs) * len(alphas) * len(scales) * len(lows) * len(pcas)) )

runs = []
for epoch in epochs:
    for scale in scales:
        for low in lows:
            for alpha in alphas:
                for pca in pcas:
                    runs.append((epoch, scale, low, alpha, pca))

num_runs = len(runs)
parallel_runs = 4

for run in range(0, num_runs, parallel_runs):
    threads = []
    for parallel_run in range(parallel_runs):
        args = runs[run + parallel_run]
        t = threading.Thread(target=run_command, args=args)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

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
                    table.append( (epoch, alpha, scale, low, acc) )
                    print ("epochs %d alpha %f scale %f low %f pca %d acc %f" % (epoch, alpha, scale, low, pca, acc))
     

header = "epochs,alpha,scale,low,pca,acc"
np.savetxt(fname='results.csv', X=table, delimiter=',', header=header)

################################################


