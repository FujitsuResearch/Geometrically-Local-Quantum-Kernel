import os
os.environ["OMP_NUM_THREADS"] = "1" 
print(os.cpu_count())

import numpy as np
import math
import random as rand
import statistics
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
from utils.krr_utils import * 


import argparse
psr  = argparse.ArgumentParser()
psr.add_argument('-n', '--nqubits', required=True)
psr.add_argument('-nt', '--num_train', required=True)
psr.add_argument('-t', '--target', required=True)
args = psr.parse_args()
N = int(args.nqubits)
num_train = int(args.num_train)
target = str(args.target)



# calculate accuracy
num_test = 500
nsamp = 500
train_list = [i for i in range(num_train)]
test_list = [i for i in range(num_train,num_train+num_test)]

kfold=5
ite=1
alpha_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
we_list = [(2,1),(2,2),(4,1),(4,2),(6,1),(6,2)]

#Gram matrix
G0_list = []
for (w,e) in we_list:
    filename = f'gram_matrix_N={N}/TESK_ndata=1500_nsamp={nsamp}_w={w}_e={e}.csv'
    G0 = np.loadtxt(filename, delimiter=",", dtype=np.float64)
    G0_list.append( G_normalize(G0) )

#label vector
filename = f'gram_matrix_N={N}/{target}.csv'
y = np.loadtxt(filename, delimiter=",", dtype=np.float64)

#
rand.seed(314)
G0,alpha = optimize_hyperparameters(train_list, test_list, G0_list, y, kfold, ite, alpha_list)
G_train, G_test, y_train, y_test = divide_data(G0, y, train_list, test_list)
krr = KernelRidge(kernel="precomputed", alpha=alpha)
krr.fit(G_train, y_train)
ptest = krr.predict(G_test)
fin_acc = r2_score(y_test, ptest)
np.save(f"results/TESK_scatter_{target}_N={N}_nt={num_train}_normalize.npy", np.array([y_test, ptest]))
