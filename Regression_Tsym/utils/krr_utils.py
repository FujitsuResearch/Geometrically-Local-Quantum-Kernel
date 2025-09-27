import numpy as np
import math
import random as rand
import statistics
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score

# Grid search with cross validation

'''
From `num` data points (where `num` is the dimension of y), randomly select `num_train` training data points and `num_test` test data points.
Then, perform cross-validation on the training data (kfold is the number of splits for cross-validation, ite is the number of cross-validation iterations) to determine the optimal hyperparameters.
The hyperparameters to be optimized are the regularization parameter (candidate values are stored in `alpha_list`) and the kernel's own hyperparameters (multiple gram matrices with different hyperparameters are stored in `G0_list`).
Afterward, train the model using the optimal hyperparameters and output the prediction accuracy for the test data.
Repeat the above procedure `rep` times and output the mean and standard deviation of the accuracy.
- num_train: Number of training data points
- num_test: Number of test data points
- G0_list: List of gram matrices (each component corresponds to a gram matrix with different hyperparameters. The optimal one will be found from this list.)
- y: Label vector
- kfold: Number of splits for cross-validation
- ite: Number of cross-validation iterations
- rep: Number of repetitions for the entire procedure
- seed: Random seed
- alpha_list: List of regularization parameter candidates
'''
def calculate_pred_acc(num_train, num_test, G0_list, y, kfold, ite, rep, seed, alpha_list):
    rand.seed(seed)
    num = len(y)
    
    fin_acc_list = []
    # Calculate accuracy `rep` times, varying training and test data, to find the average accuracy.
    for _ in range(rep):   
        a = [i for i in range(num)]
        rand.shuffle(a)
        train_list = sorted(a[0:num_train], key=abs)
        test_list = sorted(a[num_train:num_train+num_test], key=abs)

        ## Determine the optimal alpha and hyperparameters
        G0, alpha = optimize_hyperparameters(train_list, test_list, G0_list, y, kfold, ite, alpha_list)
        
        ## Re-run KRR with the optimal G0 and calculate accuracy
        G_train, G_test, y_train, y_test = divide_data(G0, y, train_list, test_list)
        krr = KernelRidge(kernel="precomputed", alpha=alpha)
        krr.fit(G_train, y_train)
        ptest = krr.predict(G_test)
        fin_acc = r2_score(y_test, ptest)
        fin_acc_list.append(fin_acc)
    
    # Calculate mean and standard deviation for `rep` repetitions
    if rep==1:
        mean = fin_acc_list[0]
        stdv = 0.0
    else:
        mean = statistics.mean(fin_acc_list)
        stdv = statistics.stdev(fin_acc_list)
        
    #print(f"fin_acc_list = {fin_acc_list}")
    print(f"num_train = {num_train}: mean = {mean}, stdv = {stdv}")
    print(f"----------------------------")
    
    return mean, stdv



'''
Optimize hyperparameters by performing a grid search on the training data `train_list` using cross-validation.
- train_list: List of training data indices
- test_list: List of test data indices
- G0_list: List of gram matrices
- y: Label vector
- kfold: Number of splits for cross-validation
- ite: Number of cross-validation iterations
- alpha_list: List of regularization parameter candidates
'''
def optimize_hyperparameters(train_list, test_list, G0_list, y, kfold, ite, alpha_list):
    ## Determine the optimal alpha and hyperparameters
    alpha_cand = []
    acc_cand = []
    ### First, optimize alpha for each G0
    for G0 in G0_list:
        # Extract num_train training data and num_test test data from num data points
        G_train, G_test, y_train, y_test = divide_data(G0, y, train_list, test_list)
        alpha, acc = grid_search_KRR(kfold, ite, G_train, y_train, alpha_list)
        alpha_cand.append(alpha)
        acc_cand.append(acc)

    ### Next, select the optimal G0
    index = np.argmax(acc_cand) # Select the hyperparameter with the highest accuracy.
    alpha = alpha_cand[index]
    G0 = G0_list[index]
    
    return G0, alpha



'''
Extract sub-matrices of the gram matrix G0 and label vector y for the training data `train_list` and test data `test_list`.
- G0: Gram matrix
- y: Label vector
- train_list: List of training data indices
- test_list: List of test data indices
'''
def divide_data(G0, y, train_list, test_list):
    
    G_train = np.take(G0, train_list, axis=0)
    G_train = np.take(G_train, train_list, axis=1)
    
    G_test = np.take(G0, test_list, axis=0)
    G_test = np.take(G_test, train_list, axis=1)
    
    y_train = np.take(y, train_list, axis=0)
    y_test = np.take(y, test_list, axis=0)
    
    return G_train, G_test, y_train, y_test



'''
Perform a grid search for the regularization parameter alpha from `alpha_list`.
By varying alpha, perform k-fold cross-validation (k=kfold) `ite` times on the training data's gram matrix (`G_train`) and label vector (`y_train`), then determine the optimal regularization parameter alpha from their average values.
- kfold: Number of splits for cross-validation
- ite: Number of cross-validation iterations
- G_train: Gram matrix of the training data
- y_train: Label vector of the training data
- alpha_list: List of regularization parameter candidates
'''
def grid_search_KRR(kfold, ite, G_train, y_train, alpha_list):
    
    acc_list = []
    for alpha in alpha_list:
        acc = cross_validation(kfold, ite, G_train, y_train, alpha)
        acc_list.append(acc)
        #print(f"alpha = {alpha}, mse = {mse}")
        #print(f"------------------------------------")
        
    index = np.argmax(acc_list)
    print(f"Best alpha = {alpha_list[index]}, acc = {acc_list[index]}")
    
    return alpha_list[index], acc_list[index]



'''
Cross-validation
For a given regularization parameter alpha, compute the average accuracy of k-fold cross-validation (k=kfold) repeated `ite` times, using the training data's gram matrix (`G_train`) and label vector (`y_train`).
- kfold: Number of splits for cross-validation
- ite: Number of cross-validation iterations
- G_train: Gram matrix of the training data
- y_train: Label vector of the training data
- alpha: Regularization parameter
'''
def cross_validation(kfold, ite, G_train, y_train, alpha):
    
    num_data = len(y_train)
    index_list = [i for i in range(num_data)]
    
    acc_list = []
    for _ in range(ite):
        group_list = shuffle_and_divide(num_data, kfold)  
        for i in range(kfold):
            train_data_cv = sorted([x for x in index_list if x not in group_list[i]],key=abs)
            valid_data_cv = group_list[i] 

            G_train_cv = partition_train(G_train, valid_data_cv)
            G_valid_cv = partition_valid(G_train, valid_data_cv)
            y_train_cv = [y_train[j] for j in train_data_cv]
            y_valid_cv = [y_train[j] for j in valid_data_cv]

            # Training & Prediction
            krr = KernelRidge(kernel="precomputed", alpha=alpha)
            krr.fit(G_train_cv, y_train_cv)
            prediction = krr.predict(G_valid_cv)
            acc = r2_score(y_valid_cv, prediction)
            #print(f"i={i}: acc = {acc}")
            acc_list.append(acc)
        
    return statistics.mean(acc_list)



'''
Partition the training data's gram matrix `G_train` into validation data `cv_valid` and the rest.
- G_train: Gram matrix of the training data
- valid_data_cv: List of validation data indices
'''
def partition_train(G_train, valid_data_cv):
    G_train_cv = np.delete(G_train, valid_data_cv, axis=0)
    G_train_cv = np.delete(G_train_cv, valid_data_cv, axis=1)
    
    return G_train_cv

def partition_valid(G_train, valid_data_cv):
    num_data = len(G_train)
    index_list = [i for i in range(num_data)]
    train_list_cv = sorted([x for x in index_list if x not in valid_data_cv],key=abs)
    
    G_valid_cv = np.delete(G_train, train_list_cv, axis=0)
    G_valid_cv = np.delete(G_valid_cv, valid_data_cv, axis=1)
    
    return G_valid_cv



'''
Used for CV
Divide 0 to num-1 into `g` groups and create a list of lists.
For example, divide 0 to 8 into `group_list=[[3,4,7],[0,2,8],[2,4,5]]`. The inner lists are sorted.
'''
# Randomly divide 0 to num-1 into g groups
def shuffle_and_divide(num, g):
    quot = num//g 
    extra = num%g

    a = [i for i in range(num)]
    rand.shuffle(a)
    group_list = []
    ini = 0
    for i in range(g):
        if i<extra:
            b = a[ini:ini+quot+1]
            group_list.append(sorted(b, key=abs)) # Sort each group in ascending order
            ini += quot+1
        else:
            b = a[ini:ini+quot]
            group_list.append(sorted(b, key=abs))
            ini += quot
            
    return group_list


# Gram matrix normalization
def G_normalize(G):
    
    dim = len(G)
    G_norm = np.zeros((dim,dim))
    
    for i in range(dim):
        for j in range(dim):
            G_norm[i,j] = G[i,j]/np.sqrt(G[i,i]*G[j,j])
            
    return G_norm