# -*- coding: utf-8 -*-
"""
Estimation of the regularisation parameter : lambda 

Created on Tue Mar 23 16:02:14 2021

@author: caroline-pinte
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from forward_backward_optimisation import forward_backward_optimisation

def lambda_choice(Dtrain, rep_train, nb_freq_band, reg_function, lambdas, disp_, logger) :
    '''
    Estimate the regularisation parameter : lambda.

            Parameters:
                    Dtrain (array): design matrix for learning.
                    rep_train (array): NF variable for learning.
                    nb_freq_band (int): number of freq bands (default 10).
                    reg_function (string): regularisation function, must be 'lasso' (matlab), 'fistaL1' or 'L12'.
                    lambdas (array): list of int values to be tested.
                    disp_ (bool): display figure or not.
                    logs (handler): formatting logs.
                    
            Returns:
                    regul_lambda (int): regularisation parameter lambda.
    '''
    
    K = 10
    
    if (rep_train.ndim == 2) :
        if (np.shape(rep_train)[0] == 2) or (np.shape(rep_train)[1] == 2) :
            weights = [1/3, 1/2, 2/3]
            if (np.shape(rep_train)[0] == 2) :
                NF_1 = rep_train[0,:]
                NF_2 = rep_train[1,:]
            elif (np.shape(rep_train)[1] == 2) :
                NF_1 = rep_train[:,0]
                NF_2 = rep_train[:,1]                
    else :
        weights = -1
    
    # sparsity_threshold = np.shape(Dtrain)[1] * 40/100 # we want at most 40 % of nonzeros coefficients

    size_cv_dataset = np.ceil(np.shape(Dtrain)[0]/10)
    delay_cv = (np.shape(Dtrain)[0] - size_cv_dataset) / K
    
    ind = 0
    size_ind = len(lambdas)
    size_k_ind = len(np.arange(0, np.shape(Dtrain)[0]-size_cv_dataset, delay_cv))
    sparsity_ = np.zeros((size_ind,size_k_ind))
    CV = np.zeros((size_ind,size_k_ind))
    Cost_train = np.zeros((size_ind,size_k_ind))
    CV_mean_ = np.zeros(size_ind)
    Cost_train_mean = np.zeros(size_ind)
    sparsity_mean = np.zeros(size_ind)
    lambdas_end = lambdas[-1]
    
    # for l in lambdas :
        
    #     logger.info("... Computing lambda = {}  /{}".format(l,lambdas_end))
    #     rho = l
    #     k_ind = 0

    #     for k in np.arange(0, np.shape(Dtrain)[0]-size_cv_dataset, delay_cv) :
    #         cv_set = np.arange(np.floor(k),np.floor(k)+size_cv_dataset+1,dtype=int)
    #         train_set = np.arange(0,np.shape(Dtrain)[0])
    #         train_set = np.delete(train_set, cv_set)
            
    #         Dtrain_k = Dtrain[train_set,:,:]
    #         rep_train_k = rep_train[train_set]
    #         Dcv_k = Dtrain[cv_set,:,:]
    #         rep_cv_k = rep_train[cv_set]
            
    #         SStot = np.sum( (rep_cv_k - np.mean(rep_cv_k))**2 )
    #         SStot_train = np.sum( (rep_train_k - np.mean(rep_train_k))**2 )
            
    #         if (reg_function == 'lasso') :
    #             logger.error("Not implemented.")
    #         elif (reg_function == 'fistaL1') :
    #             method_ = 1
    #             alpha = forward_backward_optimisation(Dtrain_k, rep_train_k, l, method_, rho)
    #             sparsity_[ind,k_ind] = len(alpha[alpha!=0])
                
    #             if (len(np.shape(Dcv_k)) == 3) :
    #                 predicted_values = np.zeros(np.shape(Dcv_k)[0])
    #                 for t in range(0,np.shape(Dcv_k)[0]) :
    #                     predicted_values[t] = np.trace( np.matmul(np.squeeze(Dcv_k[t,:,:]).T,alpha) )
    #             else :
    #                 predicted_values = np.matmul(Dcv_k,alpha)
                    
    #             SSres = sum( (rep_cv_k - predicted_values)**2 )
    #             CV[ind,k_ind] = SSres/SStot
                
    #             if (len(np.shape(Dtrain_k)) == 3) :
    #                 predicted_values = np.zeros(np.shape(Dtrain_k)[0])
    #                 for t in range(0,np.shape(Dtrain_k)[0]) :
    #                     predicted_values[t] = np.trace( np.matmul(np.squeeze(Dtrain_k[t,:,:]).T,alpha) )
    #             else :
    #                 predicted_values = np.matmul(Dtrain_k,alpha)

    #             SSres_train = sum( (rep_train_k - predicted_values)**2 )
    #             Cost_train[ind,k_ind] = SSres_train/SStot_train
                
    #         k_ind = k_ind + 1
        
    #     CV_mean_[ind] = np.mean(CV[ind,:])
    #     Cost_train_mean[ind] = np.mean(Cost_train[ind,:])
    #     sparsity_mean[ind] = np.mean(sparsity_[ind,:])
        
    #     if (sparsity_mean[ind] <= 2) :
    #         logger.info("Breaking at lambda = {}".format(l))
    #         ind = ind + 1
    #         break

    #     ind = ind + 1
    
    ########### debug ###########"
    # with open("CV_mean_.txt", "wb") as fp:   #Pickling
    #     pickle.dump(CV_mean_, fp)
    
    # with open("Cost_train_mean.txt", "wb") as fp:   #Pickling
    #     pickle.dump(Cost_train_mean, fp)
        
    with open("CV_mean_.txt", "rb") as fp:   # Unpickling
        CV_mean_ = pickle.load(fp)
        
    with open("Cost_train_mean.txt", "rb") as fp:   # Unpickling
        Cost_train_mean = pickle.load(fp)
        
    biais_var = (CV_mean_ + Cost_train_mean) / 2

    sort_biais_var = biais_var.copy()
    ind_sorted = [i[0] for i in sorted(enumerate(sort_biais_var), key=lambda x:x[1])]
    
    if (disp_ == 1) :      
        # plot 1
        plt.plot(lambdas, CV_mean_, label = "cv error", color='blue')  
        # plot 2
        plt.plot(lambdas, Cost_train_mean, label = "training error", color='red')
        # plot 3
        plt.plot(lambdas, biais_var, label = "cv error + training error", color='yellow', marker='.')
          
        plt.xlabel('lambda')
        plt.ylabel('error')
        plt.title('lambda choice')
        plt.legend()
        # plt.show() # out of this function to save the fig
    
    regul_lambda = lambdas[ind_sorted[0]]
    
    return regul_lambda