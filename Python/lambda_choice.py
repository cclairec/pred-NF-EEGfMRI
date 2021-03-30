# -*- coding: utf-8 -*-
"""
Estimation of the regularisation parameter : lambda 

Created on Tue Mar 23 16:02:14 2021

@author: caroline-pinte
"""

import numpy as np
from forward_backward_optimisation import forward_backward_optimisation

def lambda_choice(Dtrain, rep_train, nb_freq_band, reg_function, lambdas, disp_, logger=None) :
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
    
    ind = 1
    lambdas_end = lambdas[-1]
    for l in lambdas :
        
        if (logger==None) :
            print(l)
        else :
            logger.info("... Computing lambda = {}  /{}".format(l,lambdas_end))
        rho = l
        k_ind = 1

        for k in np.arange(0, np.shape(Dtrain)[0]-size_cv_dataset, delay_cv) :
            cv_set = np.arange(np.floor(k),np.floor(k)+size_cv_dataset+1,dtype=int)
            train_set = np.arange(0,np.shape(Dtrain)[0])
            train_set = np.delete(train_set, cv_set)
            
            Dtrain_k = Dtrain[train_set,:,:]
            rep_train_k = rep_train[train_set]
            Dcv_k = Dtrain[cv_set,:,:]
            rep_cv_k = rep_train[cv_set]
            
            SStot = np.sum( (rep_cv_k - np.mean(rep_cv_k))**2 )
            SStot_train = np.sum( (rep_train_k - np.mean(rep_train_k))**2 )
            
            if (reg_function == 'lasso') :
                logger.error("Not implemented.")
            elif (reg_function == 'fistaL1') :
                method_ = 1
                alpha = forward_backward_optimisation(Dtrain_k, rep_train_k, l, method_, rho)
            
            print('debug')
            
    
    
    
    
    
    
    
    
    
    
    
    
    return 0