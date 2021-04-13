# -*- coding: utf-8 -*-
"""
Basic Pursuit with Forward-Backward

Created on Tue Mar 30 11:19:42 2021

@author: caroline-pinte
"""

import numpy as np

#====================================================================
# Utilities
#====================================================================

def F(x,l) :
    return l * np.linalg.norm(x,1)

#=========

# Not tested
def G(A,x,y) :
    return 1/2 * np.linalg.norm(y-A*x,2)**2

def G3(A,x,y) :
    G = 0
    for i in range(0,np.shape(A)[0]) :
        G = G + (y[i] - np.trace( np.matmul(np.squeeze(A[i,:,:]).T,x) ))**2
    return G

#=========

# Not tested
def GradG(A,x,y) :
    return np.matmul(A.T , (np.matmul(A,x)-y))

def GradG3(A,x,y) :
    gradG = np.zeros(np.shape(x))
    for i in range(0,np.shape(A)[0]) :
        gradG = gradG + np.squeeze(A[i,:,:])*(np.trace( np.matmul(np.squeeze(A[i,:,:]).T,x) ) - y[i])
    return gradG

#=========

def prox_L21_1(x,l,mu) :
    prox = np.zeros((np.shape(x)[0],np.shape(x)[1]))
    for p in range(0,np.shape(x)[0]) :
        for k in range(0,np.shape(x)[1]) :
            div_ = 0
            for kk in range(0,np.shape(x)[1]) :
                div_ = div_ + max(abs(x[p,kk])-mu , 0)**2
            div_ = np.sqrt(div_)
            if (div_ == 0) :
                frac = 0
            else :
                frac = l/div_
            prox[p,k] = np.sign(x[p,k]) * max(abs(x[p,k])-mu , 0) * max((1 - frac) , 0)
    return prox
 
def ProxF(x,tau,l,rho) :
    return prox_L21_1(x, l*tau, rho*tau)

#=========

def perform_fb(x, L, method, niter, A, y, l, rho) :
    eps = 0.0003 # or 0.0004 if too slow
    
    t = 1
    Y = x.copy()
    
    R = []
    for i in range(0,niter) :
        #print(i)
        f = F(x,l)
        if (len(np.shape(A)) == 3) :
            g = G3(A,x,y)
        else :
            g = G(A,x,y)
        R.append( f+g ) 
        
        if (method == 'fista') :
            if (len(np.shape(A)) == 3) :
                grad = GradG3(A,Y,y)
            else :
                grad = GradG(A,Y,y)
            xnew = ProxF( Y - 1/L*grad, 1/L, l, rho );
            tnew = (1+np.sqrt(1+4*t**2)) / 2
            Y = xnew + (t-1)/(tnew)*(xnew-x)
            x = xnew
            t = tnew
            # stopping criteria
            if (i>1) and ((R[i-1]-R[i])/R[i] < eps) :
                 break

        else :
            print('Error in perform_fb : name isnt fista')
            return
    return x

#====================================================================
# forward_backward_optimisation
#====================================================================
def forward_backward_optimisation(A, y, l, method_i=1, rho=1500) :
    '''
    Compute the optimal activation pattern : alpha.

            Parameters:
                    A (array): slice of design matrix for learning.
                    y (array): slice of NF variable for learning.
                    l (int): lambda to be tested.
                    method_i (int): regularisation function, 1 corresponds to fistaL1.
                    rho (int): regularisation parameter rho.
                    
            Returns:
                    alpha (array): optimal activation pattern
    '''
    
    n = np.shape(A)[1]*np.shape(A)[2]
    
    # Lipschitz constant.
    if (len(np.shape(A)) == 2) :
        L = np.linalg.norm(A,2)
    elif (len(np.shape(A)) == 3) :
        # size AA: d2 x d1*d3
        test = np.squeeze(A[0,:,:])
        for i in range(1,np.shape(A)[0]) :
            s = np.squeeze(A[i,:,:])
            test = np.hstack((test,s))

        U, S, V = np.linalg.svd(test)
        L = max(S)**2

    # List of benchmarked algorithms.
    methods = ['fb', 'fista', 'nesterov']
    
    method = methods[method_i]
    niter = 6000
        
    x_init = np.zeros((n,1))
    if (len(np.shape(A)) == 3) :
        x_init = np.zeros( (np.shape(A)[1], np.shape(A)[2]) )

    x = perform_fb(x_init, 2*L, method, niter, A, y, l, rho)
        
    alpha = x
    return alpha