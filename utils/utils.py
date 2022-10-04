#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 09:54:03 2022

@author: ying
"""

import numpy as np
import pandas as pd 


def gen_data(setting, n, sig): 
    X = np.random.uniform(low=-1, high=1, size=n*20).reshape((n,20))
    
    if setting == 1: 
        # mu_x = (X[:,0] * X[:,1] > 0 ) * (X[:,3]>0.5) * (0.25+X[:,3]) + (X[:,0] * X[:,1] <= 0 ) * (X[:,3]<-0.5) * (X[:,3]-0.25)
        mu_x = (X[:,0] * X[:,1] > 0 ) * (X[:,3]*(X[:,3]>0.5) + 0.5*(X[:,3]<=0.5)) + (X[:,0] * X[:,1] <= 0 ) * (X[:,3]*(X[:,3]<-0.5) - 0.5*(X[:,3]>-0.5))
        mu_x = mu_x * 4
        Y = mu_x + np.random.normal(size=n) * sig
        # plt.scatter(mu_x, Y)
        return X, Y, mu_x
    
    if setting == 2:
        mu_x = (X[:,0] * X[:,1] + np.exp(X[:,3] - 1)) * 5
        Y = mu_x + np.random.normal(size=n) * 1.5 * sig 
        return X, Y, mu_x
    if setting == 3:
        mu_x = (X[:,0] * X[:,1] + np.exp(X[:,3] - 1)) * 5
        Y = mu_x + np.random.normal(size=n) * (5.5 - abs(mu_x))/2 * sig 
        return X, Y, mu_x
    
    if setting == 4:
        mu_x = (X[:,0] * X[:,1] + np.exp(X[:,3] - 1)) * 5
        sig_x = 0.25 * mu_x**2 * (np.abs(mu_x) < 2) + 0.5 * np.abs(mu_x) * (np.abs(mu_x) >= 1)
        Y = mu_x + np.random.normal(size=n) * sig_x * sig
        return X, Y, mu_x
    
    if setting == 5:
        mu_x = (X[:,0] * X[:,1] > 0 ) * (X[:,3]>0.5) * (0.25+X[:,3]) + (X[:,0] * X[:,1] <= 0 ) * (X[:,3]<-0.5) * (X[:,3]-0.25)
        mu_x = mu_x  
        Y = mu_x + np.random.normal(size=n) * sig
        return X, Y, mu_x
    
    if setting == 6:
        mu_x = (X[:,0] * X[:,1] + X[:,2]**2 + np.exp(X[:,3] - 1) - 1) * 2
        Y = mu_x + np.random.normal(size=n) * 1.5 * sig 
        return X, Y, mu_x
    
    if setting == 7:
        mu_x = (X[:,0] * X[:,1] + X[:,2]**2 + np.exp(X[:,3] - 1) - 1) * 2
        Y = mu_x + np.random.normal(size=n) * (5.5 - abs(mu_x))/2 * sig 
        return X, Y, mu_x
    
    if setting == 8:
        mu_x = (X[:,0] * X[:,1] + X[:,2]**2 + np.exp(X[:,3] - 1) - 1) * 2
        sig_x = 0.25 * mu_x**2 * (np.abs(mu_x) < 2) + 0.5 * np.abs(mu_x) * (np.abs(mu_x) >= 1)
        Y = mu_x + np.random.normal(size=n) * sig_x * sig
        return X, Y, mu_x
     
 

def BH(calib_scores, test_scores, q = 0.1):
    ntest = len(test_scores)
    ncalib = len(calib_scores)
    pvals = np.zeros(ntest)
    
    for j in range(ntest):
        pvals[j] = (np.sum(calib_scores < test_scores[j]) + np.random.uniform(size=1)[0] * (np.sum(calib_scores == test_scores[j]) + 1)) / (ncalib+1)
         
    
    # BH(q) 
    df_test = pd.DataFrame({"id": range(ntest), "pval": pvals}).sort_values(by='pval')
    
    df_test['threshold'] = q * np.linspace(1, ntest, num=ntest) / ntest 
    idx_smaller = [j for j in range(ntest) if df_test.iloc[j,1] <= df_test.iloc[j,2]]
    
    if len(idx_smaller) == 0:
        return(np.array([]))
    else:
        idx_sel = np.array(df_test.index[range(np.max(idx_smaller)+1)])
        return(idx_sel)
    
