#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 16:07:05 2022

@author: ying
"""

import numpy as np
import pandas as pd 
import random
import sys
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from utils import gen_data, BH


sig_id = int(sys.argv[1]) - 1
nt_id = int(sys.argv[2]) - 1
set_id = int(sys.argv[3])  
q = int(sys.argv[4]) / 10 
seed = int(sys.argv[5])
    
    
n = 1000
ntests = [10, 100, 500, 1000]
ntest = ntests[nt_id]
sig_seq = np.linspace(0.1, 1, num = 10)
sig = sig_seq[sig_id]
reg_names = ['gbr', 'rf', 'svm']
 
all_res = pd.DataFrame()

out_dir = "../results/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    print("Output diretory created!")
    
random.seed(seed)


for reg_method in range(3):
    reg_method = reg_method + 1
    reg_name = reg_names[reg_method - 1] 
    
    Xtrain, Ytrain, mu_train = gen_data(set_id, n, sig)
    Xcalib, Ycalib, mu_calib = gen_data(set_id, n, sig)
    
    Xtest, Ytest, mu_test = gen_data(set_id, ntest, sig)
    
    # training the prediction model
    if reg_method == 1:
        regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=0)
    if reg_method == 2:
        regressor = RandomForestRegressor(max_depth=5, random_state=0)
    if reg_method == 3:
        regressor = SVR(kernel="rbf", gamma=0.1)
    
    regressor.fit(Xtrain, 1*(Ytrain>0))
    
    # calibration 
    calib_scores = Ycalib - regressor.predict(Xcalib) 
    calib_scores0 = - regressor.predict(Xcalib) 
    calib_scores_clip = Ycalib * (Ycalib > 0) - regressor.predict(Xcalib)
    calib_scores_2clip = 1000 * (Ycalib > 0) - regressor.predict(Xcalib)
     
    test_scores = - regressor.predict(Xtest) 
    
    # BH using residuals
    BH_res= BH(calib_scores, test_scores, q )
    # summarize
    if len(BH_res) == 0:
        BH_res_fdp = 0
        BH_res_power = 0
    else:
        BH_res_fdp = np.sum(Ytest[BH_res] < 0) / len(BH_res)
        BH_res_power = np.sum(Ytest[BH_res] >= 0) / sum(Ytest >= 0)
        
    
    # only use relevant samples to calibrate
    BH_rel = BH(calib_scores0[Ycalib <= 0], test_scores, q )
    if len(BH_rel) == 0:
        BH_rel_fdp = 0
        BH_rel_power = 0
    else:
        BH_rel_fdp = np.sum(Ytest[BH_rel] < 0) / len(BH_rel)
        BH_rel_power = np.sum(Ytest[BH_rel] >= 0) / sum(Ytest >= 0)
    
        
    # use clipped scores
    BH_2clip = BH(calib_scores_2clip, test_scores, q )
    if len(BH_2clip) == 0:
        BH_2clip_fdp = 0
        BH_2clip_power = 0
    else:
        BH_2clip_fdp = np.sum(Ytest[BH_2clip] < 0) / len(BH_2clip)
        BH_2clip_power = np.sum(Ytest[BH_2clip] >= 0) / sum(Ytest >= 0)
    

    all_res = pd.concat((all_res, 
                         pd.DataFrame({'BH_res_fdp': [BH_res_fdp], 
                                       'BH_res_power': [BH_res_power],
                                       'BH_res_nsel': [len(BH_res)],
                                       'BH_rel_fdp': [BH_rel_fdp], 
                                       'BH_rel_power': [BH_rel_power], 
                                       'BH_rel_nsel': [len(BH_rel)], 
                                       'BH_2clip_fdp': [BH_2clip_fdp], 
                                       'BH_2clip_power': [BH_2clip_power], 
                                       'BH_2clip_nsel': [len(BH_2clip)],
                                       'q': [q], 'regressor': [reg_name],
                                       'seed': [seed], 'sigma': [sig], 'ntest': [ntest]})))


all_res.to_csv("../results/prob_set"+str(set_id)+"q"+str(int(q*10))+"sig"+str(sig_id)+"nt"+str(nt_id)+"seed"+str(seed)+".csv")


















