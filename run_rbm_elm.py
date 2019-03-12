# -*- coding: utf-8 -*-
"""

Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This code will perform the RBM-ELM and compare its results with the ELM and ELM-RO.
To run this code, you need to clone my RBM and ELM repositories (check the README.md)

If you find any bug or have any question, let me know: pacheco.comp@gmail.com

"""

import sys

# Insert the path to RBM, ELM and utils. You can find them in my repositories:
sys.path.insert (0, '../RBM')
sys.path.insert (0, '../ELM')


import numpy as np
from elm import *
from elm_tensorflow import *
from utilsClassification import *
from rbm import *
from rbm_tensorflow import *
import time
import gc
import tensorflow as tf


# Set the parameters. All parameters are better described in the paper and in the RBM implementation
it = 30
hidNeurons = 250
maxIterRbm = 15
lrRbm = 0.01
wcRbm = 0.01
iMomRbm = 0.5
fMomRbm=0.9
cdIterRbm = 1
batchSizeRbm = 25
freqPrint = 10 # the frequece to print information about the training
###########################################################################

# Load the dataset you wanna run. You must have a train testing partitions.
# You can get the input and output for each partition here in code, but if you have in disk, you can also load them.
print 'Loading the dataset...'
dataTestIn = np.genfromtxt('data_test.csv', delimiter=',')
dataTestOut = np.genfromtxt('data_test.csv', delimiter=',')

dataTrainIn = np.genfromtxt('data_train.csv', delimiter=',')
dataTrainOut = np.genfromtxt('data_train.csv', delimiter=',')


# Some list to get statistics
acc = list()
tim = list()

acc2 = list()
tim2 = list()

acc3 = list()
tim3 = list()


normRBMELM = list()
normELM = list()
normELMRO = list()

# If you wann use tensorflow: RBM_TF, otherwise, use just RBM. The same is valid for ELM.
for i in range(it):

    
    ###########################################################################
    print 'Starting training RBM ', i , ' ...'  
    
    init = time.time() # getting the start time
    rbmNet = RBM_TF (dataIn=dataTrainIn, numHid=hidNeurons, rbmType='GBRBM')    
    rbmNet.train (maxIter=maxIterRbm, lr=lrRbm, wc=wcRbm, iMom=iMomRbm, fMom=fMomRbm, cdIter=cdIterRbm, batchSize=batchSizeRbm, freqPrint=freqPrint)
    W = np.concatenate ((rbmNet.getWeights(), rbmNet.getHidBias()), axis = 0)        
    del(rbmNet)    
    
    print 'Starting training RBM-ELM ', i , ' ...' 
    elmNet = ELM (hidNeurons, dataTrainIn, dataTrainOut, W)
    elmNet.train(aval=True)    
    end = time.time() # getting the end time     
    res, a = elmNet.getResult (dataTestIn,dataTestOut,True)
    
    nor,_ = elmNet.getNorm()
    normRBMELM.append(nor)  
    acc.append(a)
    tim.append(end-init)    
    del(elmNet)
    
    ###########################################################################
    print '\n\n'
    print 'Starting training ELM ', i , ' ...' 
    init2 = time.time()
    elmNet = ELM (hidNeurons, datatrainIn, datatrainOut)
    elmNet.train(aval=True)    
    end2 = time.time() # getting the end time     
    res, a = elmNet.getResult (datatestIn,datatestOut,True)   
    
    nor,_ = elmNet.getNorm()
    normELM.append(nor)
    acc2.append(a)
    tim2.append(end2-init2)    
    del(elmNet)

    ###########################################################################
    print '\n\n'
    print 'Starting training ELM-RO ', i , ' ...' 
    init3 = time.time()    
    elmNet = ELM (hidNeurons, datatrainIn, datatrainOut, init='RO')
    elmNet.train(aval=True)     
    end3 = time.time()    
    res, a = elmNet.getResult (datatestIn,datatestOut,True) 
    
    nor,_ = elmNet.getNorm()
    normELMRO.append(nor)
    acc3.append(a)
    tim3.append(end3-init3)
    del(elmNet)
    
    ###########################################################################   
    print '\nIteration time: ', end3-init, ' sec', 'Predict to end: ', (end3-init)*(it-i)/60, ' min'
    
    gc.collect()
	
print '######### Statistics ',maxIterRbm, ' ############' 
print 'Both RBM-ELM:'
acc = np.asarray(acc)
tim = np.asarray(tim)
normRBMELM = np.asarray(normRBMELM)
print 'Accuracy - Mean: ', acc.mean(), ' | Std: ', acc.std()
print 'Time - Mean ', tim.mean(), ' | Std: ', tim.std()
print 'Norm - Mean ', normRBMELM.mean(), ' | Std: ', normRBMELM.std()

print '\nOnly ELM:'
acc2 = np.asarray(acc2)
tim2 = np.asarray(tim2)
normELM = np.asarray(normELM)
print 'Accuracy -  mean: ', acc2.mean(), '| Std: ', acc2.std()
print 'Time - mean: ', tim2.mean(), ' | Std: ', tim2.std()
print 'Norm - Mean ', normELM.mean(), ' | Std: ', normELM.std()

print '\nOnly ELM-RO:'
acc3 = np.asarray(acc3)
tim3 = np.asarray(tim3)
normELMRO = np.asarray(normELMRO)
print 'Accuracy -  mean: ', acc3.mean(), '| Std: ', acc3.std()
print 'Time - mean: ', tim3.mean(), ' | Std: ', tim3.std()
print 'Norm - Mean ', normELMRO.mean(), ' | Std: ', normELMRO.std()
