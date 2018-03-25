#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 11:24:41 2018

@author: dulan
"""

import numpy as np

#X = np.array([[0,0,1],
#              [0,1,0],
#              [1,0,1],
#              [1,1,0]])
#
#y = np.array([[0,0,1,1]]).T
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

y = np.array([[0,1,1,0]]).T

syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

def sigmoid(val):
    return 1/(1+np.exp(-val))

def sig_der(val):
    return val*(1-val)

for j in xrange(60000):
    l0 = X
    
    l1 = sigmoid(np.dot(X,syn0))
    l2 = sigmoid(np.dot(l1,syn1))
    
    l2_error = y-l2
    l2_delta = l2_error * sig_der(l2)
    l1_delta = l2_delta.dot(syn1.T) * sig_der(l1)
    
    syn1 += np.dot(l1.T,l2_delta)
    syn0 += np.dot(l0.T,l1_delta)
    
print l2