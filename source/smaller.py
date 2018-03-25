#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 11:46:48 2018

@author: dulan
"""
import numpy as np


X = np.array([[0,0,1],
              [0,1,0],
              [1,0,1],
              [1,1,0]])

y = np.array([[0,0,1,1]]).T

syn0 = 2*np.random.random((3,1)) - 1

def sigmoid(val):
    #Calculate sigmoid function
    return 1/(1+np.exp(-val))

def sig_der(val):
    #Calculate the diravative of sigmoid
    return val*(1-val)

for j in xrange(100000):
    l0 = X
    l1 = sigmoid(np.dot(X,syn0))
    l1_error = y-l1
    l1_delta = l1_error * sig_der(l1)
    delta_add = np.dot(l0.T, l1_delta)
    syn0 += delta_add
    
#print l1_delta
#print delta_add
print l1
print syn0