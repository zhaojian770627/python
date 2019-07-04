import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import std


def normalize(X):
    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean) / std
    return X


boston = tf.contrib.learn.datasets.load_dataset('boston')
X_train,Y_train=boston.data[:,5],boston.target
#X_train=normalize(X_train) # This step is optional here
print(X_train.shape[0])
n_samples=len(X_train) 
print(n_samples)