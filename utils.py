import matplotlib
import numpy as np
import pandas as pd
import math

def add_noise(weights, s, epsilon):
    noise_vector=np.random.laplace(0, s, weights.shape)
    return weights+noise_vector

def calculate_regularized_sensitivity(n,L, k, d, lam):
    return (4*L*k*math.sqrt(d))/(n*lam)

def calculate_unregularized_sensitivity(b, d, lam):
        return math.sqrt((2*d*b)/lam)
