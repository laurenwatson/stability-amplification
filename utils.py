import matplotlib
import numpy as np
import pandas as pd
import math
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn import preprocessing

def add_noise(weights, s, epsilon):
    noise_vector=np.random.laplace(0, s/epsilon, weights.shape)
    return weights+noise_vector

def calculate_regularized_sensitivity(n,L, k, d, lam):
    return (4*L*k*math.sqrt(d))/(n*lam)

def calculate_unregularized_sensitivity(b, d, lam):
        return math.sqrt((2*d*b)/lam)

def train_logistic_reg(x, y, lam, rand=None):
    model = SGDClassifier(loss='log',penalty='l2', max_iter=1000, alpha=lam,  random_state=rand)

    model.fit(x,y)

    return model

def train_svm_reg(x, y, lam, rand=None):
    model = SGDClassifier(loss='hinge',penalty='l2', max_iter=1000, alpha=lam,  random_state=rand)
    model.fit(x,y)
    return model

def single_model_performance(x, y, model):
    new_y=model.predict(x)
    print('Test Accuracy', accuracy_score(y, new_y))

def private_model(x, y, original_model, s, epsilon, rand=None):
    new_model=clone(original_model)
    new_model.fit(x, y)
    new_model.coef_=add_noise(new_model.coef_, s, epsilon)
    new_model.intercept_=add_noise(new_model.intercept_, s, epsilon)
    return new_model
