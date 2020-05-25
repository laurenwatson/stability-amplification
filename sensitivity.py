import math

def elastic_net_sensitivity(d, kappa, n, lambda2, model):
    if model in ['svm-hinge']:
        L=1
    beta=(2*(L**2)*kappa)/(n*lambda2)
    return math.sqrt((2*d*beta)/lambda2)
