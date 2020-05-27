import math

def elastic_net_sensitivity(kappa, n, lambda2, model):
    if model in ['svm-hinge', 'svm-huber']:
        L=1
    if model in ['log']:
        L=kappa
    beta=(2*(L**2)*kappa)/(n*lambda2)
    return math.sqrt((2*beta)/lambda2)
