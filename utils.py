import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.base import clone
from sklearn import preprocessing
from sklearn.model_selection import learning_curve
import statistics

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
    # print('Test Accuracy', accuracy_score(y, new_y))
    return accuracy_score(y, new_y)

def private_model(x, y, original_model, s, epsilon, rand=None):
    new_model=clone(original_model)
    new_model.fit(x, y)
    new_model.coef_=add_noise(new_model.coef_, s, epsilon)
    new_model.intercept_=add_noise(new_model.intercept_, s, epsilon)
    return new_model

def plot_conf_matrix(x, y, model, save=False, fname=''):
    if save == False:
        plot_confusion_matrix(model, x, y, normalize='true', cmap='Blues')
        plt.show()
    else:
        plot_confusion_matrix(model, x, y, normalize='true', cmap='Blues')
        plt.savefig(fname)



def plot_datasize_curve(estimator, private_estimator,title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.05, 1.0, 5)):

    if axes is None:
        _, ax = plt.subplots(figsize=(5, 5))

    ax.set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    ax.set_xlabel("No. Training examples")
    ax.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    priv_train_sizes, priv_train_scores, priv_test_scores, priv_fit_times, _ = \
        learning_curve(private_estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ptrain_scores_mean = np.mean(priv_train_scores, axis=1)
    ptrain_scores_std = np.std(priv_train_scores, axis=1)
    ptest_scores_mean = np.mean(priv_test_scores, axis=1)
    ptest_scores_std = np.std(priv_test_scores, axis=1)


    # Plot learning curve
    ax.grid()
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    ax.fill_between(priv_train_sizes, ptrain_scores_mean - ptrain_scores_std, ptrain_scores_mean + ptrain_scores_std, alpha=0.1,
                         color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    ax.plot(train_sizes, ptrain_scores_mean, 'o-', color="g",
                 label="Private Training score")
    ax.legend(loc="best")


    plt.show()

def plot_reg_curve(estimator, private_estimator,title, X, y, x_test, y_test, s, epsilon,no_runs=10,axes=None, ylim=None, cv=None,
                        n_jobs=None, reg_sizes=np.linspace(.01, 1.0, 10)):

    if axes is None:
        _, ax = plt.subplots(figsize=(5, 5))

    ax.set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    ax.set_xlabel("Regularization term")
    ax.set_ylabel("Score")

    train_errors_mean, priv_train_errors_mean, train_errors_std, priv_train_errors_std =[], [], [],[]
    for alpha in reg_sizes:
        errs=[]
        p_errs=[]
        for run in range(0, no_runs):
            estimator.set_params(alpha=alpha).fit(X, y)
            priv_est=private_model(X, y, estimator, s, epsilon)
            errs.append(single_model_performance(x_test, y_test, estimator))
            p_errs.append(single_model_performance(x_test, y_test, priv_est))
        train_errors_mean.append(sum(errs)/len(errs))
        priv_train_errors_mean.append(sum(p_errs)/len(p_errs))
        train_errors_std.append(statistics.stdev(errs))
        priv_train_errors_std.append(statistics.stdev(p_errs))


    ax.grid()

    ax.fill_between(reg_sizes, np.array(train_errors_mean) - np.array(train_errors_std),
                         np.array(train_errors_mean) + np.array(train_errors_std), alpha=0.1,
                         color="r")
    ax.fill_between(reg_sizes, np.array(priv_train_errors_mean) - np.array(priv_train_errors_std),
                         np.array(priv_train_errors_mean) + np.array(priv_train_errors_std), alpha=0.1,
                         color="g")
    ax.plot(reg_sizes, train_errors_mean, 'o-', color="r",
                 label="Training score")
    ax.plot(reg_sizes, priv_train_errors_mean, 'o-', color="g",
                 label="Private Training score")
    ax.legend(loc="best")


    plt.show()
