import numpy as np
import scipy.stats

def bytecode_matrix_lstsqr(x, y):
    X = np.vstack([x, np.ones(len(x))]).T
    return (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(y)


def bytecode_auto_numpy_lstsqr(x, y):
    X = np.vstack([x, np.ones(len(x))]).T
    return np.linalg.lstsq(X,y)[0]


def bytecode_auto2_numpy_lstsqr(x, y):
    return np.polyfit(x, y, 1)


def bytecode_auto_scipy_lstsqr(x,y):
    return scipy.stats.linregress(x, y)[0:2]


def bytecode_untyped_lstsqr(x, y):
    x_avg = sum(x)/len(x)
    y_avg = sum(y)/len(y)
    var_x = sum([(x_i - x_avg)**2 for x_i in x])
    cov_xy = sum([(x_i - x_avg)*(y_i - y_avg) for x_i,y_i in zip(x,y)])
    slope = cov_xy / var_x
    y_interc = y_avg - slope*x_avg
    return (slope, y_interc)