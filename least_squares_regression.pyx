import numpy as np
import scipy.stats

cimport cython
cimport openmp
from cython.parallel cimport prange

def matrix_lstsqr(x, y):
    X = np.vstack([x, np.ones(len(x))]).T
    return (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(y)


def auto_numpy_lstsqr(x, y):
    X = np.vstack([x, np.ones(len(x))]).T
    return np.linalg.lstsq(X,y)[0]


def auto2_numpy_lstsqr(x, y):
    return np.polyfit(x, y, 1)


def auto_scipy_lstsqr(x,y):
    return scipy.stats.linregress(x, y)[0:2]


def untyped_lstsqr(x, y):
    x_avg = sum(x)/len(x)
    y_avg = sum(y)/len(y)
    var_x = sum([(x_i - x_avg)**2 for x_i in x])
    cov_xy = sum([(x_i - x_avg)*(y_i - y_avg) for x_i,y_i in zip(x,y)])
    slope = cov_xy / var_x
    y_interc = y_avg - slope*x_avg
    return (slope, y_interc)

def simply_typed_lstsqr(x, y):
    cdef double x_avg, y_avg, var_x, cov_xy, slope, y_interc, x_i, y_i
    x_avg = sum(x)/x.shape[0]
    y_avg = sum(y)/y.shape[0]
    var_x = sum([(x_i - x_avg)**2 for x_i in x])
    cov_xy = sum([(x_i - x_avg)*(y_i - y_avg) for x_i,y_i in zip(x,y)])
    slope = cov_xy / var_x
    y_interc = y_avg - slope*x_avg
    return (slope, y_interc)

def memoryview_lstsqr(double[:] x, double[:] y):
    cdef double x_avg, y_avg, var_x, cov_xy, slope, y_interc, x_i, y_i
    x_avg = sum(x)/x.shape[0]
    y_avg = sum(y)/y.shape[0]
    var_x = sum([(x_i - x_avg)**2 for x_i in x])
    cov_xy = sum([(x_i - x_avg)*(y_i - y_avg) for x_i,y_i in zip(x,y)])
    slope = cov_xy / var_x
    y_interc = y_avg - slope*x_avg
    return (slope, y_interc)

@cython.boundscheck(False)
def fully_typed_lstsqr(double[:] x, double[:] y):
    cdef:
        Py_ssize_t idx
        double x_diff, y_diff
        double x_avg, y_avg
        double var_x=0, cov_xy=0
        double x_tot=0, y_tot=0

    assert x.shape[0] == y.shape[0]

    for idx in range(x.shape[0]):
        x_tot += x[idx]
        y_tot += y[idx]

    x_avg = x_tot / x.shape[0]
    y_avg = y_tot / y.shape[0]

    for idx in range(x.shape[0]):
        x_diff = x[idx] - x_avg
        y_diff = y[idx] - y_avg

        var_x += x_diff ** 2
        cov_xy += x_diff * y_diff

    slope = cov_xy / var_x
    y_interc = y_avg - slope*x_avg
    return slope, y_interc


@cython.boundscheck(False)
def parallel_lstsqr(double[:] x, double[:] y):
    cdef:
        Py_ssize_t idx
        double x_diff, y_diff
        double x_avg, y_avg
        double var_x=0, cov_xy=0
        double x_tot=0, y_tot=0

    assert x.shape[0] == y.shape[0]

    for idx in prange(x.shape[0], schedule="static", nogil=True):
        x_tot += x[idx]
        y_tot += y[idx]

    x_avg = x_tot / x.shape[0]
    y_avg = y_tot / y.shape[0]

    for idx in prange(x.shape[0], schedule="static", nogil=True):
        var_x += (x[idx] - x_avg) ** 2
        cov_xy += (x[idx] - x_avg) * (y[idx] - y_avg)

    slope = cov_xy / var_x
    y_interc = y_avg - slope*x_avg
    return slope, y_interc
