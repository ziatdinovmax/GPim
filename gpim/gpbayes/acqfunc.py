'''
acqfunc.py
======
Acquisition functions
'''

import numpy as np
from scipy.stats import norm


def confidence_bound(gpmodel, X_full, **kwargs):
    """
    Confidence bound acquisition function
    (a modification of upper confidence bound)

    Args:
        gpmodel (gpim reconstructor object):
            Surrogate function that allows computing
            mean and standard deviation
        X_full (ndarray):
            Full grid indices
        **alpha (float):
            :math:`\\alpha` coefficient in :math:`\\alpha \\mu + \\beta \\sigma`
        **beta (float):
            :math:`\\beta` coefficient in :math:`\\alpha \\mu + \\beta \\sigma`
    """
    if kwargs.get("alpha") is None:
        alpha = 0
    if kwargs.get("beta") is None:
        beta = 1
    mean, sd = gpmodel.predict(X_full)
    return alpha * mean + beta * sd


def expected_improvement(gpmodel, X_full, X_sparse, **kwargs):
    """
    Expected improvement acquisition function

    Args:
        gpmodel (gpim reconstructor object):
            Surrogate function that allows computing
            mean and standard deviation
        X_full (ndarray):
            Full grid indices
        X_sparse (ndarray):
            Sparse grid indices
        **xi (float):
            xi constant value
    """
    if kwargs.get("xi") is None:
        xi = 0.01
    mean, sd = gpmodel.predict(X_full)
    mean_sample, _ = gpmodel.predict(X_sparse)

    mean_sample_opt = np.amax(mean_sample)
    imp = mean - mean_sample_opt - xi
    z = imp / sd
    return imp * norm.cdf(z) + sd * norm.pdf(z)


def probability_of_improvement(gpmodel, X_full, X_sparse, **kwargs):
    """
    Probability of improvement acquisition function

    Args:
        gpmodel (gpim reconstructor object):
            Surrogate function that allows computing
            mean and standard deviation
        X_full (ndarray):
            Full grid indices
        X_sparse (ndarray):
            Sparse grid indices
        **xi (float):
            xi constant value
    """
    if kwargs.get("xi") is None:
        xi = 0.01
    mean, sd = gpmodel.predict(X_full)
    mean_sample = gpmodel.predict(X_sparse)

    mean_sample_opt = np.amax(mean_sample)
    z = mean - mean_sample_opt - xi
    z = z / sd
    return norm.cdf(z)
