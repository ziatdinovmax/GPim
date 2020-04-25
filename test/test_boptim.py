import os
import numpy as np
import pytest
from numpy.testing import assert_allclose
from gpim.gpbayes.boptim import boptimizer
from gpim import gprutils
np.random.seed(0)

test_img_ei = os.path.join(
    os.path.dirname(__file__), 'test_data/test_ei.npy')
test_img_poi = os.path.join(
    os.path.dirname(__file__), 'test_data/test_poi.npy')
test_img_cb = os.path.join(
    os.path.dirname(__file__), 'test_data/test_cb.npy')


def trial_func(idx, **kwargs):
    """Trial function, which takes a list of indices as input"""
    x0 = kwargs.get("x0", 5)
    y0 = kwargs.get("y0", 10)
    fwhm = kwargs.get("fwhm", 4.5)
    Z = np.exp(-4*np.log(2) * ((idx[0]-x0)**2 + (idx[1]-y0)**2) / fwhm**2)
    return Z


def initial_seed():
    """Creates sparse data (initial seed)"""
    np.random.seed(0)
    x = np.arange(0, 25, 1.)
    y = x[:, np.newaxis]
    Z = trial_func([y, x])
    idx = np.random.randint(0, Z.shape[0], size=(2, 5))
    Z_sparse = np.ones_like(Z) * np.nan
    Z_sparse[idx[0], idx[1]] = Z[idx[0], idx[1]]
    return Z_sparse


@pytest.mark.parametrize(
    "acqf, result",
    [("ei", test_img_ei),
     ("poi", test_img_poi),
     ("cb", test_img_cb)])
def test_boptim(acqf, result):
    Z_sparse = initial_seed()
    X_full = gprutils.get_full_grid(Z_sparse)
    X_sparse = gprutils.get_sparse_grid(Z_sparse)
    expected_result = np.load(result)
    boptim = boptimizer(
        X_sparse, Z_sparse, X_full,
        trial_func, acquisition_function=acqf,
        exploration_steps=20,
        use_gpu=False, verbose=1)
    boptim.run()
    assert_allclose(boptim.target_func_vals[-1], expected_result)
