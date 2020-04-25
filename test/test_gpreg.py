import numpy as np
import pytest
from numpy.testing import assert_
from gpim.gpreg import gpr, skgpr
from gpim import gprutils
np.random.seed(0)


def get_dummy_data():
    h = 5
    x_min, x_max = 0, 100
    y_min, y_max = 0, 100
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h))
    Z = np.exp(-((xx-25)**2+(yy-50)**2)/300)
    for _ in range(200):
        i = np.random.randint(Z.shape[0])
        j = np.random.randint(Z.shape[1])
        Z[i, j] = np.nan
    return Z


@pytest.mark.parametrize('kernel', ['RBF', 'Matern52'])
def test_gpr_2d(kernel):  # sanity check only, due to comput cost
    R = get_dummy_data()
    X = gprutils.get_sparse_grid(R)
    X_true = gprutils.get_full_grid(R)
    mean, sd, _ = gpr.reconstructor(
        X, R, X_true,
        kernel=kernel, learning_rate=0.1,
        iterations=2, use_gpu=False,
        verbose=False).run()
    assert_(mean.shape == sd.shape == R.shape)
    assert_(not np.isnan(mean).any())
    assert_(not np.isnan(sd).any())


@pytest.mark.parametrize('kernel', ['RBF', 'Matern52'])
def test_skgpr_2d(kernel):  # sanity check only, due to comput cost
    R = get_dummy_data()
    X = gprutils.get_sparse_grid(R)
    X_true = gprutils.get_full_grid(R)
    mean, sd, _ = skgpr.skreconstructor(
        X, R, X_true, kernel=kernel,
        learning_rate=0.1, iterations=2,
        use_gpu=False, verbose=False).run()
    assert_(mean.shape == sd.shape == R.shape)
    assert_(not np.isnan(mean).any())
    assert_(not np.isnan(sd).any())
