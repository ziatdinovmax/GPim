import os
import numpy as np
import pytest
from skimage.metrics import structural_similarity as ssim
from gpim import gpr, gprutils

test_data2d = os.path.join(
    os.path.dirname(__file__), 'test_data/2D_testdata.npy')
test2d_expected_result = os.path.join(
    os.path.dirname(__file__), 'test_data/2D_reconst_gpr.npy')
test_data3d = os.path.join(
    os.path.dirname(__file__), 'test_data/bepfm_test_data_sparse.npy')


@pytest.mark.parametrize('kernel', ['RBF', 'Matern52'])
def test_gpr_2d(kernel):
    R = np.load(test_data2d)
    R_ = np.load(test2d_expected_result)
    X = gprutils.get_sparse_grid(R)
    X_true = gprutils.get_grid_indices(R)
    mean, _, _ = gpr.reconstructor(
        X, R, X_true,
        kernel=kernel, lengthscale=[[1., 1.], [4., 4.]],
        indpoints=250, learning_rate=0.1, iterations=200,
        use_gpu=False, verbose=False).run()
    assert ssim(mean, R_) > 0.95
    assert np.linalg.norm(mean - R_) < 3

@pytest.mark.parametrize('kernel', ['RBF', 'Matern52'])
def test_gpr_3d(kernel): # sanity check only due to comput cost
    R = np.load(test_data3d)
    X = gprutils.get_sparse_grid(R)
    X_true = gprutils.get_grid_indices(R)
    mean, sd, hyperparam = gpr.reconstructor(
        X, R, X_true,
        kernel=kernel, lengthscale=None,
        indpoints=50, learning_rate=0.1,
        iterations=2, use_gpu=False,
        verbose=True).run()
    assert mean.shape == sd.shape == R.flatten().shape
    assert not np.isnan(mean).any()
    assert not np.isnan(sd).any()
