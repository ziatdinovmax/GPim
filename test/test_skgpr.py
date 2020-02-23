import os
import numpy as np
import pytest
from skimage.metrics import structural_similarity as ssim
from gpim import skgpr, gprutils

test_data = os.path.join(
    os.path.dirname(__file__), 'test_data/2D_testdata.npy')
test_expected_result = os.path.join(
    os.path.dirname(__file__), 'test_data/2D_reconst_skgpr.npy')


@pytest.mark.parametrize('kernel', ['RBF', 'Matern52'])
def test_skgpr_kernels(kernel):
    R = np.load(test_data)
    R_ = np.load(test_expected_result)
    X = gprutils.get_sparse_grid(R)
    X_true = gprutils.get_grid_indices(R)
    mean, _ = skgpr.skreconstructor(
        X, R, X_true, kernel=kernel,
        lengthscale=[[1., 1.], [4., 4.]],
        grid_points_ratio=1., learning_rate=0.1,
        iterations=20, calculate_sd=False, num_batches=1,
        use_gpu=False, verbose=False).run()
    assert ssim(mean, R_) > 0.98
    assert np.linalg.norm(mean - R_) < 1
