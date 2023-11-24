"""
gprutils.py
===========

Utility functions for the analysis of sparse image and hyperspectral data
with Gaussian processes.

Author: Maxim Ziatdinov (email: maxim.ziatdinov@ai4microcopy.com)
"""

import copy
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pyro
import torch
from mpl_toolkits.mplot3d import Axes3D
from torch.distributions import constraints, transform_to


def prepare_training_data(X, y=None, vector_valued=False, **kwargs):
    """
    Reshapes and converts data to torch tensors for GP analysis

    Args:
        X (ndarray):
            Grid indices with dimensions
            :math:`c \\times N \\times M \\times L`,
            where *c* is equal to the number of coordinates
            (for example, for xyz coordinates, *c* = 3)
        y (ndarray):
            Observations (data points) with dimensions N x M x L
        **precision (str):
            Choose between single ('single') and double ('double') precision

    Returns:
        Pytorch tensors with dimensions
        :math:`N \\times M \\times L \\times c`
        and :math:`N \\times M \\times L`
    """
    precision = kwargs.get("precision", "double")
    if precision == 'single':
        tensor_type = torch.FloatTensor
    else:
        tensor_type = torch.DoubleTensor
    tor = lambda n: torch.from_numpy(n).type(tensor_type)
    X = X.reshape(X.shape[0], np.product(X.shape[1:])).T
    X = tor(X[~np.isnan(X).any(axis=1)])
    if y is None:
        return X, y
    if vector_valued:
        y = y.reshape(np.product(y.shape[:-1]), y.shape[-1])
        y = tor(y[~np.isnan(y).any(axis=1)])
    else:
        y = tor(y.flatten()[~np.isnan(y.flatten())])

    return X, y


def prepare_test_data(X, **kwargs):
    """
    Reshapes and converts data to torch tensors for GP analysis

    Args:
        X (ndarray):
            Grid indices with dimensions :math:`c \\times N \\times M \\times L`
            where *c* is equal to the number of coordinates
            (for example, for xyz coordinates, *c* = 3)
        **precision (str):
            Choose between single ('single') and double ('double') precision

    Returns:
        Pytorch tensor with dimensions :math:`N \\times M \\times L \\times c`
    """
    precision = kwargs.get("precision", "double")
    if precision == 'single':
        tensor_type = torch.FloatTensor
    else:
        tensor_type = torch.DoubleTensor
    X = X.reshape(X.shape[0], np.product(X.shape[1:])).T
    X = torch.from_numpy(X).type(tensor_type)

    return X


def get_grid_indices(R, dense_x=1.):
    """
    Returns full and sparse grid indices for 2D and 3D arrays

    Args:
        R (ndarray):
            Sparse grid measurements as 2D or 3D numpy array
        dense_x (float):
            Determines grid density
            (can be increased at prediction stage)
    """
    if np.ndim(R) > 3:
        raise NotImplementedError(
            "Currently supports only 2D and 3D arrays")
    dense_x = np.float64(dense_x)
    X_full = get_full_grid(R, dense_x)
    X_sparse = get_sparse_grid(R)
    return X_full, X_sparse


def get_full_grid(R, extent=None, dense_x=1.):
    """
    Creates grid indices for 2D-4D numpy arrays

    Args:
        R (ndarray):
            Grid measurements as 2D-4D numpy array
        extent (list of lists):
            Define multi-dimensional data bounds. For example, for 2D data,
            the extent parameter is [[xmin, xmax], [ymin, ymax]]
        dense_x (float):
            Determines grid density
            (can be increased at prediction stage)

    Returns:
            Grid indices as numpy array
    """
    dense_x = np.float64(dense_x)
    if np.ndim(R) == 2:
        e1, e2 = R.shape
        if extent:
            dx = extent[0][1] - extent[0][0]
            dy = extent[1][1] - extent[1][0]
            dx = dense_x / (e1//dx)
            dy = dense_x / (e2//dy)
            c1, c2 = np.mgrid[
                extent[0][0]:extent[0][1]:dx, extent[1][0]:extent[1][1]:dy]
        else:
            c1, c2 = np.mgrid[:e1:dense_x, :e2:dense_x]
        X_grid = np.array([c1, c2])
    elif np.ndim(R) == 3:
        e1, e2, e3 = R.shape
        if extent:
            dx = extent[0][1] - extent[0][0]
            dy = extent[1][1] - extent[1][0]
            dz = extent[2][1] - extent[2][0]
            dx = dense_x / (e1//dx)
            dy = dense_x / (e2//dy)
            dz = dense_x / (e3//dz)
            c1, c2 = np.mgrid[
                extent[0][0]:extent[0][1]:dx, extent[1][0]:extent[1][1]:dy,
                extent[2][0]:extent[2][1]:dz]
        else:
            c1, c2, c3 = np.mgrid[:e1:dense_x, :e2:dense_x, :e3:dense_x]
        X_grid = np.array([c1, c2, c3])
    elif np.ndim(R) == 4:
        e1, e2, e3, e4 = R.shape
        if extent:
            dx = extent[0][1] - extent[0][0]
            dy = extent[1][1] - extent[1][0]
            dz = extent[2][1] - extent[2][0]
            df = extent[3][1] - extent[3][0]
            dx = dense_x / (e1//dx)
            dy = dense_x / (e2//dy)
            dz = dense_x / (e3//dz)
            df = dense_x / (e4//df)
            c1, c2 = np.mgrid[
                extent[0][0]:extent[0][1]:dx, extent[1][0]:extent[1][1]:dy,
                extent[2][0]:extent[2][1]:dz, extent[3][0]:extent[3][1]:df]
        else:
            c1, c2, c3, c4 = np.mgrid[:e1:dense_x, :e2:dense_x, :e3:dense_x, :e4:dense_x]
        X_grid = np.array([c1, c2, c3, c4])
    else:
        raise NotImplementedError("Currently works only for 2D-4D sets")
    return X_grid


def get_sparse_grid(R, extent=None):
    """
    Returns sparse grid for sparse image data

    Args:
        R (ndarray):
            Sparse grid measurements (missing values are NaNs)

    Returns:
        Sparse grid indices
    """
    if not np.isnan(R).any():
        raise NotImplementedError(
            "Missing values in sparse data must be represented as NaNs")
    X_true = get_full_grid(R, extent)
    if np.ndim(R) == 2:
        e1, e2 = R.shape
        X = X_true.copy().reshape(2, e1*e2)
        X[:, np.where(np.isnan((R.flatten())))] = np.nan
        X = X.reshape(2, e1, e2)
    elif np.ndim(R) == 3 and not np.isnan(R[..., -1]).any():
        e1, e2, e3 = R.shape
        X = X_true.copy().reshape(3, e1*e2, e3)
        indices = np.where(np.isnan((R.reshape(e1*e2, e3))))[0]
        X[:, indices] = np.nan
        X = X.reshape(3, e1, e2, e3)
    elif np.ndim(R) == 3 and np.isnan(R[..., -1]).any():
        e1, e2, e3 = R.shape
        X = X_true.copy().reshape(3, e1*e2*e3)
        indices = np.where(np.isnan((R.reshape(e1*e2*e3))))[0]
        X[:, indices] = np.nan
        X = X.reshape(3, e1, e2, e3)
    else:
        raise NotImplementedError(
            "Currently supports only 2D and 3D sets with sparsity in xy and xyz dims")
    return X


def to_constrained_interval(state_dict, lscale, amp):
    """
    Transforms kernel's unconstrained lenghscale and variance
    to their constrained domains (intervals)

    Args:
        state_dict (dict):
            Kernel's state dictionary;
            can be obtained from self.spgr.kernel.state_dict
        lscale (list):
            List of two lists with lower and upper bound(s)
            for lenghtscale prior. Number of elements in each list
            is usually equal to the number of (independent) input dimensions
        amp (list):
            List with two floats corresponding to lower and upper
            bounds for variance (square of amplitude) prior

    Returns:
        Lengthscale and variance in the constrained domain (interval)
    """
    l_ = state_dict()['lenghtscale_map_unconstrained']
    a_ = state_dict()['variance_map_unconstrained']
    l_interval = constraints.interval(
        torch.tensor(lscale[0]), torch.tensor(lscale[1]))
    a_interval = constraints.interval(
        torch.tensor(amp[0]), torch.tensor(amp[1]))
    l = transform_to(l_interval)(l_)
    a = transform_to(a_interval)(a_)
    return l, a


def corrupt_data_xy(X_true, R_true, prob=0.5, replace_w_zeros=False):
    """
    Replaces certain % of 2D or 3D image data with NaNs;
    see gprutils.corrupt_image2d and gprutils.corrupt_image3d

    Args:
        X_true (ndarray):
            Grid indices for 2D image or 3D hyperspectral data
            (3D and 4D numpy arrays, respectively)
        R_true (ndarray):
            Observations as 2D image or 3D hyperspectral data
        prob (float):
            Controls % of data to be corrupted
            (takes values between 0 and 1)
        replace_w_zeros (bool):
            Corrupts data with zeros instead of NaNs

    Returns:
        ndarays of grid indices (3D or 4D) and observations (2D or 3D)
    """
    if np.ndim(R_true) == 2:
        X, R = corrupt_image2d(X_true, R_true, prob, replace_w_zeros)
    elif np.ndim(R_true) == 3:
        X, R = corrupt_image3d(X_true, R_true, prob, replace_w_zeros)
    else:
        raise NotImplementedError("Currently supports only 2D and 3D sets")
    return X, R


def corrupt_image2d(X_true, R_true, prob, replace_w_zeros):
    """
    Replaces certain % of 2D image data with NaNs.

    Args:
        X_true (ndarray):
            3D array with grid indices for 2D image
        R_true (ndarray):
            2D image with observations
        prob (float):
            Controls % of data to be corrupted
            (takes values between 0 and 1)
        replace_w_zeros (bool):
            Corrupts data with zeros instead of NaNs


    Returns:
        3D ndarray of grid coordinates and 2D ndarray of observatons
        where the part of points is replaced with NaNs.
    """
    e1, e2 = R_true.shape
    if np.isnan(R_true).any():
        X = X_true.copy().reshape(2, e1*e2)
        X[:, np.where(np.isnan((R_true.flatten())))] = np.nan
        X = X.reshape(2, e1, e2)
        return X, R_true
    pyro.set_rng_seed(0)
    brn = pyro.distributions.Bernoulli(prob)
    indices = [i for i in range(e1*e2) if brn.sample() == 1]
    R = R_true.copy().reshape(e1*e2)
    R[indices] = np.nan
    R = R.reshape(e1, e2)
    X = X_true.copy().reshape(2, e1*e2)
    X[:, indices] = np.nan
    X = X.reshape(2, e1, e2)
    if replace_w_zeros:
        X = np.nan_to_num(X)
        R = np.nan_to_num(R)
    return X, R


def corrupt_image3d(X_true, R_true, prob, replace_w_zeros):
    """
    Replaces certain % of 3D hyperspectral data with NaNs.
    Applies differently in xy and in z dimensions.
    Specifically, for every corrupted (x, y) point
    we remove all z values associated with this point.

    Args:
        X_true (ndarray):
            4D array with grid indices for 3D hyperspectral data
        R_true (ndarray):
            3D hyperspectral data with observations
        prob (float):
            Controls % of data to be corrupted
            (takes values between 0 and 1)
        replace_w_zeros (bool):
            Corrupts data with zeros instead of NaNs


    Returns:
        4D ndarray of grid coordinates and
        3D ndarray of observatons where
        certain % of points is replaced with NaNs
        (note that for every corrupted (x, y) point
        we remove all z values associated with this point)
    """
    e1, e2, e3 = R_true.shape
    if np.isnan(R_true).any():
        X = X_true.copy().reshape(3, e1*e2, e3)
        indices = np.where(np.isnan((R_true.reshape(e1*e2, e3))))[0]
        X[:, indices] = np.nan
        X = X.reshape(3, e1, e2, e3)
        return X, R_true
    pyro.set_rng_seed(0)
    brn = pyro.distributions.Bernoulli(prob)
    indices = [i for i in range(e1*e2) if brn.sample() == 1]
    R = R_true.copy().reshape(e1*e2, e3)
    R[indices, :] = np.nan
    R = R.reshape(e1, e2, e3)
    X = X_true.copy().reshape(3, e1*e2, e3)
    X[:, indices, :] = np.nan
    X = X.reshape(3, e1, e2, e3)
    if replace_w_zeros:
        X = np.nan_to_num(X)
        R = np.nan_to_num(R)
    return X, R


def open_edge_points(R, R_true, s=6):
    """
    Opens measured curves at the edges of FOV

    Args:
        R (ndarray):
            empty/sparse data
        R_true (ndarray):
            "ground truth"
        s (int):
            step value, which determines the density of opened edge points

    Returns:
        3D ndarray with opened edge points
    """
    e1, e2 = R_true.shape[:2]
    R[0, ::s] = R_true[0, ::s]
    R[::s, 0] = R_true[::s, 0]
    R[e1-1, s:e2-s:s] = R_true[e1-1, s:e2-s:s]
    R[s::s, e2-1] = R_true[s::s, e2-1]
    return R


def plot_kernel_hyperparams(hyperparams):
    """
    Plots evolution of kernel hyperparameters
    as a function of training steps

    Args:
        hyperparams (dict):
            dictionary with kernel hyperparameters
            (see gpreg.gpr.reconstructor)
    """
    if "weights" in hyperparams.keys():
        plot_mixture_hyperparams(hyperparams)
        return
    if 'variance' in hyperparams.keys():
        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
    else:
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    l = ax1.plot(hyperparams['lengthscale'], linewidth=3)
    ax1.set_title('lengthscale')
    ax1.set_xlabel('SVI iteration')
    ax1.set_ylabel('lengthscale (px)')
    ax1.legend(l, ('dim 1', 'dim 2', 'dim 3'))
    ax2.plot(hyperparams['noise'], linewidth=3)
    ax2.set_yscale('log')
    ax2.set_title('noise')
    ax2.set_xlabel('SVI iteration')
    ax2.set_ylabel('noise (px)')
    plt.subplots_adjust(wspace=.5)
    if 'variance' in hyperparams.keys():
        ax3.plot(hyperparams['variance'], linewidth=3)
        ax3.set_yscale('log')
        ax3.set_title('variance')
        ax3.set_xlabel('SVI iteration')
        ax3.set_ylabel('variance (px)')
    plt.show()


def plot_mixture_hyperparams(hyperparams):
    """
    Plots evolution of spectral mixture kernel hyperparameters
    as a function of training iterations

    Args:
        hyperparams (dict):
            dictionary with kernel hyperparameters
            (see gpreg.skgpr.skreconstructor)
    """
    means = hyperparams["means"]
    scales = hyperparams["scales"]
    weights = hyperparams["weights"]
    noise = hyperparams["noise"]
    maxdim = hyperparams["maxdim"]

    if scales[0].shape[-1] != 2:
        raise NotImplementedError(
            "Currently supports plotting only for 2D cases"
        )

    print("Mixture (final) weights:")
    for i, w in enumerate(weights[-1]):
        print("Component {}: w = {}".format(i, w.astype(np.float64).round(5)))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))
    for i, m in enumerate(means):
        label1 = "x coordinate" if i == len(means) - 1 else None
        label2 = "y coordinate" if i == len(means) - 1 else None
        ax1.scatter(np.tile(i, len(m)), m[:, 0, 0], s=18, c=np.arange(len(m)), cmap='jet', label=label1)
        ax1.scatter(np.tile(i, len(m)), m[:, 0, 1], s=18, marker='x', c=np.arange(len(m)), cmap='jet', label=label2)
    ax1.set_xlabel("Iteration", fontsize=14)
    ax1.set_ylabel("Mixture mean/period (px)", fontsize=14)
    ax1.set_title("Mixtures mean (period)", fontsize=14)
    ax1.legend()

    for i, s in enumerate(scales):
        label1 = "x coordinate" if i == len(scales) - 1 else None
        label2 = "y coordinate" if i == len(scales) - 1 else None
        ax2.scatter(np.tile(i, len(s)), s[:, 0, 0], s=18, c=np.arange(len(s)), cmap='jet', label=label1)
        ax2.scatter(np.tile(i, len(s)), s[:, 0, 1], s=18, marker='x', c=np.arange(len(s)), cmap='jet', label=label2)
    ax2.set_xlabel("Iteration", fontsize=14)
    ax2.set_ylabel("Mixture scale (px)", fontsize=14)
    ax2.set_title("Mixtures scales", fontsize=14)
    ax2.legend()

    ax3.plot(noise, linewidth=3)
    ax3.set_ylabel("noise (px)", fontsize=14)
    ax3.set_xlabel("Iteration", fontsize=14)
    ax3.set_title("noise", fontsize=14)

    ax1.set_ylim(0, maxdim)
    ax2.set_ylim(0, maxdim)

    clrbar = np.linspace(1, len(m)).reshape(-1, 1)
    ax_ = fig.add_axes([.36, -.12, .3, .8])
    img = plt.imshow(clrbar, cmap='jet')
    plt.gca().set_visible(False)
    clrbar = plt.colorbar(img, ax=ax_, orientation='horizontal')
    clrbar.set_label('Mixture component', fontsize=14, labelpad=10)
    plt.show()


def plot_raw_data(raw_data, slice_number, pos,
                  spec_window=2, norm=False, **kwargs):
    """
    Plots hyperspectral data as 2D image
    integrated over a certain range of energy/frequency
    and selected individual spectroscopic curves

    Args:
        raw_data (3D ndarray):
            hyperspectral cube (the first two dimensions are *xy* coordinates
            and the last dimension is a "spectroscopic" dimension)
        slice_number (int):
            slice from datacube to visualize
        pos (list of lists):
            list with [x, y] coordinates of points where
            single spectroscopic curves will be extracted and visualized
        spec_window (int):
            window to integrate over in frequency dimension (for 2D "slices")
        **cmap (str):
            cmap for 2D image ("slice") plot
        **z_vec (1D ndarray):
            spectroscopic measurements values (e.g. frequency, bias)
        **z_vec_label (str):
            spectroscopic measurements label (e.g. frequency, bias voltage)
        **z_vec_units (str):
            spectroscopic measurements units (e.g. Hz, V)
    """
    cmap = kwargs.get('cmap', 'magma')
    z_vec = kwargs.get('z_vec')
    z_vec_label = kwargs.get('z_vec_label')
    z_vec_units = kwargs.get('z_vec_units')
    z_vec = np.arange(raw_data.shape[-1]) if z_vec is None else z_vec
    # colors sequence
    my_colors = ['black', 'red', 'green', 'gray', 'orange', 'blue']
    # Plotting
    s = slice_number
    spw = spec_window
    _, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    ax[0].imshow(np.sum(raw_data[:, :, s-spw:s+spw], axis=-1), cmap=cmap)
    for p, col in zip(pos, my_colors):
        ax[0].scatter(p[1], p[0], c=col)
        ax[1].plot(z_vec, raw_data[p[0], p[1], :], c=col)
    ax[1].axvspan(z_vec[s-spw], z_vec[s+spw], linestyle='--', alpha=.2)
    if norm:
        ax[1].set_ylim(-0.1, 1.1)
    if z_vec_label is not None and z_vec_units is not None:
        ax[1].set_xlabel(z_vec_label+', '+z_vec_units)
        ax[1].set_ylabel('Response (arb. units)')
    ax[0].set_title('Grid spectroscopy data')
    ax[1].set_title('Individual spectroscopic curves')
    plt.subplots_adjust(wspace=.3)
    plt.show()


def plot_reconstructed_data2d(R, mean, save_fig=False, **kwargs):
    """
    Plots original and GP-reconstructed data for 2D images

    Args:
        R (2D ndarray):
            Input image for GP regression
        mean (1D ndarray):
            Predictive mean, usually an output of gpr.reconstructor or
            skgpr.skreconstructor. The array is flattened
            (the actual dimensions are the same as for R)
        **cmap (str):
            cmap for 2D image plot
        **savedir (str):
            directory to save output figure
        **filepath (str):
            name of input file (to create a unique filename for plot)
        **sparsity (float)
            indicates % of data points removed (used only for figure title)
    """

    if save_fig:
        mdir = kwargs.get('savedir', 'Output')
        if not os.path.exists(mdir):
            os.makedirs(mdir)
        fpath = kwargs.get('filepath')
    sparsity = kwargs.get('sparsity')
    cmap = kwargs.get('cmap', 'nipy_spectral')
    e1, e2 = R.shape
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
    ax1.imshow(R, cmap=cmap, origin='bottom')
    ax2.imshow(mean.reshape(e1, e2), cmap=cmap, origin='bottom')
    ax1.set_title('Input/corrupted data')
    if sparsity:
        ax2.set_title(
            'Corrupted input data\n{}% of observations removed'.format(sparsity*100))
    else:
        ax2.set_title('Input data')
    ax2.set_title('GP reconstruction')
    if save_fig:
        if fpath:
            fig.savefig(os.path.join(mdir, os.path.basename(
                os.path.splitext(fpath)[0])))
        else:
            fig.savefig(os.path.join(mdir, 'reconstruction'))
    plt.show()


def plot_reconstructed_data3d(R, mean, sd, slice_number, pos,
                              spec_window=2, save_fig=False,
                              **kwargs):
    """
    Plots original and GP-reconstructed data for 3D images

    Args:
        R (3D ndarray):
            Input image for GP regression
        mean (1D ndarray):
            Predictive mean, usually an output of gpr.reconstructor or
            skgpr.skreconstructor. The array is flattened
            (the actual dimensions are the same as for R)
        sd (1D ndarray):
            Standard deviation
            (can be flattened; actual dimensions are the same as in R)
        slice_number (int):
            slice from datacube to visualize
        pos (list of lists):
            list with [x, y] coordinates of points where
            single spectroscopic curves will be extracted and visualized
        spec_window (int):
            window to integrate over in frequency dimension (for 2D "slices")
        **cmap (str):
            colormap for 2D image ("slices") plots
        **savedir (str):
            directory to save output figure
        **sparsity (float):
            indicates % of data points removed (used only for figure title)
        **filepath (str):
            path/name of input file (to create a unique filename for plot)
        **z_vec (1D ndarray):
            spectroscopic measurements values (e.g. frequency, bias)
        **z_vec_label (str):
            spectroscopic measurements label (e.g. frequency, bias voltage)
        **z_vec_units (str):
            spectroscopic measurements units (e.g. Hz, V)
    """
    if save_fig:
        mdir = kwargs.get('savedir')
        if mdir is None:
            mdir = 'Output'
        if not os.path.exists(mdir):
            os.makedirs(mdir)
        fpath = kwargs.get('filepath')
    sparsity = kwargs.get('sparsity')
    cmap = kwargs.get('cmap', 'nipy_spectral')
    z_vec = kwargs.get('z_vec')
    z_vec_label = kwargs.get('z_vec_label')
    z_vec_units = kwargs.get('z_vec_units')
    z_vec = np.arange(R.shape[-1]) if z_vec is None else z_vec
    s = slice_number
    e1, e2, e3 = R.shape
    spw = spec_window
    Rtest = mean.reshape(e1, e2, e3)
    R_sd = sd.reshape(e1, e2, e3)
    my_colors = ['black', 'red', 'green', 'gray', 'orange', 'blue']
    fig, ax = plt.subplots(2, 2, figsize=(14, 14))
    ax[0, 0].imshow(
        np.sum(R[:, :, s-spw:s+spw], axis=-1), cmap=cmap)
    for p, col in zip(pos, my_colors):
        ax[0, 0].scatter(p[1], p[0], c=col)
        ax[0, 1].plot(z_vec, R[p[0], p[1], :], c=col)
    ax[0, 1].axvspan(z_vec[s-spw], z_vec[s+spw], linestyle='--', alpha=.15)
    ax[0, 1].set_ylim(-0.1, 1.1)
    if z_vec_label is not None and z_vec_units is not None:
        ax[0, 1].set_xlabel(z_vec_label+', '+z_vec_units)
        ax[0, 1].set_ylabel('Response (arb. units)')
    for _ax in [ax[0, 0], ax[0, 1]]:
        if sparsity:
            _ax.set_title(
                'Corrupted input data\n{}% of observations removed'.format(sparsity*100))
        else:
            _ax.set_title('Input data')
    ax[1, 0].imshow(
        np.sum(Rtest[:, :, s-spw:s+spw], axis=-1), cmap=cmap)
    for p, col in zip(pos, my_colors):
        ax[1, 0].scatter(p[1], p[0], c=col)
        ax[1, 1].plot(z_vec, Rtest[p[0], p[1], :], c=col)
        ax[1, 1].fill_between(z_vec,
                        (Rtest[p[0], p[1], :] -
                         2.0 * R_sd[p[0], p[1], :]),
                        (Rtest[p[0], p[1], :]
                         + 2.0 * R_sd[p[0], p[1], :]),
                        color=col, alpha=0.15)
    ax[1, 1].axvspan(z_vec[s-spw], z_vec[s+spw], linestyle='--', alpha=.15)
    ax[1, 1].set_ylim(-0.1, 1.1)
    if z_vec_label is not None and z_vec_units is not None:
        ax[1, 1].set_xlabel(z_vec_label+', '+z_vec_units)
        ax[1, 1].set_ylabel('Response (arb. units)')
    for _ax in [ax[1, 0], ax[1, 1]]:
        _ax.set_title('GPR reconstruction')
    plt.subplots_adjust(hspace=.3)
    if save_fig:
        if fpath:
            fig.savefig(os.path.join(mdir, os.path.basename(
                os.path.splitext(fpath)[0])))
        else:
            fig.savefig(os.path.join(mdir, 'reconstruction'))
    plt.show()


def plot_exploration_results(R_all, mean_all, sd_all, R_true,
                             episodes, slice_number, pos, dist_edge,
                             spec_window=2, mask_predictions=False,
                             **kwargs):
    """
    Plots predictions at different stages ("episodes")
    of maximum uncertainty-based sample exploration with GP

    Args:
        R_all (list with ndarrays):
            Observed data points at each exploration step
        mean_all (list of ndarrays):
            Predictive mean at each exploration step
        sd_all (list of ndarrays):
            Integrated (along energy dimension) SD at each exploration step
        R_true (ndarray):
            3D array with ground truth data (full observations) for simulated
            experiment OR a 3D array of zeros/NaNs for real experiment
        episodes (list of ints):
            list with the numbers indicating which iteration steps to visualize
        slice_number (int):
            slice from datacube to visualize
        pos (list of lists):
            list with [x, y] coordinates of points where
            single spectroscopic curves will be extracted and visualized
        dist_edge (list with two integers):
            this should be the same as in exploration analysis
        spec_win (int):
            window to integrate over in frequency dimension (for 2D "slices")
        mask_predictions (bool):
            mask edge regions not used in max uncertainty evaluation
            in predictive mean plots
        **sparsity (float):
            indicates % of data points removed (used only for figure title)
        **z_vec (1D ndarray):
            spectroscopic measurements values (e.g. frequency, bias)
        **z_vec_label (str):
            spectroscopic measurements label (e.g. frequency, bias voltage)
        **z_vec_units (str):
            spectroscopic measurements units (e.g. Hz, V)
    """

    s = slice_number
    spw = spec_window
    e1, e2, e3 = R_true.shape
    z_vec = kwargs.get('z_vec')
    z_vec_label = kwargs.get('z_vec_label')
    z_vec_units = kwargs.get('z_vec_units')
    z_vec = np.arange(e3) if z_vec is None else z_vec
    _colors = ['black', 'red', 'green', 'blue', 'orange']
    # plot ground truth data if available
    if not np.isnan(R_true).any() or np.unique(R_true).any():
        _, ax = plt.subplots(1, 2, figsize=(7, 3), dpi=100)
        ax[0].imshow(np.sum(R_true[:, :, s-spw:s+spw], axis=-1), cmap='jet')
        for p, col in zip(pos, _colors):
            ax[0].scatter(p[1], p[0], c=col)
            ax[1].plot(z_vec, R_true[p[0], p[1], :], c=col)
        ax[1].axvspan(z_vec[s-spw], z_vec[s+spw], linestyle='--', alpha=.2)
        ax[1].set_ylim(-0.1, 1.1)
        if z_vec_label is not None and z_vec_units is not None:
            ax[1].set_xlabel(z_vec_label+', '+z_vec_units)
            ax[1].set_ylabel('Response (arb. units)')
        ax[0].set_title('Grid spectroscopy\n(ground truth)')
        ax[1].set_title('Individual spectroscopic curves\n(ground truth)')

    # Plot predictions
    n = len(episodes) + 1
    fig = plt.figure(figsize=(20, 17), dpi=100)

    for i in range(1, n):
        Rcurr = R_all[episodes[i-1]].reshape(e1, e2, e3)
        Rtest = mean_all[episodes[i-1]].reshape(e1, e2, e3)
        R_sd = sd_all[episodes[i-1]].reshape(e1, e2, e3)

        ax = fig.add_subplot(4, n, i)
        ax.imshow(np.sum(Rcurr[:, :, s-spw:s+spw], axis=-1), cmap='jet')
        ax.set_title('Observations (step {})'.format(episodes[i-1]))

        ax = fig.add_subplot(4, n, i + n)
        Rtest_to_plot = copy.deepcopy((np.sum(Rtest[:, :, s-spw:s+spw], axis=-1)))
        mask = np.zeros(Rtest_to_plot.shape, bool)
        mask[dist_edge[0]:e1-dist_edge[0],
             dist_edge[1]:e2-dist_edge[1]] = True
        if mask_predictions:
            Rtest_to_plot[~mask] = np.nan
        ax.imshow(Rtest_to_plot, cmap='jet')
        for p, col in zip(pos, _colors):
            ax.scatter(p[1], p[0], c=col)
        ax.set_title('GPR reconstruction (step {})'.format(episodes[i-1]))
        ax = fig.add_subplot(4, n, i + 2*n)
        for p, col in zip(pos, _colors):
            ax.plot(z_vec, Rtest[p[0], p[1], :], c=col)
            ax.fill_between(z_vec,
                            (Rtest[p[0], p[1], :] - 2.0 *
                            R_sd[p[0], p[1], :]),
                            (Rtest[p[0], p[1], :] + 2.0 *
                            R_sd[p[0], p[1], :]),
                            color=col, alpha=0.15)
            ax.axvspan(z_vec[s-spw], z_vec[s+spw], linestyle='--', alpha=.15)
        ax.set_ylim(-0.1, 1.1)
        if z_vec_label is not None and z_vec_units is not None:
            ax.set_xlabel(z_vec_label+', '+z_vec_units)
            ax.set_ylabel('Response (arb. units)')
        ax.set_title('GPR reconstruction (step {})'.format(episodes[i-1]))

        ax = fig.add_subplot(4, n, i + 3*n)
        R_sd_to_plot = copy.deepcopy(R_sd)
        R_sd_to_plot = np.sum(R_sd_to_plot, axis=-1)
        R_sd_to_plot[~mask] = np.nan
        ax.imshow(R_sd_to_plot, cmap='jet')
        ax.set_title('Integrated uncertainty (step {})'.format(episodes[i-1]))

    plt.subplots_adjust(hspace=.4)
    plt.subplots_adjust(wspace=.3)
    plt.show()


def plot_inducing_points(hyperparams, **kwargs):
    """
    Plots inducing points evolution during training
    """
    dims_ = hyperparams['inducing_points'][0].shape[-1]
    if dims_ == 2:
        plot_inducing_points_2d(hyperparams, **kwargs)
    elif dims_ == 3:
        plot_inducing_points_3d(hyperparams, **kwargs)
    else:
        raise NotImplementedError('Supports only 2D and 3D datasets')


def plot_inducing_points_2d(hyperparams, **kwargs):
    """
    Plots 2D trajectories if inducing points

    Args:
        hyperparams (dict):
            Dictionary of hyperparameters
        **plot_from (int):
            plot from specific step
        **plot_to (int):
            plot till specific step
        **slice_step (int):
            plot every nth inducing point
    """
    learned_inducing_points = hyperparams['inducing_points']
    indp_nth = kwargs.get('slice_step')
    plot_from, plot_to = kwargs.get('plot_to'), kwargs.get('plot_from')
    if plot_from is None:
        plot_from = 0
    if plot_to is None:
        plot_to = len(learned_inducing_points)
    if indp_nth is None:
        indp_nth = 1
    fig = plt.figure(figsize=(20, 9))
    ax = fig.add_subplot(121)
    ax.set_xlabel('x coordinate (px)', fontsize=14)
    ax.set_ylabel('y coordinate (px)', fontsize=14)
    ax.set_title('Evolution of inducing points', fontsize=16)
    ax.set_aspect('auto')# 'equal' doesn't work in matplotlib 3.1.1
    colors = plt.cm.jet(
        np.linspace(0, 1,len(learned_inducing_points[plot_from:plot_to]))
    )
    for xy, c in zip(learned_inducing_points[plot_from:plot_to], colors):
        y, x = xy.T
        ax.scatter(x[::indp_nth], y[::indp_nth], c=[c], s=.15)
    clrbar = np.linspace(
        0, len(learned_inducing_points[plot_from:plot_to])).reshape(-1, 1)
    ax2 = fig.add_axes([.42, .1, .1, .8])
    img = plt.imshow(clrbar, cmap="jet")
    plt.gca().set_visible(False)
    clrbar_ = plt.colorbar(img, ax=ax2, orientation='vertical')
    clrbar_.set_label('SVI iterations', fontsize=14, labelpad=10)
    plt.show()


def plot_inducing_points_3d(hyperparams, **kwargs):
    """
    Plots 3D trajectories if inducing points during model training

    Args:
        hyperparams (dict):
            dictionary of hyperparameters
        plot_from (int):
            plot from specific step
        plot_to (int):
            plot till specific step
        slice_step (int):
            plot every nth inducing point
    """
    learned_inducing_points = hyperparams['inducing_points']
    indp_nth = kwargs.get('slice_step')
    plot_from, plot_to = kwargs.get('plot_to'), kwargs.get('plot_from')
    if plot_from is None:
        plot_from = 0
    if plot_to is None:
        plot_to = len(learned_inducing_points)
    if indp_nth is None:
        indp_nth = 1
    fig = plt.figure(figsize=(22, 9))
    ax = fig.add_subplot(121, projection='3d')
    ax.view_init(20, 30)
    ax.set_xlabel('x coordinate (px)', fontsize=14)
    ax.set_ylabel('y coordinate (px)', fontsize=14)
    ax.set_zlabel('frequency (px)', fontsize=14)
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    ax.zaxis.labelpad = 5
    ax.set_title('Evolution of inducing points', fontsize=16)
    ax.dist = 10 # use it to zoom in/out
    ax.set_aspect('auto')# 'equal' doesn't work in matplotlib 3.1.1
    colors = plt.cm.jet(
        np.linspace(0,1,len(learned_inducing_points[plot_from:plot_to]))
    )
    for xyz, c in zip(learned_inducing_points[plot_from:plot_to], colors):
        x, y, z = xyz.T
        ax.scatter(x[::indp_nth], y[::indp_nth], z[::indp_nth], c=[c], s=.15)
    clrbar = np.linspace(
        0, len(learned_inducing_points[plot_from:plot_to])).reshape(-1, 1)
    ax2 = fig.add_axes([.37, .1, .1, .8])
    img = plt.imshow(clrbar, cmap="jet")
    plt.gca().set_visible(False)
    clrbar_ = plt.colorbar(img, ax=ax2, orientation='vertical')
    clrbar_.set_label('SVI iterations', fontsize=14, labelpad=10)
    plt.show()


def plot_query_points(inds_all, **kwargs):
    """
    Plots the exploration path (all the query points)
    in GP-based Bayesian optimization. Currently supports only 2D data.

    Args:
        inds_all (list): list of indices
        **cmap (str): colormap
    """
    cmap = kwargs.get("cmap", "cool")
    plot_lines = kwargs.get("plot_lines", False)
    inds_all = np.array(inds_all)  # transform list to ndarray for plotting
    cvals = np.arange(len(inds_all))
    clrbar = np.linspace(0, len(inds_all)).reshape(-1, 1)
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    ax1.scatter(inds_all[:, 1], inds_all[:, 0], c=cvals, cmap=cmap)
    if plot_lines:
        ax1.plot(inds_all[:, 1], inds_all[:, 0])
    ax2 = fig.add_axes([.78, .1, .2, .8])
    img = plt.imshow(clrbar, cmap)
    plt.gca().set_visible(False)
    clrbar_ = plt.colorbar(img, ax=ax2, orientation='vertical')
    clrbar_.set_label('Exploration steps', fontsize=14, labelpad=10)
    plt.show()
