"""
boptim.py
===========

Utility functions for the Gaussian process-based
Bayesian optimization for selecting the next query points in
images and image-like data.

Author: Maxim Ziatdinov (email: maxim.ziatdinov@ai4microcopy.com)
"""

import torch
import numpy as np
from scipy import spatial
from gpim.gpreg import gpr
from gpim.gpbayes import acqfunc
from gpim import gprutils


class boptimizer:
    """
    Gaussian process-based Bayesian optimization for selecting
    next point(s) to measure/evaluate. The Bayesian optimization strategy
    consists of: i) defining prior and posterior distributions over
    the objective (target) function :math:`f` using GP; ii) using the posterior
    to derive an acquistion function :math:`\\alpha (x)`; iii) using the
    acquisition function to derive the next query point according to
    :math:`x_{next}=argmin(\\alpha (x))`; iv) evaluating :math:`f` in
    :math:`x_{next}` and updating the posterior.

    Args:
        X_seed (ndarray):
            Seed sparse grid indices with dimensions :math:`c \\times N \\times M`
            or :math:`c \\times N \\times M \\times L`
            where *c* is equal to the number of coordinates
            (for example, for *xyz* coordinates, *c* = 3)
        y_seed (ndarray):
            Seed sparse "observations" (data points) with dimensions
            :math:`N \\times M` or :math:`N \\times M \\times L`.
            Typically, for 2D image *N* and *M* are image height and width,
            whereas for 3D hyperspectral data *N* and *M* are spatial dimensions
            and *L* is a spectorcopic dimension (e.g. voltage or wavelength).
        X_full (ndarray):
            Full grid indices (for prediction with a trained GP model)
            with dimensions :math:`N \\times M` or :math:`N \\times M \\times L`
        target_function (python function):
            Target (or objective) function. Takes a list of indices and
            returns a function value as a float.
        acquisition_function (str):
            Acquisition function choise
            ('cb' is confidence bound, 'ei' is expected improvement,
            'poi' is a probability of improvement)
        exploration_steps (int):
            Number of exploration-exploitation steps
            (the expltation-exploration trade-off is
            determined by the acquisition function)
        batch_size (int):
                Number of query points in one batch.
                Returns a single next query point.
        batch_update:
            Filters the query points based on the kernel lengthscale.
            Returns a batch of points when set to True.
            The number of points in the batch may be different from
            batch_size as points are filtered based on the lengthscale.
        kernel (str):
            Kernel type ('RBF', 'Matern52', 'RationalQuadratic')
        lengthscale (list of int or list of two lists with int):
            Determines lower (1st value or 1st list) and upper (2nd value or 2nd list)
            bounds for kernel lengthscales. For list with two integers,
            the kernel will have only one lenghtscale, even if the dataset
            is multi-dimensional. For lists of two lists, the number of elements
            in each list must be equal to the dataset dimensionality.
        sparse (bool):
            Uses sparse GP regression when set to True.
        indpoints (int):
            Number of inducing points for SparseGPRegression.
            Defaults to total_number_of_points // 10.
        learning_rate (float):
            Learning rate for GP model training
        iterations (int): Number of SVI training iteratons for GP model
        seed (int):
            for reproducibility
        **use_gpu (bool):
            Uses GPU hardware accelerator when set to 'True'.
            Notice that for large datasets training model without GPU
            is extremely slow.
        **verbose (int):
            Level of verbosity (0, 1, or 2)
        **lscale (float):
            Lengthscale determining the separation (euclidean)
            distance between query points. Defaults to the kernel
            lengthscale at a given step
    """
    def __init__(self,
                 X_seed,
                 y_seed,
                 X_full,
                 target_function,
                 acquisition_function='cb',
                 exploration_steps=10,
                 batch_size=100,
                 batch_update=False,
                 kernel='RBF',
                 lengthscale=None,
                 sparse=False,
                 indpoints=None,
                 iterations=1000,
                 seed=0,
                 **kwargs):
        """
        Initiates Bayesian optimizer parameters
        """

        self.verbose = kwargs.get("verbose", 1)
        self.use_gpu = kwargs.get("use_gpu", False)
        learning_rate = kwargs.get("learning_rate", 5e-2)

        if self.use_gpu and torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        else:
            torch.set_default_tensor_type(torch.DoubleTensor)

        self.surrogate_model = gpr.reconstructor(
            X_seed, y_seed, X_full, kernel, lengthscale, sparse, indpoints,
            learning_rate, iterations, self.use_gpu, self.verbose, seed)

        self.X_sparse = X_seed
        self.y_sparse = y_seed
        self.X_full = X_full
        if self.use_gpu and torch.cuda.is_available():
            self.X_sparse = self.X_sparse.cuda()
            self.y_sparse = self.y_sparse.cuda()

        self.target_function = target_function
        self.acquisition_function = acquisition_function
        self.exploration_steps = exploration_steps
        self.batch_update = batch_update
        self.batch_size = batch_size
        self.alpha = kwargs.get("alpha", None)
        self.beta = kwargs.get("beta", None)
        self.xi = kwargs.get("xi", None)
        self.lscale = kwargs.get("lscale", None)
        self.indices_all, self.vals_all = [], []
        self.target_func_vals_all = [y_seed.copy()]

    def update_posterior(self):
        """
        Updates GP posterior
        """
        X_sparse_new, y_sparse_new = gprutils.prepare_training_data(
            self.X_sparse, self.y_sparse)
        if self.use_gpu and torch.cuda.is_available():
            X_sparse_new, y_sparse_new = X_sparse_new.cuda(), y_sparse_new.cuda()
        self.surrogate_model.model.X = X_sparse_new
        self.surrogate_model.model.y = y_sparse_new
        self.surrogate_model.train()
        return

    def evaluate_function(self, indices):
        """
        Evaluates target function in the new point(s)
        """
        indices = [indices] if not self.batch_update else indices
        for idx in indices:
            self.y_sparse[tuple(idx)] = self.target_function(idx)
        self.X_sparse = gprutils.get_sparse_grid(self.y_sparse)
        self.target_func_vals_all.append(self.y_sparse.copy())
        return

    def next_point(self):
        """
        Calculates next query point(s)
        """
        indices_list, vals_list = [], []
        if self.acquisition_function == 'cb':
            acq = acqfunc.confidence_bound(
                self.surrogate_model, self.X_full,
                alpha=self.alpha, beta=self.beta)
        elif self.acquisition_function == 'ei':
            acq = acqfunc.expected_improvement(
                self.surrogate_model, self.X_full,
                self.X_sparse, xi=self.xi)
        elif self.acquisition_function == 'poi':
            acq = acqfunc.probability_of_improvement(
                self.surrogate_model, self.X_full,
                self.X_sparse, xi=self.xi)
        else:
            raise NotImplementedError(
                "Choose between 'cb', 'ei', and 'poi' acquisition functions")
        for i in range(self.batch_size):
            amax_idx = [i[0] for i in np.where(acq == acq.max())]
            indices_list.append(amax_idx)
            vals_list.append(acq.max())
            acq[tuple(amax_idx)] = acq.min() - 1
        if not self.batch_update:
            return vals_list, indices_list
        if self.lscale is None:
            lscale_ = self.surrogate_model.model.kernel.lengthscale.mean().item()
        else:
            lscale_ = self.lscale
        vals_list, indices_list = self.update_points(
            np.array(vals_list),
            np.vstack(indices_list),
            lscale_)
        return vals_list, indices_list

    @classmethod
    def update_points(cls, acqfunc_values, indices, lscale):
        """
        Returns a batch of query points whose separation distance
        is determined by kernel lengthscale
        Args:
            acqfunc_values (ndarray):
                (*N*,) numpy array with values of acquisition function
            indices (ndarray):
                (*N*, *c*) numpy array with corresponding indices,
                where c ia a number of dimensions of the dataset
            lscale (float):
                kernel lengthscale

        Returns:
            Tuple with computed indices and corresponding values
        """
        minval = acqfunc_values.min()
        new_max = acqfunc_values.max()
        new_max_id = np.argmax(acqfunc_values)
        max_val_all, max_id_all = [], []
        ck = indices[new_max_id]
        tree = spatial.cKDTree(indices)
        while new_max > minval - 1:
            max_val_all.append(new_max)
            max_id_all.append(new_max_id)
            nn_indices = tree.query_ball_point(ck, lscale)
            acqfunc_values[nn_indices] = minval - 1
            new_max = acqfunc_values.max()
            new_max_id = np.argmax(acqfunc_values)
            ck = indices[new_max_id]
        return max_val_all, indices[max_id_all].tolist()

    def checkvalues(self, idx_list, val_list):
        """
        Checks if the indices were already used
        and if so selects the next-in-line indices
        (helps not to get stuck in one point during the exploration-exploitation)

        Args:
            idx_list (list of lists with ints):
                Indices for max values of acquisition function;
                the list is ordered (max -> min)
            val_list (list with floats):
                Standard deviation values for each index in idx_list

        Returns:
            The first element in the input list (idx_list)
            if no previous occurences found.
            Otherwise, returns the next/closest value from the list.
        """
        _idx = 0
        if self.verbose:
            print('Acquisition function value {} at {}'.format(
                val_list[_idx], idx_list[_idx]))
        if len(self.indices_all) == 0:
            return idx_list[_idx], val_list[_idx]
        while 1 in [1 for a in self.indices_all if a == idx_list[_idx]]:
            if self.verbose:
                print("Finding the next max point...")
            _idx = _idx + 1
            if self.verbose:
                print('Acquisition function value {} at {}'.format(
                    val_list[_idx], idx_list[_idx]))
        return idx_list[_idx], val_list[_idx]

    def single_step(self, *args):
        """
        Single Bayesian optimization step
        """
        e = args[0]
        if self.verbose:
            print("\nExploration step {} / {}".format(
                e, self.exploration_steps))
        # train with seeded data
        if e == 0:
            self.surrogate_model.train()
        # calculate acquisition function and get next query points
        vals, inds = self.next_point()
        if not self.batch_update:
            inds, vals = self.checkvalues(inds, vals)
        # evaluate function
        self.evaluate_function(inds)
        # update posterior
        self.update_posterior()
        # store indices and values
        self.indices_all.append(inds)
        self.vals_all.append(vals)
        return

    def run(self):
        """
        Run GP-based Bayesian optimization loop
        """
        for i in range(self.exploration_steps):
            self.single_step(i)
        if self.verbose:
            print("Exploration completed")
        return self.indices_all, self.target_func_vals_all
