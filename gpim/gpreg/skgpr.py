'''
skgpr.py
======
Gaussian process regression model with a structured kernel interpolation
or a spectral mixture kernel. Serves as a high-level wrapper
for GPyTorch's (https://gpytorch.ai) Gaussian process modules with
structred kernel interpolation and spectral mixture kernel methods.

Author: Maxim Ziatdinov (email: maxim.ziatdinov@ai4microcopy.com)
'''

import time
import numpy as np
from gpim.kernels import gpytorch_kernels
from gpim import gprutils
import torch
import gpytorch
import warnings


class skreconstructor:
    """
    GP regression model with structured kernel interpolation
    or spectral mixture kernel for 2D/3D/4D image data reconstruction

    Args:
        X (ndarray):
            Grid indices with dimension :math:`c \\times N \\times M`,
            :math:`c \\times N \\times M \\times L` or
            :math:`c \\times N \\times M \\times L \\times K`,
            where *c* is equal to the number of coordinates
            (for example, for xyz coordinates, *c* = 3)
        y (ndarray):
            Observations (data points) with dimension :math:`N \\times M`,
            :math:`N \\times M \\times L` or
            :math:`N \\times M \\times L \\times K`.
            Typically, for 2D image *N* and *M* are image height and width.
            For 3D hyperspectral data *N* and *M* are spatial dimensions
            and *L* is a "spectroscopic" dimension (e.g. voltage or wavelength).
            For 4D datasets, both *L* and *K* are "spectroscopic" dimensions.
        Xtest (ndarray):
            "Test" points (for prediction with a trained GP model)
            with dimension :math:`N \\times M`, :math:`N \\times M \\times L`
            or :math:`N \\times M \\times L \\times K`
        kernel (str):
            Kernel type ('RBF' or 'Matern52')
        lengthscale (list of int  list of two list with ins):
            Determines lower (1st list) and upper (2nd list) bounds
            for kernel lengthscales. The number of elements in each list
            is equal to the dataset dimensionality.
        ski (bool):
            Perform structured kernel interpolation GP. Set to True by default.
        iterations (int):
            Number of training steps
        learning_rate (float):
            Learning rate for model training
        use_gpu (bool):
            Uses GPU hardware accelerator when set to 'True'
        verbose (int):
            Level of verbosity (0, 1, or 2)
        seed (int):
            for reproducibility
        **grid_points_ratio (float):
            Ratio of inducing points to overall points
        **n_mixtures (int):
            number of mixtures for spectral mixture kernel
        **isotropic (bool):
            one kernel lengthscale in all dimensions
        **max_root (int):
            Maximum number of Lanczos iterations to perform
            in prediction stage
        **num_batches (int):
            Number of batches for splitting the Xtest array
            (for large datasets, you may not have enough GPU memory
            to process the entire dataset at once)
        **precision (str):
            Choose between single ('single') and double ('double') precision
    """
    def __init__(self,
                 X,
                 y,
                 Xtest=None,
                 kernel='RBF',
                 lengthscale=None,
                 ski=True,
                 learning_rate=.1,
                 iterations=50,
                 use_gpu=1,
                 verbose=1,
                 seed=0,
                 **kwargs):
        """
        Initiates reconstructor parameters
        and pre-processes training and test data arrays
        """
        self.precision = kwargs.get("precision", "double")
        if self.precision == 'single':
            self.tensor_type = torch.FloatTensor
            self.tensor_type_gpu = torch.cuda.FloatTensor
        else:
            self.tensor_type = torch.DoubleTensor
            self.tensor_type_gpu = torch.cuda.DoubleTensor
        torch.manual_seed(seed)
        if use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.set_default_tensor_type(self.tensor_type_gpu)
        input_dim = np.ndim(y)
        if Xtest is not None:
            self.fulldims = Xtest.shape[1:]
        else:
            self.fulldims = X.shape[1:]
        X, y = gprutils.prepare_training_data(X, y, precision=self.precision)
        if Xtest is not None:
            Xtest = gprutils.prepare_test_data(Xtest, precision=self.precision)
        self.X, self.y, self.Xtest = X, y, Xtest
        self.do_ski = ski
        if kernel == "Spectral":
            self.do_ski = False
        self.toeplitz = gpytorch.settings.use_toeplitz(True)
        maxroot = kwargs.get("maxroot", 100)
        self.maxroot = gpytorch.settings.max_root_decomposition_size(maxroot)
        if use_gpu and torch.cuda.is_available():
            self.X, self.y = self.X.cuda(), self.y.cuda()
            if self.Xtest is not None:
                self.Xtest = self.Xtest.cuda()
            self.toeplitz = gpytorch.settings.use_toeplitz(False)
        else:
            torch.set_default_tensor_type(self.tensor_type)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        isotropic = kwargs.get("isotropic")
        n_mixtures = kwargs.get("n_mixtures")
        _kernel = gpytorch_kernels.get_kernel(
            kernel, input_dim, use_gpu, lengthscale=lengthscale,
            isotropic=isotropic, precision=self.precision,
            n_mixtures=n_mixtures)
        grid_points_ratio = kwargs.get("grid_points_ratio", 1.)
        self.model = skgprmodel(self.X, self.y,
                                _kernel, self.likelihood, input_dim,
                                grid_points_ratio, self.do_ski)
        if use_gpu:
            self.model.cuda()
        self.iterations = iterations
        self.num_batches = kwargs.get("num_batches", 1)
        self.learning_rate = learning_rate

        self.noise_all = []
        if kernel == "Spectral":
            self.scales, self.means, self.weights = [], [], []
            self.hyperparams = {
                "scales": self.scales,
                "means": self.means,
                "weights": self.weights,
                "noise": self.noise_all
            }
        else:
            self.lscales = []
            self.hyperparams = {
                "lengthscale": self.lscales,
                "noise": self.noise_all,
            }
        self.verbose = verbose

    def train(self, **kwargs):
        """
        Training GP regression model

        Args:
            **learning_rate (float): learning rate
            **iterations (int): number of SVI training iteratons
        """
        if kwargs.get("learning_rate") is not None:
            self.learning_rate = kwargs.get("learning_rate")
        if kwargs.get("iterations") is not None:
            self.iterations = kwargs.get("iterations")
        if kwargs.get("verbose") is not None:
            self.verbose = kwargs.get("verbose")
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(
            [{'params': self.model.parameters()}], lr=self.learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.model)
        if self.verbose:
            print('Model training...')
        start_time = time.time()
        for i in range(self.iterations):
            optimizer.zero_grad()
            output = self.model(self.X)
            loss = -mll(output, self.y)
            loss.backward()
            optimizer.step()
            if self.do_ski:
                self.lscales.append(
                    self.model.covar_module.base_kernel.base_kernel.lengthscale.tolist()[0]
                )
            else:
                if not hasattr(self.model.covar_module, "num_mixtures"):
                    self.lscales.append(
                        self.model.covar_module.base_kernel.lengthscale.tolist()[0]
                    )
                else:
                    weights = self.model.covar_module.mixture_weights.detach()
                    scales = 1 / torch.sqrt(
                        self.model.covar_module.mixture_scales.detach())
                    means = 1 / self.model.covar_module.mixture_means.detach()
                    self.weights.append(weights.cpu().numpy().copy())
                    self.scales.append(scales.cpu().numpy().copy())
                    self.means.append(means.cpu().numpy().copy())
            self.noise_all.append(
                self.model.likelihood.noise_covar.noise.item())
            if self.verbose == 2 and (i % 10 == 0 or i == self.iterations - 1):
                if not hasattr(self.model.covar_module, "num_mixtures"):
                    template = 'iter: {} ... loss: {} ... length: {} ... noise: {} ...'
                    print(template.format(
                        i, np.around(loss.item(), 4),
                        np.around(self.lscales[-1], 4),
                        np.around(self.noise_all[-1], 7)))
                else:
                    template = 'iter: {} ... loss: {} ... noise: {} ...'
                    print(template.format(
                        i, np.around(loss.item(), 4),
                        np.around(self.noise_all[-1], 7)))
                    sort_idx = torch.argsort(weights, descending=True)
                    weights_sorted = weights[sort_idx].double()
                    scales_sorted = scales[sort_idx]
                    means_sorted = means[sort_idx]
                    print("weight" + 10*" " +  "mean" + 10*" " + "scale")
                    for w, m, s in zip(weights_sorted, means_sorted, scales_sorted):
                        print("{}  {}  {}".format(
                            w.cpu().numpy().round(4),
                            m[0].cpu().numpy().round(4),
                            s[0].cpu().numpy().round(4)))
            if self.verbose and i == 10:
                print('average time per iteration: {} s'.format(
                    np.round(time.time() - start_time, 2) / 10))
        if self.verbose:
            print('training completed in {} s'.format(
                np.round(time.time() - start_time, 2)))
            if not hasattr(self.model.covar_module, "num_mixtures"):
                print('Final parameter values:\n',
                    'lengthscale: {}, noise: {}'.format(
                        np.around(self.lscales[-1], 4),
                        np.around(self.noise_all[-1], 7)))
            else:
                print('Final parameter values:\n',
                'noise: {}'.format(np.around(self.noise_all[-1], 7)))
        return

    def predict(self, Xtest=None, **kwargs):
        """
        Makes a prediction with trained GP regression model

        Args:
            Xtest (ndarray):
                "Test" points (for prediction with a trained GP model)
                with dimension :math:`N \\times M`, :math:`N \\times M \\times L`
                or :math:`N \\times M \\times L \\times K`
            max_root (int):
                Maximum number of Lanczos iterations to perform
                in prediction stage
            num_batches (int):
                Number of batches for splitting the Xtest array
                (for large datasets, you may not have enough GPU memory
                to process the entire dataset at once)
        """
        if Xtest is None and self.Xtest is None:
            warnings.warn(
                "No test data provided. Using training data for prediction",
                UserWarning)
            self.Xtest = self.X
        elif Xtest is not None:
            self.Xtest = gprutils.prepare_test_data(
                Xtest, precision=self.precision)
            self.fulldims = Xtest.shape[1:]
            if next(self.model.parameters()).is_cuda:
                self.Xtest = self.Xtest.cuda()
        if kwargs.get("verbose") is not None:
            self.verbose = kwargs.get("verbose")
        if kwargs.get("num_batches") is not None:
            self.num_batches = kwargs.get("num_batches")
        if kwargs.get("max_root") is not None:
            self.max_root = kwargs.get("max_root")
        self.model.eval()
        self.likelihood.eval()
        batch_range = len(self.Xtest) // self.num_batches
        dtype_ = np.float32 if self.precision == 'single' else np.float64 
        mean = np.zeros((self.Xtest.shape[0]), dtype_)
        sd = np.zeros((self.Xtest.shape[0]), dtype_)
        if self.verbose:
            print('Calculating predictive mean and uncertainty...')
        for i in range(self.num_batches):
            if self.verbose:
                print("\rBatch {}/{}".format(i+1, self.num_batches), end="")
            Xtest_i = self.Xtest[i*batch_range:(i+1)*batch_range]
            with torch.no_grad(), gpytorch.settings.fast_pred_var(), self.toeplitz, self.maxroot:
                covar_i = self.likelihood(self.model(Xtest_i))
            mean[i*batch_range:(i+1)*batch_range] = covar_i.mean.detach().cpu().numpy()
            sd[i*batch_range:(i+1)*batch_range] = covar_i.stddev.detach().cpu().numpy()
        sd = sd.reshape(self.fulldims)
        mean = mean.reshape(self.fulldims)
        if self.verbose:
            print("\nDone")
        return mean, sd

    def run(self):
        """
        Combines train and step methods
        """
        self.train()
        mean, sd = self.predict()
        if next(self.model.parameters()).is_cuda:
            self.model.cpu()
            torch.set_default_tensor_type(self.tensor_type)
            self.X, self.y = self.X.cpu(), self.y.cpu()
            self.Xtest = self.Xtest.cpu()
            torch.cuda.empty_cache()
        return mean, sd, self.hyperparams

    def step(self, acquisition_function=None,
             batch_size=100, batch_update=False,
             lscale=None, **kwargs):
        """
        Performs single train-predict step and computes next query point with
        maximum value of acquisition function. Notice that it doesn't seem to
        work properly with a structred kernel.

        Args:
            acquisition_function (python function):
                Function that takes two parameters, mean and sd,
                and applies some math operation to them
                (e.g. :math:`\\upmu - 2 \\times \\upsigma`)
            batch_size (int):
                Number of query points to return
            batch_update:
                Filters the query points based on the specified lengthscale
            lscale (float):
                Lengthscale determining the separation (euclidean)
                distance between query points. Defaults to the kernel
                lengthscale
            **learning_rate (float):
                Learning rate for GP regression model training
            **steps (int):
                Number of SVI training iteratons
        Returns:
            Lists of indices and values for points with maximum uncertainty,
            predictive mean and standard deviation (as flattened numpy arrays)

        """
        if self.do_ski or hasattr(self.model.covar_module, "num_components"):
            raise NotImplementedError(
        "The Bayesian optimization routines are not available for structured or spectral kernel")
        if kwargs.get("learning_rate") is not None:
            self.learning_rate = kwargs.get("learning_rate")
        if kwargs.get("iterations") is not None:
            self.iterations = kwargs.get("iterations")
        if lscale is None:
            lscale = self.model.covar_module.base_kernel.lengthscale.mean().item()
        # train a model
        self.train(learning_rate=self.learning_rate, iterations=self.iterations)
        # make prediction
        mean, sd = self.predict()
        # find point with maximum value of acquisition function
        sd_ = sd.reshape(self.fulldims)
        mean_ = mean.reshape(self.fulldims)
        vals, inds = gprutils.acquisition(
            mean_, sd_, acquisition_function,
            batch_size, batch_update, lscale)
        return vals, inds, mean, sd


class skgprmodel(gpytorch.models.ExactGP):
    """
    GP regression model with structured kernel interpolation
    or spectral mixture kernel.
    
    Args:
        X (ndarray):
            Grid indices with dimension :math:`n \\times c`,
            where *n* is the number of observation points
            and *c* is equal to the number of coordinates
            (for example, for xyz coordinates, *c* = 3)
        y (ndarray):
            Observations (data points) with dimension n
        kernel (gpytorch kernel object):
            Kernel
        likelihood (gpytorch likelihood object):
            The Gaussian likelihood
        input_dim (int):
            Number of input dimensions
            (equal to number of feature vector columns)
        grid_points_ratio (float):
            Ratio of inducing points to overall points
    """

    def __init__(self, X, y, kernel, likelihood,
                 input_dim=3, grid_points_ratio=1.,
                 do_ski=False):
        """
        Initializes model parameters
        """
        super(skgprmodel, self).__init__(X, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if hasattr(kernel, "num_mixtures"):
            self.covar_module = kernel
            self.covar_module.initialize_from_data(X, y)
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(kernel)
        if not hasattr(kernel, "num_mixtures") and do_ski:
            grid_size = gpytorch.utils.grid.choose_grid_size(
                X, ratio=grid_points_ratio)
            self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                self.covar_module, grid_size=grid_size, num_dims=input_dim)

    def forward(self, x):
        """
        Forward path
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
