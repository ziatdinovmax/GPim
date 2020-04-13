'''
vgpr.py
======
Gaussian process regression model for vector-valued functions.
Serves as a high-level wrapper for GPyTorch's (https://gpytorch.ai)
Gaussian processes with correlated and independent output dimensions.
Author: Maxim Ziatdinov (email: maxim.ziatdinov@ai4microcopy.com)
'''

import time
import numpy as np
from gpim import gprutils
from gpim.kernels import gpytorch_kernels
import torch
import gpytorch
import warnings


class vreconstructor:
    """
    Multi-output GP regression model for vector-valued
    2D/3D/4D functions.

    Args:
        X (ndarray):
            Grid indices with dimension :math:`c \\times N \\times M`,
            :math:`c \\times N \\times M \\times L` or
            :math:`c \\times N \\times M \\times L \\times K`,
            where *c* is equal to the number of coordinates
            (for example, for xyz coordinates, *c* = 3)
        y (ndarray):
            Observations (data points) with dimension :math:`N \\times M`,
            :math:`N \\times M \\times L \\times d` or
            :math:`N \\times M \\times L \\times K \\times d`,
            where d is a number of output dimensions.
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
        independent (bool):
            Indicates whether output dimensions are independent or correlated
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
        **isotropic (bool):
            one kernel lengthscale in all dimensions
        **max_root (int):
            Maximum number of Lanczos iterations to perform
            in prediction stage
        **num_batches (int):
            Number of batches for splitting the Xtest array
            (for large datasets, you may not have enough GPU memory
            to process the entire dataset at once)
    """
    def __init__(self,
                 X,
                 y,
                 Xtest=None,
                 kernel='RBF',
                 lengthscale=None,
                 independent=False,
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
        precision = kwargs.get("precision", "double")
        if precision == 'single':
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
        input_dim = np.ndim(y) - 1
        X, y = gprutils.prepare_training_data(X, y, vector_valued=True)
        num_tasks = y.shape[-1]
        if Xtest is not None:
            self.fulldims = Xtest.shape[1:] + (num_tasks,)
        else:
            self.fulldims = X.shape[1:] + (num_tasks,)
        if Xtest is not None:
            Xtest = gprutils.prepare_test_data(Xtest)
        self.X, self.y, self.Xtest = X, y, Xtest
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
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks)
        isotropic = kwargs.get("isotropic")
        _kernel = gpytorch_kernels.get_kernel(
            kernel, input_dim, use_gpu,
            lengthscale=lengthscale, isotropic=isotropic)

        if not independent:
            self.model = vgprmodel(self.X, self.y,
                                    _kernel, self.likelihood, num_tasks)
        else:
            self.model = ivgprmodel(self.X, self.y,
                                    _kernel, self.likelihood, num_tasks)

        if use_gpu:
            self.model.cuda()
        self.iterations = iterations
        self.num_batches = kwargs.get("num_batches", 1)
        self.learning_rate = learning_rate
        self.independent = independent
        self.lscales, self.noise_all = [], []
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
            if not self.independent:
                self.lscales.append(
                    self.model.covar_module.data_covar_module.lengthscale.tolist()[0])
            else:
                self.lscales.append(
                    self.model.covar_module.base_kernel.lengthscale.tolist()[0])
            self.noise_all.append(
                self.model.likelihood.noise_covar.noise.tolist())
            if self.verbose == 2 and (i % 10 == 0 or i == self.iterations - 1):
                print('iter: {} ...'.format(i),
                      'loss: {} ...'.format(np.around(loss.item(), 4)),
                      'length: {} ...'.format(np.around(self.lscales[-1], 4)),
                      'noise: {} ...'.format(np.around(self.noise_all[-1], 7)))
            if self.verbose and i == 10:
                print('average time per iteration: {} s'.format(
                    np.round(time.time() - start_time, 2) / 10))
        if self.verbose:
            print('training completed in {} s'.format(
                np.round(time.time() - start_time, 2)))
            print('Final parameter values:\n',
                'lengthscale: {}, noise: {}'.format(
                    np.around(self.lscales[-1], 4),
                    np.around(self.noise_all[-1], 7)))
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
            self.Xtest = gprutils.prepare_test_data(Xtest)
            self.fulldims = Xtest.shape[1:] + (self.y.shape[-1],)
            if next(self.model.parameters()).is_cuda:
                self.Xtest = self.Xtest.cuda()
        if kwargs.get("verbose") is not None:
            self.verbose = kwargs.get("verbose")
        if kwargs.get("num_batches") is not None:
            self.num_batches = kwargs.get("num_batches")
        if kwargs.get("max_root") is not None:
            self.max_root = kwargs.get("max_root")
        n_samples = kwargs.get("n_samples", 100)
        self.model.eval()
        self.likelihood.eval()
        batch_range = len(self.Xtest) // self.num_batches
        mean = np.zeros((self.Xtest.shape[0], self.y.shape[-1]))
        sd = np.zeros((self.Xtest.shape[0], self.y.shape[-1]))
        if self.verbose:
            print('Calculating predictive mean and uncertainty...')
        for i in range(self.num_batches):
            if self.verbose:
                print("\rBatch {}/{}".format(i+1, self.num_batches), end="")
            Xtest_i = self.Xtest[i*batch_range:(i+1)*batch_range]
            with torch.no_grad(), gpytorch.settings.fast_pred_var(), self.toeplitz, self.maxroot:
                covar_i = self.likelihood(self.model(Xtest_i))
            samples = torch.cat([covar_i.rsample() for _ in range(n_samples)])
            samples = samples.reshape(n_samples, Xtest_i.shape[0], self.y.shape[1])
            mean_i = torch.mean(samples, dim=0)
            sd_i = torch.sqrt(torch.var(samples, dim=0))
            mean[i*batch_range:(i+1)*batch_range] = mean_i.detach().cpu().numpy()
            sd[i*batch_range:(i+1)*batch_range] = sd_i.detach().cpu().numpy()
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


class vgprmodel(gpytorch.models.ExactGP):
    """
    GP regression model for vector-valued functions
    with correlated output dimensions

    Args:
        X (ndarray):
            Grid indices with dimension :math:`n \\times c`,
            where *n* is the number of observation points
            and *c* is equal to the number of coordinates
            (for example, for xyz coordinates, *c* = 3)
        y (ndarray):
            Observations (data points) with dimension
            :math:`n \\times d`, where d is number of
            the function components
        kernel (gpytorch kernel object):
            'RBF' or 'Matern52' kernels
        likelihood (gpytorch likelihood object):
            The Gaussian likelihood
        num_tasks (int):
            Number of tasks (equal to number of outputs)
    """
    def __init__(self, X, y, kernel, likelihood, num_tasks):
        super(vgprmodel, self).__init__(X, y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks)
        self.covar_module = gpytorch.kernels.MultitaskKernel(kernel, num_tasks)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class ivgprmodel(gpytorch.models.ExactGP):
    """
    GP regression model for vector-valued functions
    with independent output dimensions

    Args:
        X (ndarray):
            Grid indices with dimension :math:`n \\times c`,
            where *n* is the number of observation points
            and *c* is equal to the number of coordinates
            (for example, for xyz coordinates, *c* = 3)
        y (ndarray):
            Observations (data points) with dimension
            :math:`n \\times d`, where d is number of
            the function components
        kernel (gpytorch kernel object):
            'RBF' or 'Matern52' kernels
        likelihood (gpytorch likelihood object):
            The Gaussian likelihood
        num_tasks (int):
            Number of tasks (equal to number of outputs)
    """
    def __init__(self, X, y, kernel, likelihood, num_tasks):
        super(ivgprmodel, self).__init__(X, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_tasks]))
        kernel.batch_shape = torch.Size([num_tasks])
        self.covar_module = gpytorch.kernels.ScaleKernel(
            kernel, batch_shape=torch.Size([num_tasks]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x))
