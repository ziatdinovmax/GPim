'''
skgpr.py
======

Gaussian process regression model with a structured kernel interpolation.

Serves as a high-level wrapper for GPyTorch's (https://gpytorch.ai)
Gaussian processes module with a structred kernel interpolation method
for easy work with scientific image (2D) and hyperspectral (3D, 4D) data.

Author: Maxim Ziatdinov (email: maxim.ziatdinov@ai4microcopy.com)
'''

import time
import numpy as np
from gpim import gprutils
import torch
import gpytorch
import warnings


class skreconstructor:
    """
    GP regression model with structured kernel interpolation
    for 2D/3D/4D image data reconstruction

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
        lengthscale (list of two lists):
            Determines lower (1st list) and upper (2nd list) bounds
            for kernel lengthscales. The number of elements in each list
            is equal to the dataset dimensionality.
        lengthscale_init (list with floats):
            Initializes lenghtscales at this values
        iterations (int):
            Number of training steps
        learning_rate (float):
            Learning rate for model training
        grid_points_ratio (float):
            Ratio of inducing points to overall points
        max_root (int):
            Maximum number of Lanczos iterations to perform
            in prediction stage
        num_batches (int):
            Number of batches for splitting the Xtest array
            (for large datasets, you may not have enough GPU memory
            to process the entire dataset at once)
        calculate_sd (bool):
            Calculates SD in prediction stage
            (possible only when num_batches == 1)
        use_gpu (bool):
            Uses GPU hardware accelerator when set to 'True'
        verbose (bool):
            Print statistics after each training iteration
        seed (int):
            for reproducibility
    """
    def __init__(self,
                 X,
                 y,
                 Xtest=None,
                 kernel='RBF',
                 lengthscale=None,
                 lengthscale_init=None,
                 iterations=50,
                 learning_rate=.1,
                 grid_points_ratio=1.,
                 maxroot=100,
                 num_batches=10,
                 calculate_sd=0,
                 use_gpu=1,
                 verbose=0,
                 seed=0):
        """
        Initiates reconstructor parameters
        and pre-processes training and test data arrays
        """
        torch.manual_seed(seed)
        if use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        gpr2sgpr_thresh = 1e4
        input_dim = np.ndim(y)
        X, y = gprutils.prepare_training_data(X, y)
        Xtest = gprutils.prepare_test_data(Xtest)
        self.X, self.y, self.Xtest = X, y, Xtest
        self.toeplitz = gpytorch.settings.use_toeplitz(True)
        self.maxroot = gpytorch.settings.max_root_decomposition_size(maxroot)
        if use_gpu and torch.cuda.is_available():
            self.X, self.y = self.X.cuda(), self.y.cuda()
            if self.Xtest is not None:
                self.Xtest = self.Xtest.cuda()
            self.toeplitz = gpytorch.settings.use_toeplitz(False)
        else:
            torch.set_default_tensor_type(torch.DoubleTensor)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        _kernel = get_kernel(kernel, input_dim,
                             use_gpu, lengthscale=lengthscale,
                             lengthscale_init=lengthscale_init)
        self.model = skgprmodel(self.X, self.y,
                                _kernel, self.likelihood,
                                input_dim, grid_points_ratio)
        if use_gpu:
            self.model.cuda()
        self.iterations = iterations
        self.num_batches = num_batches
        self.calculate_sd = calculate_sd
        self.lr = learning_rate

        self.lscales, self.noise_all = [], []
        self.hyperparams = {
            "lengthscale": self.lscales,
            "noise": self.noise_all,
        }
        self.verbose = verbose

    def train(self):
        """
        Trains GP regression model with structured kernel interpolation
        """
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(
            [{'params': self.model.parameters()}], lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.model)
        print('Model training...')
        start_time = time.time()
        for i in range(self.iterations):
            optimizer.zero_grad()
            output = self.model(self.X)
            loss = -mll(output, self.y)
            loss.backward()
            optimizer.step()
            if len(self.X) < gpr2sgpr_thresh:
                self.lscales.append(
                    self.model.covar_module.base_kernel.lengthscale.tolist()[0]
                )
            else:
                self.lscales.append(
                    self.model.covar_module.base_kernel.base_kernel.lengthscale.tolist()[0]
                )
            self.noise_all.append(
                self.model.likelihood.noise_covar.noise.item())
            if self.verbose and (i % 10 == 0 or i == self.iterations - 1):
                print('iter: {} ...'.format(i),
                      'loss: {} ...'.format(np.around(loss.item(), 4)),
                      'length: {} ...'.format(np.around(self.lscales[-1], 4)),
                      'noise: {} ...'.format(np.around(self.noise_all[-1], 7)))
            if i == 10:
                print('average time per iteration: {} s'.format(
                    np.round(time.time() - start_time, 2) / 10))
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
            self.fulldims = Xtest.shape[1:]
            if next(self.model.parameters()).is_cuda:
                self.Xtest = self.Xtest.cuda()
        if kwargs.get("num_batches") is not None:
            self.num_batches = kwargs.get("num_batches")
        if kwargs.get("max_root") is not None:
            self.max_root = kwargs.get("max_root")
        self.model.eval()
        self.likelihood.eval()
        batch_range = len(self.Xtest) // self.num_batches
        mean = np.zeros((self.Xtest.shape[0]))
        if self.calculate_sd:
            sd = np.zeros((self.Xtest.shape[0]))
        if self.calculate_sd:
            print('Calculating predictive mean and uncertainty...')
        else:
            print('Calculating predictive mean...')
        for i in range(self.num_batches):
            print("\rBatch {}/{}".format(i+1, self.num_batches), end="")
            Xtest_i = self.Xtest[i*batch_range:(i+1)*batch_range]
            with torch.no_grad(), gpytorch.settings.fast_pred_var(), self.toeplitz, self.maxroot:
                covar_i = self.likelihood(self.model(Xtest_i))
            mean[i*batch_range:(i+1)*batch_range] = covar_i.mean.detach().cpu().numpy()
            if self.calculate_sd:
                sd[i*batch_range:(i+1)*batch_range] = covar_i.stddev.detach().cpu().numpy()
        print("\nDone")
        if self.calculate_sd:
            return (mean, sd)
        return mean

    def run(self):
        """
        Combines train and step methods
        """
        self.train()
        prediction = self.predict()
        if next(self.model.parameters()).is_cuda:
            self.model.cpu()
            torch.set_default_tensor_type(torch.DoubleTensor)
            self.X, self.y = self.X.cpu(), self.y.cpu()
            self.Xtest = self.Xtest.cpu()
            torch.cuda.empty_cache()
        return prediction, self.hyperparams


class skgprmodel(gpytorch.models.ExactGP):
    """
    GP regression model with structured kernel interpolation

    Args:
        X (ndarray):
            Grid indices with dimension :math:`n \\times c`,
            where *n* is the number of observation points
            and *c* is equal to the number of coordinates
            (for example, for xyz coordinates, *c* = 3)
        y (ndarray):
            Observations (data points) with dimension n
        kernel (gpytorch kernel object):
            'RBF' or 'Matern52' kernels
        likelihood (gpytorch likelihood object):
            The Gaussian likelihood
        input_dim (int):
            Number of input dimensions
            (equal to number of feature vector columns)
        grid_points_ratio (float):
            Ratio of inducing points to overall points
    """

    def __init__(self, X, y, kernel, likelihood,
                 input_dim=3, grid_points_ratio=1.):
        """
        Initializes model parameters
        """
        super(skgprmodel, self).__init__(X, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)
        if len(self.X) > gpr2sgpr_thresh:
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


def get_kernel(kernel_type, input_dim, on_gpu=True, **kwargs):
    """
    Initializes one of the following kernels: RBF, Matern

    Args:
        kernel_type (str):
            Kernel type ('RBF', Matern52')
        input_dim (int):
            Number of input dimensions
            (equal to number of feature vector columns)
        on_gpu (bool):
            Sets default tensor type to torch.cuda.DoubleTensor
        **lengthscale (list of two lists):
            Determines lower (1st list) and upper (2nd list) bounds
            for kernel lengthscale(s);
            number of elements in each list is equal to the input dimensions
        **lengthscale_init (list with float):
            Initializes lenghtscale at this value
    Returns:
        kernel object
    """
    if on_gpu and torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)

    lscale = kwargs.get('lengthscale')
    if lscale is not None:
        lscale = gpytorch.constraints.Interval(torch.tensor(lscale[0]),
                                               torch.tensor(lscale[1]))
    # initialize the kernel
    kernel_book = lambda input_dim, lscale: {
        'RBF': gpytorch.kernels.RBFKernel(
            ard_num_dims=input_dim,
            lengthscale_constraint=lscale
            ),
        'Matern52': gpytorch.kernels.MaternKernel(
            ard_num_dims=input_dim,
            lengthscale_constraint=lscale
            )
    }
    try:
        kernel = kernel_book(input_dim, lscale)[kernel_type]
    except KeyError:
        print('Select one of the currently available kernels:',\
              '"RBF", "Matern52"')
        raise
    lscale_init = kwargs.get('lengthscale_init')
    if None not in (lscale, lscale_init):
        kernel.lengthscale = torch.tensor(lscale_init)
    return kernel
