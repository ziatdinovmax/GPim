'''
gpr.py
======

Sparse Gaussian process regression:
model training, prediction and uncertainty exploration

This module serves as a high-level wrapper for sparse Gaussian processes module
from Pyro probabilistic programming library (https://pyro.ai/)
for easy work with scientific image (2D) and hyperspectral (3D) data.

Author: Maxim Ziatdinov (email: maxim.ziatdinov@ai4microcopy.com)
'''

import time
import numpy as np
import gpim.gprutils as gprutils
import torch
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import warnings


class reconstructor:
    """
    Class for Gaussian process regression-based reconstuction
    of sparse 2D image and 3D spectroscopic datasets,
    and sample exploration with hyperspectral measurements
    based on maximal uncertainty reduction.

    Args:
        X (ndarray):
            Grid indices with dimensions :math:`c \\times N \\times M`
            or :math:`c \\times N \\times M \\times L`
            where *c* is equal to the number of coordinates
            (for example, for *xyz* coordinates, *c* = 3)
        y (ndarray):
            Observations (data points) with dimensions
            :math:`N \\times M` or :math:`N \\times M \\times L`.
            Typically, for 2D image *N* and *M* are image height and width,
            whereas for 3D hyperspectral data *N* and *M* are spatial dimensions
            and *L* is a spectorcopic dimension (e.g. voltage or wavelength).
        Xtest (ndarray):
            "Test" points (for prediction with a trained GP model)
            with dimensions :math:`N \\times M` or :math:`N \\times M \\times L`
        kernel (str):
            Kernel type ('RBF', 'Matern52', 'RationalQuadratic')
        lengthscale (list of two lists):
            Determines lower (1st list) and upper (2nd list) bounds
            for kernel lengthscales. The number of elements in each list
            is equal to the dataset dimensionality.
        indpoints (int):
            Number of inducing points for SparseGPRegression
        learning_rate (float):
            Learning rate for model training
        iterations (int): Number of SVI training iteratons
        use_gpu (bool):
            Uses GPU hardware accelerator when set to 'True'.
            Notice that for large datasets training model without GPU
            is extremely slow.
        verbose (bool):
            Prints training statistics after each 100th training iteration
        **amplitude (float): kernel variance or amplitude squared
    """
    def __init__(self,
                 X,
                 y,
                 Xtest,
                 kernel,
                 lengthscale=None,
                 indpoints=1000,
                 learning_rate=5e-2,
                 iterations=1000,
                 use_gpu=False,
                 verbose=False,
                 seed=0,
                 patience=20,
                 **kwargs):
        """
        Initiates reconstructor parameters
        and pre-processes training and test data arrays
        """
        self.verbose = verbose
        torch.manual_seed(seed)
        pyro.set_rng_seed(seed)
        pyro.clear_param_store()
        if use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
            use_gpu = True
        else:
            torch.set_default_tensor_type(torch.DoubleTensor)
            use_gpu = False
        input_dim = np.ndim(y)
        self.X, self.y = gprutils.prepare_training_data(X, y)
        if indpoints > len(self.X):
            indpoints = len(self.X)
        Xu = self.X[::len(self.X) // indpoints]
        if lengthscale is None:
            lengthscale = [[0. for l in range(input_dim)],
                           [np.mean(y.shape) / 2 for l in range(input_dim)]]
        kernel = get_kernel(kernel, input_dim,
                            lengthscale, use_gpu,
                            amplitude=kwargs.get('amplitude'))
        self.fulldims = Xtest.shape[1:]
        self.Xtest = gprutils.prepare_test_data(Xtest)
        if use_gpu:
            self.X = self.X.cuda()
            self.y = self.y.cuda()
            self.Xtest = self.Xtest.cuda()
        self.sgpr = gp.models.SparseGPRegression(
            self.X, self.y, kernel, Xu, jitter=1.0e-5)
        if self.verbose:
            print("# of inducing points for GP regression: {}".format(len(Xu)))
        if use_gpu:
            self.sgpr.cuda()
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.patience = patience
        self.hyperparams = {}
        self.indpoints_all = []
        self.lscales, self.noise_all, self.amp_all = [], [], []
        self.hyperparams = {
            "lengthscale": self.lscales,
            "noise": self.noise_all,
            "variance": self.amp_all,
            "inducing_points": self.indpoints_all
        }
        

    def train(self, **kwargs):
        """
        Training sparse GP regression model

        Args:
            **learning_rate (float): learning rate
            **iterations (int): number of SVI training iteratons
        """
        if kwargs.get("learning_rate") is not None:
            self.learning_rate = kwargs.get("learning_rate")
        if kwargs.get("iterations") is not None:
            self.iterations = kwargs.get("iterations")
        pyro.clear_param_store()
        optimizer = torch.optim.Adam(self.sgpr.parameters(), lr=self.learning_rate)
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        start_time = time.time()
        if self.verbose:
            print('Model training...')
        loss_register = np.zeros(self.iterations+1)
        loss_register[0] = 1e+5
        bad_epochs = 0
        for i in range(self.iterations):
            optimizer.zero_grad()
            loss = loss_fn(self.sgpr.model, self.sgpr.guide)
            loss_register[i+1] = loss.item()
            if ( (loss_register[i]-loss_register[i+1]) < (1e-5*np.abs(loss_register[i])) ):
                bad_epochs += 1
            else:
                bad_epochs = 0

            if bad_epochs == self.patience:
                if self.verbose:
                    print("The training is stopped at {} iterations due to bad epochs".format(i))
                break
                 
            loss.backward()
            optimizer.step()
            self.lscales.append(self.sgpr.kernel.lengthscale_map.tolist())
            self.amp_all.append(self.sgpr.kernel.variance_map.item())
            self.noise_all.append(self.sgpr.noise.item())
            self.indpoints_all.append(self.sgpr.Xu.detach().cpu().numpy())
            if self.verbose and (i % 100 == 0 or i == self.iterations - 1):
                print('iter: {} ...'.format(i),
                      'loss: {} ...'.format(np.around(loss.item(), 4)),
                      'amp: {} ...'.format(np.around(self.amp_all[-1], 4)),
                      'length: {} ...'.format(np.around(self.lscales[-1], 4)),
                      'noise: {} ...'.format(np.around(self.noise_all[-1], 7)))
                print(bad_epochs)
            if self.verbose and i == 100:
                print('average time per iteration: {} s'.format(
                    np.round(time.time() - start_time, 2) / 100))
        if self.verbose:
            print('training completed in {} s'.format(
                np.round(time.time() - start_time, 2)))
            print('Final parameter values:\n',
                  'amp: {}, lengthscale: {}, noise: {}'.format(
                    np.around(self.sgpr.kernel.variance_map.item(), 4),
                    np.around(self.sgpr.kernel.lengthscale_map.tolist(), 4),
                    np.around(self.sgpr.noise.item(), 7)))
        return

    def predict(self):
        """
        Use trained GP regression model to make predictions

        Returns:
            Predictive mean and variance
        """
        if self.verbose:
            print("Calculating predictive mean and variance...", end=" ")
        with torch.no_grad():
            mean, cov = self.sgpr(self.Xtest, full_cov=False, noiseless=False)
        if self.verbose:
            print("Done")
        return mean.cpu().numpy(), cov.sqrt().cpu().numpy()

    def run(self, **kwargs):
        """
        Train the initialized model and calculate predictive mean and variance

        Args:
            **learning_rate (float):
                learning rate for GP regression model training
            **steps (int):
                number of SVI training iteratons

        Returns:
            Predictive mean and SD as flattened ndarrays and
            dictionary with hyperparameters evolution
            as a function of SVI steps

        """
        if kwargs.get("learning_rate") is not None:
            self.learning_rate = kwargs.get("learning_rate")
        if kwargs.get("iterations") is not None:
            self.iterations = kwargs.get("iterations")
        self.train(learning_rate=self.learning_rate, iterations=self.iterations)
        mean, sd = self.predict()
        if next(self.sgpr.parameters()).is_cuda:
            self.sgpr.cpu()
            torch.set_default_tensor_type(torch.DoubleTensor)
            self.X, self.y = self.X.cpu(), self.y.cpu()
            self.Xtest = self.Xtest.cpu()
            torch.cuda.empty_cache()
        return mean, sd, self.hyperparams

    def step(self, dist_edge, **kwargs):
        """
        Performs single train-predict step for exploration analysis
        returning a new point with maximum uncertainty

        Args:
            dist_edge (list with two integers):
                edge regions not considered in max uncertainty evaluation
            **learning_rate (float):
                learning rate for GP regression model training
            **steps (int):
                number of SVI training iteratons

        Returns:
            lists of indices and values for points with maximum uncertainty,
            predictive mean and standard deviation (as flattened numpy arrays)
        """
        if kwargs.get("learning_rate") is not None:
            self.learning_rate = kwargs.get("learning_rate")
        if kwargs.get("iterations") is not None:
            self.iterations = kwargs.get("iterations")
        # train a model
        self.train(learning_rate=self.learning_rate, iterations=self.iterations)
        # make prediction
        mean, sd = self.predict()
        # find point with maximum uncertainty
        sd_ = sd.reshape(self.fulldims[0], self.fulldims[1], self.fulldims[2])
        amax, uncert_list = gprutils.max_uncertainty(sd_, dist_edge)
        return amax, uncert_list, mean, sd


def get_kernel(kernel_type, input_dim, lengthscale, use_gpu=False, **kwargs):
    """
    Initalizes one of the following kernels:
    RBF, Rational Quadratic, Matern

    Args:
        kernel_type (str):
            kernel type ('RBF', 'Rational Quadratic', Matern52')
        input_dim (int):
            number of input dimensions
            (equal to number of feature vector columns)
        lengthscale (list of two lists):
            determines lower (1st list) and upper (2nd list) bounds
            for kernel lengthscale(s). Number of elements in each list
            is equal to the input dimensions
        use_gpu (bool):
            sets default tensor type to torch.cuda.DoubleTensor
        **amplitude (list with two floats):
            determines bounds on kernel amplitude parameter
            (default is from 1e-4 to 10)

    Returns:
        Pyro kernel object
    """
    if use_gpu and torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)

    amp = kwargs.get('amplitude')
    lscale = lengthscale
    amp = [1e-4, 10.] if amp is None else amp
    # Needed in Pyro < 1.0.0
    lscale_ = torch.tensor(lscale[0]) + 1e-5

    # initialize the kernel
    kernel_book = lambda input_dim: {
        'RBF': gp.kernels.RBF(
            input_dim, lengthscale=lscale_
            ),
        'RationalQuadratic': gp.kernels.RationalQuadratic(
            input_dim, lengthscale=lscale_
            ),
        'Matern52': gp.kernels.Matern52(
            input_dim, lengthscale=lscale_
            )
    }

    try:
        kernel = kernel_book(input_dim)[kernel_type]
    except KeyError:
        print('Select one of the currently available kernels:',\
              '"RBF", "RationalQuadratic", "Matern52"')
        raise

    with warnings.catch_warnings():  # TODO: use PyroSample to set priors
        warnings.filterwarnings("ignore", category=UserWarning)

        # set priors
        kernel.set_prior(
            "variance",
            dist.Uniform(
                torch.tensor(amp[0]),
                torch.tensor(amp[1])
            )
        )
        kernel.set_prior(
            "lengthscale",
            dist.Uniform(
                torch.tensor(lscale[0]),
                torch.tensor(lscale[1])
            ).independent()
        )

    return kernel
