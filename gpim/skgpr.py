'''
Gaussian process regression model with a structured kernel interpolation.

Serves as a high-level wrapper for GPyTorch's (https://gpytorch.ai) 
Gaussian processes module with a structred kernel interpolation method
for easy work with scientific image (2D) and hyperspectral (3D, 4D) data. 
WORK IN PROGRESS. More details TBA

Author: Maxim Ziatdinov (email: maxim.ziatdinov@ai4microcopy.com)
'''

import numpy as np
from gpim import gprutils
import torch
import gpytorch


class skreconstructor:
    """
    GP regression model with structured kernel interpolation
    for 2D/3D images reconstruction

    Args:
        X:  c x  N x M x L or c x N x M ndarray
            Grid indices.
            c is equal to the number of coordinate dimensions.
            For example, for xyz coordinates, c = 3.
        y: N x M x L or N x M ndarray
            Observations (data points)
        kernel: str
            kernel type
        input_dim: int
            number of input dimensions
            (equal to number of feature vector columns)
        lengthscale: list of two lists
            determines lower (1st list) and upper (2nd list) bounds
            for kernel lengthscale(s)
        lengthscale_init: list with float(s)
            initializes lenghtscale at this value
        iterations: int
            number of training steps
        learning_rate: float
            learning rate for model training
        grid_points_ratio: float
            ratio of inducing points to overall points
        max_root: int
            Maximum number of Lanczos iterations to perform
            in prediction stage
        num_batches: int
            number of batches for splitting the Xtest array
            (for large datasets, you may not have enough GPU memory
            to process the entire dataset at once)
        calculate_sd: bool
            Whether to calculate SD in prediction stage
            (possible only when num_batches = 1)
        use_gpu: bool
            Uses GPU hardware accelerator when set to 'True'
        verbose: bool
            Print statistics after each training iteration
    """
    def __init__(self, 
                 X, 
                 y, 
                 Xtest,
                 kernel='Matern52', 
                 lengthscale=None, 
                 lengthscale_init=None,
                 iterations=50, 
                 learning_rate=.1, 
                 grid_points_ratio=1.,
                 maxroot=100, 
                 num_batches=10, 
                 calculate_sd=0,
                 use_gpu=1, 
                 verbose=0):
        
        input_dim = np.ndim(y)
        X, y = gprutils.prepare_training_data(X, y)
        Xtest = gprutils.prepare_test_data(Xtest)
        self.X, self.y, self.Xtest = X, y, Xtest
        self.toeplitz = gpytorch.settings.use_toeplitz(True)
        self.maxroot = gpytorch.settings.max_root_decomposition_size(maxroot)
        if use_gpu:
            torch.cuda.empty_cache()
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
            self.X, self.y = self.X.cuda(), self.y.cuda()
            self.Xtest = self.Xtest.cuda()
            self.toeplitz = gpytorch.settings.use_toeplitz(False)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        _kernel = get_kernel(kernel, input_dim,
                             use_gpu, lengthscale=lengthscale,
                             lengthscale_init=lengthscale_init)
        self.model = skgprmodel(self.X, self.y,
                                _kernel, self.likelihood,
                                input_dim, grid_points_ratio)
        if use_gpu:
            self.model.cuda()
        self.steps = iterations
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
        print('Model training...')
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(
            [{'params': self.model.parameters()}], lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.model)
        for i in range(self.steps):
            optimizer.zero_grad()
            output = self.model(self.X)
            loss = -mll(output, self.y)
            loss.backward()
            self.lscales.append(
                self.model.covar_module.base_kernel.base_kernel.lengthscale.tolist()[0]
            )
            self.noise_all.append(
                self.model.likelihood.noise_covar.noise.item())
            if self.verbose:
                print('iter: {} ...'.format(i),
                      'loss: {} ...'.format(np.around(loss.item(), 4)),
                      'length: {} ...'.format(np.around(self.lscales[-1], 4)),
                      'noise: {} ...'.format(np.around(self.noise_all[-1], 7)))
            optimizer.step()
        return

    def predict(self, **kwargs):
        "Makes a prediction with trained GP regression model"
        if kwargs.get("Xtest") is not None:
            self.Xtest = gprutils.prepare_test_data(kwargs.get("Xtest"))
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
            mean[i*batch_range:(i+1)*batch_range] = covar_i.mean.cpu().numpy()
            if self.calculate_sd:
                sd[i*batch_range:(i+1)*batch_range] = covar_i.stddev.cpu().numpy()
        print("\nDone")
        if self.calculate_sd:
            return (mean, sd)
        return mean

    def run(self):
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
    X:  c x  N x M x L or c x N x M ndarray
            Grid indices.
            c is equal to the number of coordinate dimensions.
            For example, for xyz coordinates, c = 3.
        y: N x M x L or N x M ndarray
            Observations (data points)
        kernel: str
            gpytorch kernel object
        likelihood: gpytorch likelihood object
            Gaussian likelihood
        input_dim: int
            number of input dimensions
            (equal to number of feature vector columns)
        grid_points_ratio: float
            ratio of inducing points to overall points
    """

    def __init__(self, X, y, kernel, likelihood,
                 input_dim=3, grid_points_ratio=1.):
        super(skgprmodel, self).__init__(X, y, likelihood)
        grid_size = gpytorch.utils.grid.choose_grid_size(
            X, ratio=grid_points_ratio)
        self.mean_module = gpytorch.means.ConstantMean()
        scaled_kernel = gpytorch.kernels.ScaleKernel(kernel)
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            scaled_kernel, grid_size=grid_size, num_dims=input_dim)

    def forward(self, x):
        """Forward path"""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def get_kernel(kernel_type, input_dim, on_gpu=True, **kwargs):
    """
    Initalizes one of the following kernels: RBF, Matern
    Args:
        kernel_type: str
            kernel type ('RBF', Matern52')
        input_dim: int
            number of input dimensions
            (equal to number of feature vector columns)
        on_gpu: bool
            sets default tensor type to torch.cuda.DoubleTensor
    **Kwargs:
        lengthscale: list of two lists
            determines lower (1st list) and upper (2nd list) bounds
            for kernel lengthscale(s);
            number of elements in each list is equal to the input dimensions
        lengthscale_init: list with float(s)
            initializes lenghtscale at this value
    Returns:
        kernel object
    """
    if on_gpu and torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)

    lscale = kwargs.get('lengthscale')
    if lscale is not None:
        assert isinstance(lscale, list)
        assert isinstance(lscale[0], list) and isinstance(lscale[1], list)
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
