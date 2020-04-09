'''
pyro_kernels.py
======
Pyro kernels
(some customized kernels TBA)
'''

import pyro.contrib.gp as gp
import pyro.distributions as dist
import torch
import warnings


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
