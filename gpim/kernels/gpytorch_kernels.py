'''
gpytorch_kernels.py
======
Gpytorch kernels
(some customized kernels TBA)
'''

import gpytorch
import torch


def get_kernel(kernel_type, input_dim, on_gpu=True, **kwargs):
    """
    Initializes one of the following gpytorch kernels: RBF, Matern

    Args:
        kernel_type (str):
            Kernel type ('RBF', Matern52')
        input_dim (int):
            Number of input dimensions
            (translates into number of kernel dimensions unless isotropic=True)
        on_gpu (bool):
            Sets default tensor type to torch.cuda.DoubleTensor
        **lengthscale (list of two lists):
            Determines lower (1st list) and upper (2nd list) bounds
            for kernel lengthscale(s);
            number of elements in each list is equal to the input dimensions
        **isotropic (bool):
            one kernel lengthscale in all dimensions
    Returns:
        kernel object
    """
    if on_gpu and torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)

    lscale = kwargs.get('lengthscale')
    isotropic = kwargs.get("isotropic")
    if lscale is not None:
        lscale = gpytorch.constraints.Interval(torch.tensor(lscale[0]),
                                               torch.tensor(lscale[1]))
    input_dim = 1 if isotropic else input_dim
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
    return kernel