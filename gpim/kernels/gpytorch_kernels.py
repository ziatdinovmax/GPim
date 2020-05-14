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
            Kernel type ('RBF', Matern52', 'Spectral)
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
        **n_mixtures (int):
            number of mixtures for spectral mixture kernel
        **precision (str):
            Choose between single ('single') and double ('double') precision
    Returns:
        kernel object
    """
    precision = kwargs.get("precision", "double")
    if precision == 'single':
        tensor_type = torch.FloatTensor
        tensor_type_gpu = torch.cuda.FloatTensor
    else:
        tensor_type = torch.DoubleTensor
        tensor_type_gpu = torch.cuda.DoubleTensor

    if on_gpu and torch.cuda.is_available():
        torch.set_default_tensor_type(tensor_type_gpu)
    else:
        torch.set_default_tensor_type(tensor_type)

    lscale = kwargs.get('lengthscale')
    isotropic = kwargs.get("isotropic")
    nmix = kwargs.get("n_mixtures")
    if kernel_type == "Spectral" and nmix is None:
        nmix = 4
    if lscale is not None:
        lscale = gpytorch.constraints.Interval(torch.tensor(lscale[0]),
                                               torch.tensor(lscale[1]))
    input_dim = 1 if isotropic else input_dim

    kernel_book = lambda input_dim, lscale, **kwargs: {
        'RBF': gpytorch.kernels.RBFKernel(
            ard_num_dims=input_dim,
            lengthscale_constraint=lscale
            ),
        'Matern52': gpytorch.kernels.MaternKernel(
            ard_num_dims=input_dim,
            lengthscale_constraint=lscale
            ),
        'Spectral': gpytorch.kernels.SpectralMixtureKernel(
            ard_num_dims=input_dim,
            num_mixtures=kwargs.get("nmix")
        )
    }
    try:
        kernel = kernel_book(input_dim, lscale, nmix=nmix)[kernel_type]
    except KeyError:
        print('Select one of the currently available kernels:',\
              '"RBF", "Matern52", "Spectral"')
        raise
    return kernel
