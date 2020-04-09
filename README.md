[![Build Status](https://travis-ci.com/ziatdinovmax/atomai.svg?branch=master)](https://travis-ci.com/ziatdinovmax/atomai)
[![Documentation Status](https://readthedocs.org/projects/gpim/badge/?version=latest)](https://gpim.readthedocs.io/en/latest/?badge=latest)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/07ee1606a88b48d1bc46453f3ae1b1c8)](https://app.codacy.com/manual/ziatdinovmax/GPim?utm_source=github.com&utm_medium=referral&utm_content=ziatdinovmax/GPim&utm_campaign=Badge_Grade_Dashboard)


[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ziatdinovmax/GPim/blob/master/examples/notebooks/Quickstart_GPim.ipynb)
[![Gitpod ready-to-code](https://img.shields.io/badge/Gitpod-ready--to--code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/ziatdinovmax/GPim)
# GPim

**Under active development (expect some breaking changes)**

## What is GPim?

GPim is a python package that provides a systematic and easy way to apply Gaussian processes (GP) 
to images and hyperspectral data in [Pyro](https://pyro.ai/) and [Gpytorch](https://gpytorch.ai/) frameworks
(without a need to learn those frameworks).

For the examples, see our papers:

GP for 3D hyperspectral data: https://arxiv.org/abs/1911.11348

GP for 4D hyperspectral data: https://arxiv.org/abs/2002.03591

*The intended audience are domain scientists (for example, microscopists) with a basic knowledge of python.*
 
<p align="center">
  <img src="misc/GPim_illustration_v2.png" width="75%" title="GPim">
<p align="justify">

## Installation

To use it, first run:

```bash
pip install git+https://github.com/ziatdinovmax/GPim.git
```


## How to use

### General usage

Below is a simple example of applying GPim to reconstructing a sparse 2D image. It can be similarly applied to 3D and 4D hyperspectral data. The missing data points in sparse data must be represented as [NaNs](https://docs.scipy.org/doc/numpy/reference/constants.html?highlight=numpy%20nan#numpy.nan). In the absense of missing observation GPim can be used for image and spectroscopic data cleaning/smoothing in all the dimensions simultaneously, as well as for the resolution enhancement. Finally, when performing measurements, one can use the information about uncertainty in GP reconstruction to select the next measurement point (more details in the notebooks referenced below).

```python
import gpim
import numpy as np

# # Load dataset
R = np.load('sparse_exp_data.npy') 

# Get full (ideal) grid indices
X_full = gpim.utils.get_full_grid(R, dense_x=1)
# Get sparse grid indices
X_sparse = gpim.utils.get_sparse_grid(R)
# Kernel lengthscale constraints (optional)
lmin, lmax = 1., 4.
lscale = [[lmin, lmin], [lmax, lmax]] 

# Run GP reconstruction to obtain mean prediction and uncertainty for each predictied point
mean, sd, hyperparams = gpim.reconstructor(
    X_sparse, R, X_full, lengthscale=lscale,
    learning_rate=0.1, iterations=250, 
    use_gpu=True, verbose=False).run()

# Plot reconstruction results
gpim.utils.plot_reconstructed_data2d(R, mean, cmap='jet')
# Plot evolution of kernel hyperparameters during training
gpim.utils.plot_kernel_hyperparams(hyperparams)
```

### Running GPim notebooks in the cloud

1) Executable Google Colab [notebook](https://colab.research.google.com/github/ziatdinovmax/GPim/blob/dev/examples/notebooks/GP_2D3D_images.ipynb) with the example of applying GP to sparse spiral 2D scans in piezoresponse force microscopy (PFM) and hyperspectral 3D data in Band Excitation PFM.
2) Executable Google Colab [notebook](https://colab.research.google.com/github/ziatdinovmax/GPim/blob/master/examples/notebooks/GP_TD_cKPFM.ipynb) with the example of applying GP to 4D spectroscopic dataset for smoothing and resolution enhancement in contact Kelvin Probe Force Microscopy (cKPFM)
3) Simple example of performing GP-based exploration-exploitation on a toy dataset- TBA 

## Requirements

It is strongly recommended to run the codes with a GPU hardware accelerator (such as NVIDIA's P100 or V100 GPU). If you don't have a GPU on your local machine, you may rent a cloud GPU from [Google Cloud AI Platform](https://cloud.google.com/deep-learning-vm/). Running the [example notebook](https://colab.research.google.com/github/ziatdinovmax/GP/blob/master/notebooks/GP_BEPFM.ipynb) one time from top to bottom will cost about 1 USD with a standard deep learning VM instance (one P100 GPU and 15 GB of RAM).

## TODO

1) Add more test modules

2) Add more utility functions for 4D datasets

3) Add option to run GP on multiple GPUs

4) Add GP for image registration
