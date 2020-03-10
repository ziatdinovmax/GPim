# GP-based reconstruction of 2D images and 3D spectroscopic data
# Author: Maxim Ziatdinov (email: maxim.ziatdinov@ai4microcopy.com)

# imports
import argparse
import os
import numpy as np
from gpim import gpr, gprutils
import torch
torch.set_default_tensor_type(torch.DoubleTensor)

# filepath and GP regression parameters
parser = argparse.ArgumentParser("Gaussian processes for sparse 3D data")
parser.add_argument("FILEPATH", nargs="?", type=str,
                    help="provide 3D numpy array of spectroscopic data")
parser.add_argument("--KERNEL", nargs="?", default="Matern52", type=str)
parser.add_argument("--LENGTH_CONSTR_MIN", nargs="?", default=1, type=int)
parser.add_argument("--LENGTH_CONSTR_MAX", nargs="?", default=20, type=int)
parser.add_argument("--LEARNING_RATE", nargs="?", default=0.05, type=float)
parser.add_argument("--INDUCING_POINTS", nargs="?", default=1500, type=int)
parser.add_argument("--NORMALIZE", nargs="?", default=1, type=int,
                    help="Normalizes to [0, 1]. 1 is True, 0 is False")
parser.add_argument("--STEPS", nargs="?", default=1000, type=int)
parser.add_argument("--PROB", nargs="?", default=0.0, type=float,
                    help="Value between 0 and 1." +
                    "Controls number of data points to be removed.")
parser.add_argument("--USE_GPU", nargs="?", default=1, type=int,
                    help="1 for using GPU, 0 for running on CPU")
parser.add_argument("--SAVEDIR", nargs="?", default="Output", type=str,
                    help="directory to save outputs")

args = parser.parse_args()

# Load data (e.g. N x M image or N x M x L spectroscopic grid)
R_true = np.load(args.FILEPATH)
if args.NORMALIZE and np.isnan(R_true).any() is False:
    R_true = (R_true - np.amin(R_true))/np.ptp(R_true)
# Get "ground truth" grid indices
X_true = gprutils.get_full_grid(R_true, dense_x=1.)
# Construct lengthscale constraints for all dimensions
LENGTH_CONSTR = [
                 [float(args.LENGTH_CONSTR_MIN) for i in range(np.ndim(R_true))],
                 [float(args.LENGTH_CONSTR_MAX) for i in range(np.ndim(R_true))]
]
# Corrupt data (if args.PROB > 0)
X, R = gprutils.corrupt_data_xy(X_true, R_true, args.PROB)
# Directory to save results
if not os.path.exists(args.SAVEDIR):
    os.makedirs(args.SAVEDIR)
# Reconstruct the corrupt data. Initalize our "reconstructor" first.
reconstr = gpr.reconstructor(
    X, R, X_true, args.KERNEL, LENGTH_CONSTR, args.INDUCING_POINTS,
    args.LEARNING_RATE, args.STEPS, args.USE_GPU, verbose=True)
# Model training and prediction
mean, sd, hyperparams = reconstr.run()
# Save results
np.savez(os.path.join(args.SAVEDIR, os.path.basename(
    os.path.splitext(args.FILEPATH)[0])+'-gpr_reconstruction.npz'),
    original_data=R_true, input_data=R, mean=mean, SD=sd,
    hyperparams=hyperparams)
