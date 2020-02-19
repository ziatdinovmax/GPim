# Plotting GP reconstruction results
# Author: Maxim Ziatdinov (email: maxim.ziatdinov@ai4microcopy.com)

import numpy as np
import argparse
from gpim import gprutils

parser = argparse.ArgumentParser(
    usage="Plot the results of GP reconstruction" +
    "for 2D images and 3D spectroscopic data")
parser.add_argument("FILEPATH", nargs="?", type=str)
parser.add_argument("--SAVEDIR", nargs="?", default="Output", type=str)
args = parser.parse_args()

dataset = np.load(args.FILEPATH)
R = dataset["input_data"]
R_true = dataset["original_data"]
mean = dataset["mean"]
sd = dataset["SD"]

if np.ndim(R) == 2:
    gprutils.plot_reconstructed_data2d(
        R, mean, save_fig=True,
        savedir=args.SAVEDIR, filepath=args.FILEPATH)
if np.ndim(R) == 3:
    slice_number = int(input(
        "Enter a slice number between 0 and {}: ".format(R.shape[-1])))
    pos = np.array(input("Enter xy positions for spectroscopic curves " +
                   "to display (x1 y1 x2 y2 ...): ").split(), dtype=np.int)
    pos_x = pos[0::2]
    pos_y = pos[1::2]
    assert len(pos_x) == len(pos_y), "Enter the positions as x1 y1 x2 y2 ..."
    pos_xy = [[pos_x[i], pos_y[i]] for i in range(len(pos_x))]
    gprutils.plot_reconstructed_data3d(
        R, mean, sd, slice_number, pos_xy,
        spec_window=2, save_fig=True,
        savedir=args.SAVEDIR, filepath=args.FILEPATH)
