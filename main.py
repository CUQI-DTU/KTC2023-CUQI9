import argparse
import glob

import numpy as np
import scipy as sp

electrode_count = 32
contact_impedance = np.full(electrode_count, 1e-6)
background_conductivity = 0.8


def main():
    # TODO: Add help text
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("output_folder")
    parser.add_argument("category", type=int)
    parser.add_argument("-p", "--pattern", default="/data*.mat")
    parser.add_argument("-r", "--reference", default="/ref.mat")
    args = parser.parse_args()

    # TODO: Fix stupid appending "/data*.mat"
    files = glob.glob(args.input_folder + args.pattern)
    for idx, path in enumerate(files):
        input = sp.io.loadmat(path)

        # TODO: Reconstruct from data

        # TODO: Use regex to exact data_x.math
        output = input
        sp.io.savemat(args.output_folder + ("/%d.mat" % (idx + 1)), output)


if __name__ == "__main__":
    main()
