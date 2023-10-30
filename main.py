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
    args = parser.parse_args()

    # TODO: Fix stupid appending "/data*.mat"
    files = glob.glob(args.input_folder + "/data*.mat")
    files = sorted(files)
    for idx, path in enumerate(files):
        mat = sp.io.loadmat(path)


if __name__ == "__main__":
    main()
