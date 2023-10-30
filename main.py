import argparse
import configparser
import glob

import numpy as np
import scipy as sp

cfg = configparser.ConfigParser()
cfg.read("config.ini")

radius = cfg.getfloat("parameter", "radius")
electrode_count = cfg.getint("parameter", "electrode_count")
contact_impedance = cfg.getfloat("parameter", "contact_impedance")
background_conductivity = cfg.getfloat("parameter", "background_conductivity")

def main():

    # TODO: Add help text
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("output_folder")
    parser.add_argument("category", type=int)

    default = cfg["default"]
    parser.add_argument("-p", "--pattern", default=default["pattern"])
    parser.add_argument("-r", "--reference", default=default["reference"])

    args = parser.parse_args()

    # TODO: Load reference data

    files = glob.glob(args.input_folder + args.pattern)
    for idx, path in enumerate(files):
        input = sp.io.loadmat(path)

        # TODO: Reconstruct from data

        # TODO: Use regex to exact data_x.math
        output = input
        sp.io.savemat(args.output_folder + ("/%d.mat" % (idx + 1)), output)


if __name__ == "__main__":
    main()
