import argparse
import configparser
import glob

import numpy as np
import scipy as sp

from ktc.data import Data, Reference

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

    # Load the reference data
    refmat = sp.io.loadmat(args.input_folder + args.reference)
    reference = Reference(refmat)

    files = glob.glob(args.input_folder + args.pattern)
    for idx, path in enumerate(files):
        inmat = sp.io.loadmat(path)
        data = Data(inmat)

        # TODO: Reconstruct from data

        # TODO: Use regex to exact data_x.math
        outmat = inmat
        sp.io.savemat(args.output_folder + ("/%d.mat" % (idx + 1)), outmat)


if __name__ == "__main__":
    main()
