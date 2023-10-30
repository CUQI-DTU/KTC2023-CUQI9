import argparse


def main():
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
