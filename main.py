import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("output_folder")
    parser.add_argument("category", type=int)
    args = parser.parse_args()


if __name__ == "__main__":
    main()
