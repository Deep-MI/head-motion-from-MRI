import sys
import os
from argparse import ArgumentParser

def create_batch_csv(files, output):
    with open(output, "w") as f:
        for file in files:
            f.write(f"{os.path.realpath(file)}\n")

def main():
    parser = ArgumentParser(
        description="Create submission script for a list of files, eg. `python create_batch_csv.py */T1* -o batch_T1.csv`"
    )
    parser.add_argument("-o", "--output", default="batch.csv", help="Output CSV batch file")
    parser.add_argument("files", nargs="*", help="List of files to process")

    args = parser.parse_args()
    if not args.files:
        parser.error("No files to process")

    print(f"Creating batch with {len(args.files)} files at {args.output}")
    create_batch_csv(args.files, os.path.join("batch", args.output))

if __name__ == "__main__":
    main()
