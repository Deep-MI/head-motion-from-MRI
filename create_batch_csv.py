import sys
import os
from argparse import ArgumentParser
from math import ceil

def create_batch_csv(files, output, chunk_size=100):
    os.makedirs("batch", exist_ok=True)
    n_chunks = ceil(len(files) / chunk_size)
    print(f"Creating {n_chunks} chunks of {chunk_size} files each ({len(files)} total files)")
    cmd = []
    for i in range(1, n_chunks+1):
        chunk_path = f"{output.replace('.csv', '')}_{i}.csv"
        with open(chunk_path, "w") as f:
            for file in files[(i-1)*chunk_size:i*chunk_size]:
                f.write(f"{os.path.realpath(file)}\n")
        cmd.append(f"bash run_prediction.sh -i {chunk_path} -o reports/{os.path.basename(chunk_path)} -t $TYPE")
    return cmd

def main():
    parser = ArgumentParser(
        description="Create submission script for a list of files, eg. `python create_batch_csv.py */T1* -o batch_T1.csv`"
    )
    parser.add_argument("-o", "--output", default="batch.csv", help="Output CSV batch file")
    parser.add_argument("-c", "--chunk_size", default=100, type=int, help="Number of files per batch")
    parser.add_argument("files", nargs="*", help="List of files to process")

    args = parser.parse_args()
    if not args.files:
        parser.error("No files to process")

    print(f"Creating batch with {len(args.files)} files at {args.output}")
    cmd = create_batch_csv(args.files, os.path.join("batch", args.output), args.chunk_size)

    seq = "T1"
    if "T2" in args.files[0]:
        seq = "T2"
    elif "FLAIR" in args.files[0]:
        seq = "FLAIR"

    print("You might want to run the following commands to process all chunks:")
    print(f"export TYPE={seq}")
    print(" && ".join(cmd))

    # write script to batch/{seq}_run.sh
    with open(f"batch/{seq}_run.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"export TYPE={seq}\n")
        f.write(" && ".join(cmd))
    print(f"You can run thus as:\nsh batch/{seq}_run.sh")

if __name__ == "__main__":
    main()
