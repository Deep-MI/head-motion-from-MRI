#!/bin/bash

# hard coded solution for now
DATA_DIR="/groups/ag-reuter/projects/datasets/flair_synthesis/scans"

# commandline arguments

while getopts "i:o:t:" opt; do
    case $opt in
        i) input_file=$OPTARG ;;
        o) output_file=$OPTARG ;;
        t) type=$OPTARG ;;
        \?) echo "Invalid option: -$OPTARG" >&2 ;;
    esac
done

# Check if variables are set
if [ -z "$input_file" ] || [ -z "$output_file" ] || [ -z "$type" ]; then
    echo "Error: You must provide input file (-i), output file (-o), and type (-t)"
    exit 1
fi

echo $input_file
echo $output_file
echo $type

input_file=$(realpath $input_file)

# get absolute path of input file
output_file=$(realpath $output_file)

# get directory of output file
output_dir=$(dirname $output_file)
echo $output_dir

# make sure that input file exists
if [ ! -f $input_file ]; then
    echo "Error: Input file does not exist"
    exit 1
fi

docker run -it --rm --gpus all -u $(id -u) \
-v $output_dir:$output_dir \
-v /etc/localtime:/etc/localtime:ro \
-v "$DATA_DIR":"$DATA_DIR":ro \
-v $input_file:$input_file:ro --ipc=host \
-v $PWD:/workspace $USER/pytorch_opencv:regression_pyt1.11.0 /workspace/docker/run_prediction_docker.sh $input_file $output_file $type
