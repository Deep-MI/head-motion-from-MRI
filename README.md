# Motion Detection Network

This repository contains code used in training and evaluating a neural network to directly estimate motion during an MRI scan from MR images. Published in the proceedings of ISBI 2023. https://arxiv.org/abs/2302.14490


The executables are located in /scripts


## Build docker

The commands in this assume that you run the network training/evaluation in a docker container.
To create this container execute

```docker build -f docker -t pytorch_opencv:regression_pyt1.11.0 .```

If you want to run things locally, omit the docker directives and directly execute the python scripts (command starting with "python3").


## Training

In the first lines of the "train.py" script some paths are resolved to find a two files:

1. A text/csv file containing only paths to MRI images with each path being in a new line. This is the input data to the network. Each image is expected to be in a unique folder. (e.g. /my/path/e25122/T1.nii.gz)
2. A csv file matching the image folders to numbers (which represent the motion levels in our case - e.g.  e25122, 1.53)

After setting up these files and the paths to find them, in the parameter.json and train.py you can launch the training with:

```python3 scripts/train.py```

or (recommended) build the docker container (in the "docker" folder) to install all required packages and then launch the training with

```docker run -it --rm --gpus \"device=0\" -u $(id -u) -v /etc/localtime:/etc/localtime:ro --ipc=host -v $PWD:/workspace user/pytorch_opencv:regression_pyt1.11.0 python3 /workspace/scripts/train.py```

Output will be written to a specified output folder, ./tensorboard_outputs and to the console

To adjust network parameters take a look at the files in the dataset specific script folders like.

./scripts/train.py \
./scripts/parameters.json


## Download weights

Take care that this network was only trained on data of the Rhineland Study and is unlikely to generalize to different MR sequences and populations.

Weights can be downloaded at https://zenodo.org/record/7940494 .


## Evaluation

For evaluation edit ./scripts/eval.py to point to the correct path / and input file and then run the docker with

```docker run -it --rm --gpus \"device=0\" -u $(id -u) -v /etc/localtime:/etc/localtime:ro --ipc=host -v $PWD:/workspace pollakc/pytorch_opencv:regression_pyt1.11.0 python3 /workspace/scripts/eval.py```


## More

General learning functionalities are located in the ./modenet folder


