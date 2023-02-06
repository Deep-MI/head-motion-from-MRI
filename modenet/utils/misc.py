"""
Stuff that doesnt fit anywhere else
"""

import json
from argparse import Namespace


class EmptyContextManager:
    '''
    empty environment to be able to substitute torch.no_grad()
    '''

    def __enter__(self):
        pass
    def __exit__(self, type, value, traceback):
        pass


def load_parameters(json_file, print_json=True):
    with open(json_file, "r") as read_file:
        args = json.load(read_file)

        if print_json:
            print("Configuration:")
            for key, value in args.items():
                print(key, ":", value)

        # convert the json into a namespace - this allows to use a json file and commandline input parameters interchangable
        args = Namespace(**args)
        args.start_epoch = 0  # set the starting epoch to zero - this is replaced if we load a previous checkpoint

    return args