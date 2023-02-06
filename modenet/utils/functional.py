"""
Utility functions
"""


def unsqueeze_to_dimension(tensor, dim):
    in_dim = tensor.dim()
    if in_dim - dim == 0:
        return tensor
    elif in_dim - dim < 0:
        for i in range(abs(in_dim - dim)):
            tensor = tensor.unsqueeze(0)
        return tensor
    elif in_dim - dim > 0:
        for i in range(in_dim - dim):
            tensor = tensor.squeeze(0)

    return tensor