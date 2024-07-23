import torch.nn as nn


class FullyConnected(nn.Sequential):
    """
    Fully connected multi-layer network with ELU activations.
    """

    def __init__(self, sizes, final_activation=None, middle_activation=nn.ELU):
        layers = []
        layers.append(nn.Linear(sizes[0], sizes[1]))
        for in_size, out_size in zip(sizes[1:], sizes[2:]):
            layers.append(middle_activation())
            layers.append(nn.Linear(in_size, out_size))
        if final_activation is not None:
            layers.append(final_activation)
        self.length = len(layers)
        super().__init__(*layers)

    def append(self, module):
        assert isinstance(module, nn.Module)
        self.add_module(str(len(self)), module)

    def __len__(self):
        return self.length
