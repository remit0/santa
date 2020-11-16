import torch.nn as nn
import torch.nn.functional as F


class NeuralNet(nn.Module):

    """ Feed forward network. """

    def __init__(self, layers):
        super(NeuralNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x
