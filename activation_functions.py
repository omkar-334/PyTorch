import torch
import torch.nn as nn


class ActivationFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.config = {"name": self.name}


class Sigmoid(ActivationFunction):
    def forward(self, x):
        return 1 / (1 + torch.exp(-x))


class Tanh(ActivationFunction):
    def forward(self, x):
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))


class ReLU(ActivationFunction):
    def forward(self, x):
        return max(0, x)


class LeakyReLU(ActivationFunction):
    def forward(self, x):
        return max(0.1 * x, x)


class ELU(ActivationFunction):
    def forward(self, x):
        if x < 0:
            return torch.exp(x) - 1
        else:
            return x


class Swish(ActivationFunction):
    def forward(self, x):
        return x * torch.sigmoid(x)
