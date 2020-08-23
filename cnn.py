import random
import numpy as np
import torch
import torch.nn as nn


class NaiveCNN(nn.Module):
    def __init__(self, n_filters: int, rand_conv: bool=False) -> None:
        super(NaiveCNN, self).__init__()
        self._conv = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=4, bias=False)
        if rand_conv:
            self._conv.requires_grad_(False)
        self._decision = nn.Linear(n_filters, 1)

    def forward(self, x: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
        conved = self._conv(x).max(dim=2)[0]
        return self._decision(conved).squeeze(1)

    def __repr__(self) -> str:
        if self._conv.weight.requires_grad:
            return "Naive"
        else:
            return "RandomConv"

    def n_params(self) -> int:
        n = 0
        for param in self.parameters():
            if param.requires_grad:
                n += np.prod(param.size())

        return n


class HyperCNN(nn.Module):
    def __init__(self) -> None:
        super(HyperCNN, self).__init__()
        self._weight_maker = nn.Sequential(
            nn.Linear(2, 4),  # input: random seed
            nn.LeakyReLU(0.2),
            nn.Linear(4, 4)  # output: convolution kernel
        )
        self._bias_maker = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
        weight = self._weight_maker(seed.resize(1, seed.size(0))).unsqueeze(1)
        bias = self._bias_maker
        conved = nn.functional.conv1d(x, weight=weight)

        return conved.max(dim=2)[0][:, 0] + bias

    def __repr__(self) -> str:
        return "HyperNetwork"

    def n_params(self) -> int:
        return 5


def negative_example() -> torch.Tensor:
    noise = torch.rand(1, 1, 64) - 0.5

    return noise


def positive_example() -> torch.Tensor:
    noise = negative_example()
    i = random.randint(0, noise.size(2) - 5)
    noise[0, 0, [i, i+1, i+3]] = 1

    return noise


def example_pair() -> torch.Tensor:
    return torch.cat([
        negative_example(),
        positive_example()], dim=0)
