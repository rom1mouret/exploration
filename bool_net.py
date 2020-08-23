import torch
import torch.nn as nn


class NaiveBoolNet(nn.Module):
    def __init__(self, width: int) -> None:
        super(NaiveBoolNet, self).__init__()
        self._net = nn.Sequential(
            nn.Linear(2, width),
            nn.ReLU(),
            nn.Linear(width, 1)
        )

    def forward(self, x: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
        return self._net(x).squeeze(1)

    def __repr__(self) -> str:
        return "Naive"

    def width(self) -> int:
        return self._net[0].weight.size(0)


class HyperBoolNet(nn.Module):
    def __init__(self) -> None:
        super(HyperBoolNet, self).__init__()
        self._weight_maker1 = nn.Sequential(
            nn.Linear(2, 3),  # input: random seed
            nn.LeakyReLU(0.2),
            nn.Linear(3, 4)  # output: first layer
        )
        self._weight_maker2 = nn.Sequential(
            nn.Linear(2, 3),  # input: random seed
            nn.LeakyReLU(0.2),
            nn.Linear(3, 2)  # output: first layer
        )

        self._bias_maker = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
        seed = seed.resize(1, seed.size(0))
        layer1 = self._weight_maker1(seed).resize(2, 2)
        layer2 = self._weight_maker2(seed).resize(2, 1)
        result = (nn.ReLU()(x @ layer1) @ layer2) + self._bias_maker

        return result

    def width(self) -> int:
        return 2

    def __repr__(self) -> str:
        return "HyperNetwork"


def hardcoded(inp):
    a = nn.ReLU(-inp[:, 0])  # return 0 if inp positive, abs(inp) otherwise
    b = nn.ReLU(-inp[:, 1])
    return -100*a -100*b + 10
