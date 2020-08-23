#!/usr/bin/env python3

from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from cnn import NaiveCNN, HyperCNN, example_pair

# validation batch
val_size = 1000
val_batch = torch.cat([
    example_pair() for _ in range(val_size)
], dim=0)
val_target = torch.Tensor(
    sum([[0, 1] for _ in range(val_size)], [])
)

# specifications of the experiment
training_target = torch.Tensor([0, 1])
loss_func = nn.BCEWithLogitsLoss()

def training_loss(y_pred: torch.Tensor) -> torch.Tensor:
    return loss_func(y_pred, training_target)

perf = defaultdict(list)  # result
min_filters = 1
max_filters = 7
n_rounds = 500

def make_type1(n_filters: int) -> nn.Module:
    return NaiveCNN(n_filters, rand_conv=False)

def make_type2(n_filters: int) -> nn.Module:
    return NaiveCNN(n_filters, rand_conv=True)

def make_type3(n_filters: int) -> nn.Module:
    return HyperCNN()

# try with different architectures
for net_factory in (make_type1, make_type2, make_type3):
    # parameterize the architecture with a different number of filters
    for n_filters in tqdm(range(min_filters, max_filters+1), total=max_filters-min_filters):
        accuracy = []
        # collect accuracy over multiple rounds in order to compute statistics
        for _ in range(n_rounds):
            net = net_factory(n_filters)
            optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.5)
            for epoch in range(100):
                seed = torch.randn(2)
                pred = net(example_pair(), seed)
                optimizer.zero_grad()
                training_loss(pred).backward()
                optimizer.step()
            # evaluation
            with torch.no_grad():
                seed = torch.zeros(2)
                pred = net(val_batch, seed)
                y = (pred > 0).float()
                acc = (y * val_target + (1 - y) * (1 - val_target)).mean().item()
                accuracy.append(acc)
        perf[str(net)].append((net.n_params(), 100 * np.mean(accuracy)))

# reporting
for model_name, accuracy in perf.items():
    x, y = zip(*accuracy)
    if model_name == "HyperNetwork":
        plt.scatter(x, y, label=model_name, s=100, alpha=0.2, c='green')
    else:
        plt.plot(x, y, label=model_name)

plt.scatter([5], [100], s=100, alpha=1.0, c='red', label="hardcoded", marker="x")
plt.ylabel("accuracy (%)")
plt.xlabel("number of parameters in the trained network")
plt.title("y = TEST([1, 1, 0, 1] in Sequence)")
plt.legend()
plt.savefig("cnn_experiment.png", dpi=70)
