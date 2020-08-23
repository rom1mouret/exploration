#!/usr/bin/env python3

from collections import defaultdict
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from bool_net import NaiveBoolNet, HyperBoolNet

# specifications of the experiment
perf = defaultdict(list)  # result
loss_func = nn.BCEWithLogitsLoss()
min_width = 1
max_width = 22
n_rounds = 50

def make_type1(width: int) -> nn.Module:
    return NaiveBoolNet(width=width)

def make_type2(width: int) -> nn.Module:
    return HyperBoolNet()

# try with different architectures
for net_factory in (make_type1, make_type2):
    # parameterize the architecture with different widths
    for width in tqdm(range(min_width, max_width+1), total=max_width-min_width):
        f1 = []
        # collect F1-scores over multiple rounds in order to compute statistics
        for _ in range(n_rounds):
            net = net_factory(width)
            optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
            for epoch in range(500):
                x = torch.randn(16, 2)
                seed = torch.randn(2)
                y_true = ((x[:, 0] > 0) & (x[:, 1] > 0)).float()
                if y_true.sum() < y_true.size(0) / 4:
                    continue  # not good for training
                y_pred = net(x, seed).squeeze()
                loss = loss_func(y_pred, y_true)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # evaluation
            with torch.no_grad():
                x = torch.randn(5000, 2)
                seed = torch.zeros(2)
                y_true = ((x[:, 0] > 0) & (x[:, 1] > 0)).float()
                y_pred = (net(x, seed).squeeze() > 0).float()
                score = f1_score(y_true.numpy(), y_pred.numpy(), average='macro')
                f1.append(score)

        perf[str(net)].append((net.width(), np.mean(f1)))


# reporting
for model_name, accuracy in perf.items():
    x, y = zip(*accuracy)
    if model_name == "HyperNetwork":
        plt.scatter(x, y, label=model_name, s=100, alpha=0.2, c='green')
    else:
        plt.plot(x, y, label=model_name)

plt.scatter([2], [0.98], s=100, alpha=1.00, c='red', label="hardcoded", marker="x")
plt.ylabel("F1 score")
plt.xlabel("width of the trained network")
plt.title(r'$y = x_1 > 0 \wedge x_2 > 0$')
plt.legend()
plt.savefig("bool_experiment.png", dpi=70)
