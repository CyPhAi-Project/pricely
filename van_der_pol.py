#!/usr/bin/env python
# coding: utf-8
"""
Learning Dyanmics and Lyapunov function for the Van der Pol oscillator
"""

from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
import time
import torch

from cegus_lyapunov import cegus_lyapunov
from lyapunov_learner_cvx import QuadraticLearner
from lyapunov_learner_nnet import KnownLyapunovNet, LyapunovNetRegressor
from lyapunov_verifier import SMTVerifier
from nnet_utils import DEVICE, LyapunovNet, gen_equispace_regions


KNOWN_LYA = KnownLyapunovNet(
    W1=torch.tensor([
        [2.0125248432159423828125, -0.868285834789276123046875],
        [0.2603519856929779052734375, -0.0584303177893161773681640625],
        [-0.50644028186798095703125, 1.41624891757965087890625],
        [2.5682404041290283203125, -1.22060048580169677734375],
        [0.831752777099609375, 1.05462372303009033203125],
        [-0.23505364358425140380859375, 0.075554989278316497802734375]
    ], dtype=torch.float32, device=DEVICE),
    b1=torch.tensor([
        -1.61879479885101318359375,
        -1.07640492916107177734375,
        -0.964647591114044189453125,
        -0.829017460346221923828125,
        0.98988056182861328125,
        1.13985359668731689453125], dtype=torch.float32, device=DEVICE),
    W2=torch.tensor([[
        0.75732171535491943359375,
        -1.61542713642120361328125,
        1.23755991458892822265625,
        0.4187345802783966064453125,
        -0.8979542255401611328125,
        1.087975978851318359375]], dtype=torch.float32, device=DEVICE),
    b2=torch.tensor(
        0.516417562961578369140625,
        dtype=torch.float32, device=DEVICE)
)


def main():
    # Actual dynamical system
    X_DIM = 2
    X_ROI = torch.Tensor([
        [-1.5, -1.5],  # Lower bounds
        [1.5, 1.5]  # Upper bounds
    ])
    assert X_ROI.shape == (2, X_DIM)
    U_DIM = 0

    def f_bbox(x: torch.Tensor):
        assert x.shape[1] == X_DIM
        dxdt = torch.zeros_like(x)
        dxdt[:, 0] = -x[:, 1]
        dxdt[:, 1] = x[:, 0] + (x[:, 0]**2-1)*x[:, 1]
        return dxdt
    LIP_BB = 3.4599  # Manually derived for ROI

    x_part = (2, 2)  # Partition into subspaces
    assert len(x_part) == X_DIM
    print(
        f"Prepare {'x'.join(str(n) for n in x_part)} equispaced training samples: ",
        end="", flush=True)
    t_start = time.perf_counter()
    x_regions = gen_equispace_regions(x_part, X_ROI)
    print(f"{time.perf_counter() - t_start:.3f}s")

    if True:
        lya = QuadraticLearner(X_DIM)
    elif True:
        lya = KNOWN_LYA
    else:
        lya_net = LyapunovNet(n_input=X_DIM, n_hidden=6)
        lya = LyapunovNetRegressor(
            module=lya_net,
            optimizer=torch.optim.Adam(lya_net.parameters(), lr=10E-5)
        )

    verifier = SMTVerifier(
        x_roi=X_ROI.cpu().numpy(),
        norm_lb=0.2, norm_ub=1.2)

    x_regions_np, cex_regions = cegus_lyapunov(
        lya, verifier, x_regions, f_bbox, LIP_BB,
        max_epochs=30, max_iter_learn=1)

    x_values_np, x_lbs_np, x_ubs_np = \
        x_regions_np[:, 0], x_regions_np[:, 1], x_regions_np[:, 2]

    num_samples = len(x_values_np)
    assert X_ROI.shape[1] == 2
    sat_region_iter = (k for k, _ in cex_regions)
    k = next(sat_region_iter, None)
    for j in range(num_samples):
        if j == k:
            k = next(sat_region_iter, None)
            facecolor = "white"
        elif j >= num_samples - len(cex_regions):
            facecolor = "gray"
        else:
            facecolor = "green"
        w, h = x_ubs_np[j] - x_lbs_np[j]
        rect = Rectangle(x_lbs_np[j], w, h, fill=True,
                         edgecolor='black', facecolor=facecolor, alpha=0.3)
        plt.gca().add_patch(rect)

    plt.gca().add_patch(Circle((0, 0), 0.2, color='r', fill=False))
    plt.gca().add_patch(Circle((0, 0), 1.2, color='r', fill=False))
    plt.gca().set_xlim(*X_ROI[:, 0])
    plt.gca().set_ylim(*X_ROI[:, 1])
    plt.gca().set_aspect("equal")
    plt.savefig("out/VanDerPol_valid_regions.png")
    plt.clf()


if __name__ == "__main__":
    main()
