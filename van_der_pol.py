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
from lyapunov_learner_nnet import KnownLyapunovNet, LyapunovNetRegressor
from lyapunov_verifier import SMTVerifier
from nnet_utils import DEVICE, LyapunovNet, gen_equispace_regions


KNOWN_LYA = KnownLyapunovNet(
    W1=torch.tensor([
        [+ 2.0125248432159424, - 0.86828583478927612],
        [+ 0.26035198569297791, - 0.058430317789316177],
        [- 0.50644028186798096, + 1.4162489175796509],
        [+ 2.5682404041290283, - 1.2206004858016968],
        [+ 0.83175277709960938, + 1.0546237230300903],
        [- 0.2350536435842514, + 0.075554989278316498]
    ], device=DEVICE),
    b1=torch.tensor([
        -1.6187947988510132,
        -1.0764049291610718,
        -0.96464759111404419,
        -0.82901746034622192,
        0.98988056182861328,
        1.1398535966873169], device=DEVICE),
    W2=torch.tensor([[
        0.75732171535491943,
        -1.6154271364212036,
        1.2375599145889282,
        0.41873458027839661,
        - 0.89795422554016113,
        1.0879759788513184]], device=DEVICE),
    b2=torch.tensor(0.51641756296157837, device=DEVICE)
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

    x_part = (1, 2)  # Partition into subspaces
    assert len(x_part) == X_DIM
    print(
        f"Prepare {'x'.join(str(n) for n in x_part)} equispaced training samples: ",
        end="", flush=True)
    t_start = time.perf_counter()
    x_regions = gen_equispace_regions(x_part, X_ROI)
    print(f"{time.perf_counter() - t_start:.3f}s")

    if True:
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
        max_epochs=20, max_iter_learn=1000)

    x_values_np, x_lbs_np, x_ubs_np = x_regions_np[:, 0], x_regions_np[:, 1], x_regions_np[:, 2]

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
