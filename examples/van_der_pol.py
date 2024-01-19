#!/usr/bin/env python
# coding: utf-8
"""
Learning Dyanmics and Lyapunov function for the Van der Pol oscillator
"""

from dreal import Variable  # type: ignore
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
import time

from plot_utils_2d import add_level_sets, add_valid_regions
from pricely.cegus_lyapunov import cegus_lyapunov
from pricely.learner_cvxpy import QuadraticLearner, SOS1Learner
from pricely.utils import check_exact_lyapunov, gen_equispace_regions
from pricely.verifier_dreal import SMTVerifier


X_DIM = 2
X_ROI = np.array([
    [-1.25, -1.25],  # Lower bounds
    [1.25, 1.25]  # Upper bounds
])
assert X_ROI.shape == (2, X_DIM)
LIP_BB = 3.4599  # Manually derived for ROI
NORM_LB, NORM_UB = 0.2, 1.2


def f_bbox(x: np.ndarray):
    # Actual dynamical system
    assert x.shape[1] == X_DIM
    dxdt = np.zeros_like(x)
    dxdt[:, 0] = -x[:, 1]
    dxdt[:, 1] = x[:, 0] + (x[:, 0]**2-1)*x[:, 1]
    return dxdt


def validate(lya):
    x_vars = [Variable(f"x{i}") for i in range(X_DIM)]
    dxdt_exprs = [
        -x_vars[1],
        x_vars[0] + (x_vars[0]**2-1)*x_vars[1]
    ]
    return check_exact_lyapunov(
        x_vars, dxdt_exprs, X_ROI,
        lya.lya_expr(x_vars), NORM_LB, NORM_UB)


def main():
    x_part = [5]*X_DIM  # Partition into subspaces
    assert len(x_part) == X_DIM
    print(
        f"Prepare {'x'.join(str(n) for n in x_part)} equispaced training samples.")
    x_regions = gen_equispace_regions(x_part, X_ROI)

    lya = QuadraticLearner(X_DIM)
    verifier = SMTVerifier(
        x_roi=X_ROI, norm_lb=NORM_LB, norm_ub=NORM_UB)

    t_start = time.perf_counter()
    last_epoch, last_x_regions, cex_regions = cegus_lyapunov(
        lya, verifier, x_regions, f_bbox, LIP_BB,
        max_epochs=25, max_iter_learn=1)
    time_usage = time.perf_counter() - t_start
    print(f"Total Time: {time_usage:.3f}s")

    # Validate with exact Lyapunov conditions
    result = validate(lya)
    if result is None:
        print("Learned candidate is a valid Lyapunov function.")
    else:
        print("Learned candidate is NOT a Lyapunov function.")
        print(f"Counterexample:\n{result}")

    print("Plotting verified regions:")
    add_valid_regions(
        plt.gca(), last_epoch, time_usage, last_x_regions, cex_regions)
    plt.gca().add_patch(Circle((0, 0), NORM_LB, color='gray', fill=False))
    plt.gca().add_patch(Circle((0, 0), NORM_UB, color='gray', fill=False))

    add_level_sets(plt.gca(), lya.lya_values, X_ROI, NORM_UB)

    plt.gca().set_xlim(*X_ROI[:, 0])
    plt.gca().set_ylim(*X_ROI[:, 1])
    plt.gca().set_aspect("equal")
    plt.savefig(f"out/VanDerPol-valid_regions-{'x'.join(str(n) for n in x_part)}-{lya.__class__.__name__}.png")
    plt.clf()


if __name__ == "__main__":
    main()
