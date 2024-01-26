#!/usr/bin/env python
# coding: utf-8
"""
Learning Dyanmics and Lyapunov function for the Van der Pol oscillator
"""

from dreal import Variable  # type: ignore
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import time

from plot_utils_2d import add_level_sets, add_valid_regions
from pricely.cegus_lyapunov import PLyapunovLearner, cegus_lyapunov
from pricely.learner_cvxpy import QuadraticLearner
from pricely.learner_mock import MockQuadraticLearner
from pricely.utils import check_exact_lyapunov, gen_equispace_regions
from pricely.verifier_dreal import SMTVerifier


X_DIM = 2
X_ROI = np.array([
    [-0.75, -0.75],  # Lower bounds
    [+0.75, +0.75]  # Upper bounds
])
assert X_ROI.shape == (2, X_DIM)
LIP_BB = 3.4599  # Manually derived for ROI
ABS_X_LB = 0.0625


def f_bbox(x: np.ndarray):
    # Actual dynamical system
    assert x.shape[1] == X_DIM
    dxdt = np.zeros_like(x)
    dxdt[:, 0] = -x[:, 1]
    dxdt[:, 1] = x[:, 0] + (x[:, 0]**2-1)*x[:, 1]
    return dxdt


def validate(lya: PLyapunovLearner, 
             level_ub: float, abs_x_lb: ArrayLike, abs_x_ub: ArrayLike):
    x_vars = [Variable(f"x{i}") for i in range(X_DIM)]
    dxdt_exprs = [
        -x_vars[1],
        x_vars[0] + (x_vars[0]**2-1)*x_vars[1]
    ]
    return check_exact_lyapunov(
        x_vars, dxdt_exprs,
        lya.lya_expr(x_vars), level_ub, abs_x_lb, abs_x_ub)


def main():
    x_part = [5]*X_DIM  # Partition into subspaces
    assert len(x_part) == X_DIM
    print(
        f"Prepare {'x'.join(str(n) for n in x_part)} equispaced training samples.")
    x_regions = gen_equispace_regions(x_part, X_ROI)

    if False:
        lya = QuadraticLearner(X_DIM)
    else:
        b_mat = np.asfarray([
            [1.0, 0.0],
            [-0.25, 1.0]
        ])
        pd_mat = b_mat.T @ b_mat
        lya = MockQuadraticLearner(pd_mat)
        _, abs_x_ub = lya.find_sublevel_set_and_box(X_ROI)
        x_range = np.row_stack((-abs_x_ub, abs_x_ub))
        x_regions = gen_equispace_regions(x_part, x_range)

    verifier = SMTVerifier(
        x_roi=X_ROI, abs_x_lb=ABS_X_LB)

    t_start = time.perf_counter()
    last_epoch, last_x_regions, cex_regions = cegus_lyapunov(
        lya, verifier, x_regions, f_bbox, LIP_BB,
        max_epochs=25, max_iter_learn=1)
    time_usage = time.perf_counter() - t_start
    print(f"Total Time: {time_usage:.3f}s")

    # Validate with exact Lyapunov conditions
    level_ub, abs_x_ub = lya.find_sublevel_set_and_box(X_ROI)
    result = validate(lya, level_ub, ABS_X_LB, abs_x_ub)
    if result is None:
        print("Learned candidate is a valid Lyapunov function.")
    else:
        print("Learned candidate is NOT a Lyapunov function.")
        print(f"Counterexample:\n{result}")

    print("Plotting verified regions:")
    plt.gca().set_xlim(-1.125*abs_x_ub[0], +1.125*abs_x_ub[0])
    plt.gca().set_ylim(-1.125*abs_x_ub[1], +1.125*abs_x_ub[1])

    add_valid_regions(
        plt.gca(), last_epoch, time_usage, last_x_regions, cex_regions)
    plt.gca().add_patch(Rectangle(
        (-ABS_X_LB/2, -ABS_X_LB/2), ABS_X_LB, ABS_X_LB, color='gray', fill=False))
    plt.gca().add_patch(Rectangle(
        (-abs_x_ub[0], -abs_x_ub[1]), 2*abs_x_ub[0], 2*abs_x_ub[1], color='gray', fill=False))
    plt.gca().add_patch(Rectangle(
        (X_ROI[0][0], X_ROI[0][1]), X_ROI[1][0]-X_ROI[0][0], X_ROI[1][1]-X_ROI[0][1], color='r', fill=False))

    add_level_sets(plt.gca(), lya.lya_values, level_ub)

    plt.gca().set_aspect("equal")
    plt.savefig(f"out/VanDerPol-valid_regions-{'x'.join(str(n) for n in x_part)}-{lya.__class__.__name__}.png")
    plt.clf()


if __name__ == "__main__":
    main()
