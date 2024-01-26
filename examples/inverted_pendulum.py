from dreal import Config, sin as Sin, tanh as Tanh, Variable  # type: ignore
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import time

from plot_utils_2d import add_level_sets, add_valid_regions
from pricely.cegus_lyapunov import PLyapunovLearner, cegus_lyapunov
from pricely.learner_cvxpy import QuadraticLearner
from pricely.learner_mock import MockQuadraticLearner
from pricely.verifier_dreal import SMTVerifier
from pricely.utils import check_exact_lyapunov, gen_equispace_regions


def known_lya_values(x_values: np.ndarray) -> np.ndarray:
    W1_T = np.asfarray([
        [+0.03331, +0.03467, +2.12564, -0.39925, +0.12885, +0.95375],
        [-0.03113, -0.01892, +0.02354, -0.10678, -0.32245, +0.01298]
    ])
    b1 = np.asfarray([-0.48061, 0.88048, 0.86448, -0.87253, 0.81866, -0.26619])
    W2 = np.asfarray([
        [-0.33862, 0.65177, -0.52607, 0.23062, -0.04802, 0.66825]
    ])
    b2 = np.asfarray([0.22032])
    linear1 = x_values @ W1_T + b1
    hidden1 = np.tanh(linear1)
    linear2 = hidden1 @ W2.T + b2
    return np.tanh(linear2).squeeze()


G = 9.81  # m/s^2
M = 0.15  # kg
L = 0.5  # m

THETA_LIM = np.pi / 4

X_ROI = np.array([
    [-THETA_LIM, -4],  # Lower bounds
    [+THETA_LIM, +4]  # Upper bounds
])
X_DIM = 2
ABS_X_LB = 0.0625
LIP_BB = 33.214  # Manually derived for ROI


def ctrl(x: np.ndarray) -> np.ndarray:
    theta, omega = x[:, 0], x[:, 1]
    return 20.0*np.tanh(
        -23.28632*theta
        -5.27055*omega).reshape(len(x),)

def f_dyn(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    theta, omega = x[:, 0], x[:, 1]

    dxdt = np.zeros_like(x)
    dxdt[:, 0] = omega
    dxdt[:, 1] = (M*G*L*np.sin(theta) + u - 0.1*omega) / (M*L**2)
    return dxdt

def f_bbox(x: np.ndarray) -> np.ndarray:
    assert x.shape[1] == X_DIM
    return f_dyn(x, ctrl(x))


def validate(
        lya: PLyapunovLearner,
        level_ub: float,
        abs_x_lb: ArrayLike, abs_x_ub: ArrayLike,
        config: Config):
    # Validate with exact Lyapunov conditions
    x_vars = [Variable("θ"), Variable("ω")]
    u_expr = 20.0*Tanh(
            -23.28632*x_vars[0]
            -5.27055*x_vars[1])
    dxdt_exprs = [
        x_vars[1],
        (M*G*L*Sin(x_vars[0]) + u_expr - 0.1*x_vars[1]) / (M*L**2)
    ]
    lya_expr = lya.lya_expr(x_vars)
    result = check_exact_lyapunov(
        x_vars, dxdt_exprs,
        lya_expr, level_ub, abs_x_lb, abs_x_ub, config)


def main():
    x_part = [4]*X_DIM
    part = x_part
    x_regions = gen_equispace_regions(part, X_ROI)

    if False:
        lya = QuadraticLearner(X_DIM)
    else:
        b_mat = np.asfarray([
            [5.0, 0.0],
            [1.25, 0.25]
        ])
        pd_mat = b_mat.T @ b_mat
        lya = MockQuadraticLearner(pd_mat)
        _, abs_x_ub = lya.find_sublevel_set_and_box(X_ROI)
        x_range = np.row_stack((-abs_x_ub, abs_x_ub))
        x_regions = gen_equispace_regions(x_part, x_range)

    verifier = SMTVerifier(x_roi=X_ROI, abs_x_lb=ABS_X_LB)

    t_start = time.perf_counter()
    last_epoch, last_x_regions, cex_regions = \
        cegus_lyapunov(
            lya, verifier, x_regions,
            f_bbox, LIP_BB,
            max_epochs=30, max_iter_learn=1)
    time_usage = time.perf_counter() - t_start
    print(f"Total Time: {time_usage:.3f}s")

    level_ub, abs_x_ub = lya.find_sublevel_set_and_box(X_ROI)
    result = validate(lya, level_ub, ABS_X_LB, abs_x_ub, verifier._config)
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

    # Plot Level Set Comparison
    add_level_sets(plt.gca(), lya.lya_values, level_ub)

    plt.gca().set_aspect("equal")
    plt.savefig(f"out/InvertedPendulum-valid_regions-{'x'.join(str(n) for n in x_part)}-{lya.__class__.__name__}.png")
    plt.clf()


if __name__ == "__main__":
    main()
