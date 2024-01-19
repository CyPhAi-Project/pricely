from dreal import sin as Sin, tanh as Tanh, Variable  # type: ignore
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
import time

from plot_utils_2d import add_level_sets, add_valid_regions
from pricely.cegus_lyapunov import cegus_lyapunov
from pricely.learner_cvxpy import QuadraticLearner, SOS1Learner
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

X_ROI = np.array([
    [-4, -4],  # Lower bounds
    [+4, +4]  # Upper bounds
])
X_DIM = 2
NORM_LB, NORM_UB = 0.1, 4.0
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


def validate(lya, config):
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
        x_vars, dxdt_exprs, X_ROI,
        lya_expr, NORM_LB, NORM_UB, config)


def main():
    x_part = [4]*X_DIM
    part = x_part
    x_regions = gen_equispace_regions(part, X_ROI)

    lya = SOS1Learner(X_DIM)
    verifier = SMTVerifier(
        x_roi=X_ROI,
        norm_lb=NORM_LB, norm_ub=NORM_UB)

    t_start = time.perf_counter()
    last_epoch, last_x_regions, cex_regions = \
        cegus_lyapunov(
            lya, verifier, x_regions,
            f_bbox, LIP_BB,
            max_epochs=30, max_iter_learn=1)
    time_usage = time.perf_counter() - t_start
    print(f"Total Time: {time_usage:.3f}s")

    result = validate(lya, verifier._config)
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

    # Plot Level Set Comparison
    add_level_sets(plt.gca(), lya.lya_values, X_ROI, NORM_UB)
    add_level_sets(plt.gca(), known_lya_values, X_ROI, NORM_UB, colors="y")

    plt.gca().set_xlim(*X_ROI[:, 0])
    plt.gca().set_ylim(*X_ROI[:, 1])
    plt.gca().set_aspect("equal")
    plt.savefig(f"out/InvertedPendulum-valid_regions-{'x'.join(str(n) for n in x_part)}-{lya.__class__.__name__}.png")
    plt.clf()


if __name__ == "__main__":
    main()
