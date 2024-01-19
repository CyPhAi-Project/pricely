from dreal import atan as ArcTan, Max, Min, sin as Sin, Variable  # type: ignore
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
import time

from plot_utils_2d import add_level_sets, add_valid_regions
from pricely.cegus_lyapunov import cegus_lyapunov
from pricely.learner_cvxpy import QuadraticLearner
from pricely.verifier_dreal import SMTVerifier
from pricely.utils import check_exact_lyapunov, gen_equispace_regions


THETA_LIM = np.pi / 2  # rad
K_P = 0.45
WHEEL_BASE = 1.75  # m
VEL = 2.8  # m/s
STEER_LIM = 0.61  # rad

X_DIM = 2
X_ROI = np.array([
    [-1., -1.],  # Lower bounds
    [+1., +1.]  # Upper bounds
])
assert X_ROI.shape == (2, X_DIM)
NORM_LB, NORM_UB = 0.125, 1.0
LIP_BB = np.sqrt((VEL/WHEEL_BASE)**2 + K_P**2)  # Manually derived for ROI


def ctrl(x: np.ndarray) -> np.ndarray:
    u = np.zeros(shape=(len(x), 1))
    u[:, 0] = np.clip(x[:, 1] + np.arctan(K_P*x[:, 0] / VEL),
                      a_min=-STEER_LIM, a_max=STEER_LIM)
    return u


def f_bbox(x: np.ndarray) -> np.ndarray:
    u = ctrl(x)
    dxdt = np.zeros_like(x)
    dxdt[:, 0] = VEL*np.sin(x[:, 1] - u[:, 0])
    dxdt[:, 1] = -(VEL/WHEEL_BASE)*np.sin(u[:, 0])
    return dxdt


def validate(lya: QuadraticLearner):
    x_vars = [Variable("e"), Variable("ψ")]
    assert len(x_vars) == X_DIM
    u_expr = x_vars[1] + ArcTan(K_P*x_vars[0] / VEL)
    u_clip_expr = Min(Max(u_expr, -STEER_LIM), STEER_LIM)
    dxdt_exprs = [
        VEL*Sin(x_vars[1] - u_clip_expr),
        -(VEL/WHEEL_BASE)*Sin(u_clip_expr)
    ]
    lya_expr = lya.lya_expr(x_vars)
    return check_exact_lyapunov(
        x_vars, dxdt_exprs, X_ROI,
        lya_expr, NORM_LB, NORM_UB)


def main():
    x_part = [4]*X_DIM  # Partition into subspaces
    assert len(x_part) == X_DIM
    print(f"Prepare {'x'.join(str(n) for n in x_part)} "
          f"equispaced training samples.")
    x_regions = gen_equispace_regions(x_part, X_ROI)

    lya = QuadraticLearner(X_DIM)
    verifier = SMTVerifier(x_roi=X_ROI, norm_lb=NORM_LB, norm_ub=NORM_UB)

    t_start = time.perf_counter()
    last_epoch, last_x_regions, cex_regions = cegus_lyapunov(
        lya, verifier, x_regions, f_bbox, LIP_BB,
        max_epochs=20, max_iter_learn=1)
    time_usage = time.perf_counter() - t_start
    print(f"Total Time: {time_usage:.3f}s")

    print("Validate the Lyapunov candidate with true dynamics:")
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
    f_name = f"out/PathFollowingStanley-valid_regions-{'x'.join(str(n) for n in x_part)}.png"
    plt.savefig(f_name)
    plt.clf()


if __name__ == "__main__":
    main()
