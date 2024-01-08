import dreal
from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
import time
import torch

from cegus_lyapunov import cegus_lyapunov
from lyapunov_learner_cvx import QuadraticLearner
from lyapunov_verifier import SMTVerifier, check_exact_lyapunov
from nnet_utils import gen_equispace_regions


def main():
    THETA_LIM = torch.pi / 2  # rad
    K_P = 0.45
    WHEEL_BASE = 1.75  # m
    VEL = 2.8  # m/s
    STEER_LIM = 0.61  # rad

    X_DIM = 2
    X_ROI = torch.tensor([
        [-1.5, -THETA_LIM], # Lower bounds
        [+1.5, +THETA_LIM]  # Upper bounds
    ])
    assert X_ROI.shape == (2, X_DIM)
    NORM_LB, NORM_UB = 0.125, 1.0

    U_DIM = 1
    def ctrl(x: torch.Tensor) -> torch.Tensor:
        u = torch.zeros(size=(len(x), U_DIM), device=x.device)
        u[:, 0] = torch.clip(x[:, 1] + torch.arctan(K_P*x[:, 0] / VEL),
                             min=-STEER_LIM, max=STEER_LIM)
        return u

    def f_bbox(x: torch.Tensor) -> torch.Tensor:
        u = ctrl(x)
        dxdt = torch.zeros_like(x, device=x.device)
        dxdt[:, 0] = VEL*torch.sin(x[:, 1] - u[:, 0])
        dxdt[:, 1] = -(VEL/WHEEL_BASE)*torch.sin(u[:, 0])
        return dxdt

    LIP_BB = VEL  # Manually derived for ROI

    x_part = (X_DIM, X_DIM)  # Partition into subspaces
    assert len(x_part) == X_DIM
    print(
        f"Prepare {'x'.join(str(n) for n in x_part)} equispaced training samples: ",
        end="", flush=True)
    t_start = time.perf_counter()
    x_regions = gen_equispace_regions(x_part, X_ROI)
    print(f"{time.perf_counter() - t_start:.3f}s")

    lya = QuadraticLearner(X_DIM)
    verifier = SMTVerifier(
        x_roi=X_ROI.cpu().numpy(),
        norm_lb=NORM_LB, norm_ub=NORM_UB)

    x_regions_np, cex_regions = cegus_lyapunov(
        lya, verifier, x_regions, f_bbox, LIP_BB,
        max_epochs=20, max_iter_learn=1)

    # Validate with exact Lyapunov conditions
    x_vars = [axis_i.x for axis_i in verifier._all_vars]
    u_expr = x_vars[1] + dreal.atan(K_P*x_vars[0] / VEL)
    u_clip_expr = dreal.Min(dreal.Max(u_expr, -STEER_LIM), STEER_LIM)
    dxdt_exprs = [
        VEL*dreal.sin(x_vars[1] - u_clip_expr),
        -(VEL/WHEEL_BASE)*dreal.sin(u_clip_expr)
    ]
    lya_expr = lya.lya_expr(x_vars)
    result = check_exact_lyapunov(
        x_vars, dxdt_exprs, X_ROI.cpu().numpy(),
        lya_expr, NORM_LB, NORM_UB, verifier._config)
    if result is None:
        print("Learned candidate is a valid Lyapunov function.")
    else:
        print("Learned candidate is NOT a Lyapunov function.")
        print(f"Counterexample:\n{result}")

    print("Plotting verified regions:")
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

    plt.gca().add_patch(Circle((0, 0), NORM_LB, color='r', fill=False))
    plt.gca().add_patch(Circle((0, 0), NORM_UB, color='r', fill=False))
    plt.gca().set_xlim(*X_ROI[:, 0])
    plt.gca().set_ylim(*X_ROI[:, 1])
    plt.gca().set_aspect("equal")
    plt.savefig("out/PathFollowingStanley-valid_regions.png")
    plt.clf()


if __name__ == "__main__":
    main()
