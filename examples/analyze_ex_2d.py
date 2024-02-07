from dreal import Variable  # type: ignore
import numpy as np
import time

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

from plot_utils_2d import add_level_sets, add_valid_regions
from pricely.cegus_lyapunov import cegus_lyapunov
from pricely.learner_cvxpy import QuadraticLearner
from pricely.utils import check_lyapunov_roi, check_lyapunov_sublevel_set, gen_equispace_regions, gen_lip_bbox
from pricely.verifier_dreal import SMTVerifier, pretty_sub
from pricely.gen_cover import gen_init_cover


class CatchTime:
    @property
    def elapsed(self) -> float:
        return self._elapsed

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self._elapsed = time.perf_counter() - self._start
        self._readout = f'Time Usage: {self._elapsed:.3f} seconds'
        print(self._readout)


def validate_lip_bbox(mod, parts):
    est_lip_lb = gen_lip_bbox(mod.X_DIM, mod.f_bbox)

    regions = gen_equispace_regions(parts, mod.X_ROI)

    lip_values = mod.calc_lip_bbox(regions)
    lip_lbs = np.fromiter((est_lip_lb(reg) for reg in regions), dtype=float)
    assert np.all(lip_lbs <= lip_values)

    min_lb_idx = np.argmin(lip_lbs)
    max_lb_idx = np.argmax(lip_lbs)
    min_diff_idx = np.argmin(lip_values - lip_lbs)
    max_diff_idx = np.argmax(lip_values -lip_lbs)
    print(f"Min Estimated vs Provided: {lip_lbs[min_lb_idx]:.3f} <= {lip_values[min_lb_idx]:.3f}")
    print(f"Max Estimated vs Provided: {lip_lbs[max_lb_idx]:.3f} <= {lip_values[max_lb_idx]:.3f}")
    print(f"Best diff: {lip_lbs[min_diff_idx]:.3f} <= {lip_values[min_diff_idx]:.3f}")
    print(f"Worst diff: {lip_lbs[max_diff_idx]:.3f} <= {lip_values[max_diff_idx]:.3f}")


def main(max_epochs: int=15):
    import path_following_stanley as mod

    timer = CatchTime()

    init_part = [15, 30]
    print(" Validate local Lipschitz constants ".center(80, "="))
    with timer:
        validate_lip_bbox(mod, init_part)

    print(" Generate initial samples and cover ".center(80, "="))
    with timer:
        x_regions = gen_init_cover(
            abs_roi_ub=mod.X_ROI[1],
            f_bbox=mod.f_bbox,
            lip_bbox=mod.calc_lip_bbox,
            lip_cap=getattr(mod, "LIP_CAP", np.inf),
            abs_lb=mod.ABS_X_LB,
            init_part=init_part)

    print(" Run CEGuS ".center(80, "="))
    with timer:
        learner = QuadraticLearner(mod.X_DIM)
        verifier = SMTVerifier(x_roi=mod.X_ROI, abs_x_lb=mod.ABS_X_LB)
        last_epoch, last_x_regions, cex_regions = \
            cegus_lyapunov(
                learner, verifier, x_regions,
                mod.f_bbox, mod.calc_lip_bbox,
                max_epochs=max_epochs, max_iter_learn=1)
        level_ub, abs_x_ub = verifier._lya_cand_level_ub, verifier._abs_x_ub
    cegus_time_usage = timer.elapsed

    print(" Validate learned Lyapunov candidate ".center(80, "="))
    with timer:
        x_vars = [Variable(f"x{pretty_sub(i)}") for i in range(mod.X_DIM)]
        dxdt_exprs = mod.f_expr(x_vars)
        lya_expr = learner.lya_expr(x_vars)
        result = check_lyapunov_roi(
            x_vars, dxdt_exprs, lya_expr,
            mod.X_ROI, mod.ABS_X_LB, config=verifier._config)
        if result is None:
            print("Learned candidate is a valid Lyapunov function for ROI.")
        else:
            print("Learned candidate is NOT a Lyapunov function for ROI.")
            print(f"Counterexample:\n{result}")

        result = check_lyapunov_sublevel_set(
            x_vars, dxdt_exprs, lya_expr,
            level_ub, mod.ABS_X_LB, abs_x_ub, config=verifier._config)
        if result is None:
            print("The Basin of Attraction can cover the entire ROI.")
        else:
            print("The Basin of Attraction cannot cover the entire ROI.")
            print(f"Counterexample:\n{result}")

    print(" Plotting verified regions ".center(80, "="))
    # Calculate the axis-aligned bounding box
    abs_x_ub = np.asfarray(abs_x_ub)
    # plt.gca().set_xlim(-1.125*abs_x_ub[0], +1.125*abs_x_ub[0])
    # plt.gca().set_ylim(-1.125*abs_x_ub[1], +1.125*abs_x_ub[1])
    plt.gca().set_xlim(*(1.125*mod.X_ROI[:, 0]))
    plt.gca().set_ylim(*(1.125*mod.X_ROI[:, 1]))

    add_valid_regions(
        plt.gca(), last_epoch, cegus_time_usage, last_x_regions, cex_regions)
    plt.gca().add_patch(Rectangle(
        (-mod.ABS_X_LB, -mod.ABS_X_LB), 2*mod.ABS_X_LB, 2*mod.ABS_X_LB, color='b', fill=False))
    plt.gca().add_patch(Rectangle(
        (-abs_x_ub[0], -abs_x_ub[1]), 2*abs_x_ub[0], 2*abs_x_ub[1], color='b', fill=False))
    plt.gca().add_patch(Rectangle(
        (mod.X_ROI[0][0], mod.X_ROI[0][1]), mod.X_ROI[1][0]-mod.X_ROI[0][0], mod.X_ROI[1][1]-mod.X_ROI[0][1], color='r', fill=False))

    add_level_sets(plt.gca(), learner.lya_values, level_ub=level_ub)

    plt.gca().set_aspect("equal")
    f_name = f"out/{mod.__name__}-valid_regions-{'x'.join(str(n) for n in init_part)}.png"
    plt.savefig(f_name)
    plt.clf()
    print(f'The plot is saved to "{f_name}".')


if __name__ == "__main__":
    main()
