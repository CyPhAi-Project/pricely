from datetime import date
from dreal import Variable  # type: ignore
import numpy as np

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from pathlib import Path

from plot_utils_2d import CatchTime, add_level_sets, add_valid_regions, validate_lip_bbox
from pricely.approx.boxes import AxisAlignedBoxes
from pricely.approx.simplices import SimplicialComplex
from pricely.cegus_lyapunov import cegus_lyapunov
from pricely.gen_cover import gen_init_cover
from pricely.learner_cvxpy import QuadraticLearner
from pricely.utils import cartesian_prod, check_lyapunov_roi, check_lyapunov_sublevel_set, gen_equispace_regions
from pricely.verifier_dreal import SMTVerifier, pretty_sub


OUT_DIR = Path(f"out/{str(date.today())}")
OUT_DIR.mkdir(exist_ok=True)


def main(max_epochs: int=15):
    # import circle_following as mod
    # import lcss2020_eq5 as mod
    # import lcss2020_eq13 as mod
    import path_following_stanley as mod
    # import van_der_pol as mod
    # import inverted_pendulum as mod

    timer = CatchTime()

    init_part = [30]*mod.X_DIM
    print(" Validate local Lipschitz constants ".center(80, "="))
    with timer:
        validate_lip_bbox(mod, init_part)

    print(" Generate initial samples and cover ".center(80, "="))
    with timer:
        if False:
            x_regions = gen_equispace_regions(init_part, mod.X_ROI)
            approx = AxisAlignedBoxes(
                x_roi=mod.X_ROI,
                u_roi=getattr(mod, "U_ROI", np.empty((2, 0))),
                x_regions=x_regions,
                u_values=np.empty((len(x_regions), 0)),
                f_bbox=lambda x, u: mod.f_bbox(x),  # TODO: support systems with and without input
                lip_bbox=lambda x, u: mod.calc_lip_bbox(x))  # TODO: support systems with and without input
        else:
            axis_cuts = [
                np.linspace(start=bnd[0], stop=bnd[1], num=cuts+1)
                for bnd, cuts in zip(mod.X_ROI.T, init_part)
            ]
            x_values = cartesian_prod(*axis_cuts)
            approx = SimplicialComplex(
                x_roi=mod.X_ROI,
                u_roi=getattr(mod, "U_ROI", np.empty((2, 0))),
                x_values=x_values,
                u_values=np.empty((len(x_values), 0)),
                f_bbox=lambda x, u: mod.f_bbox(x),  # TODO: support systems with and without input
                lip_bbox=lambda x, u: mod.calc_lip_bbox(x))  # TODO: support systems with and without input

    print(" Run CEGuS ".center(80, "="))
    with timer:
        learner = QuadraticLearner(mod.X_DIM)
        verifier = SMTVerifier(x_roi=mod.X_ROI, abs_x_lb=mod.ABS_X_LB)
        last_epoch, last_approx, cex_regions = \
            cegus_lyapunov(
                learner, verifier, approx,
                max_epochs=max_epochs, max_iter_learn=1)
        level_ub, abs_x_ub = verifier._lya_cand_level_ub, verifier._abs_x_ub
    cegus_status = "Found" if not cex_regions else "Can't Find" if last_epoch < max_epochs else "Reach epoch limit"
    cegus_time_usage = timer.elapsed

    print(" Validate learned Lyapunov candidate ".center(80, "="))
    with timer:
        x_vars = [Variable(f"x{pretty_sub(i)}") for i in range(mod.X_DIM)]
        dxdt_exprs = mod.f_expr(x_vars)
        lya_expr = learner.lya_expr(x_vars)
        lya_decay_rate = learner.lya_decay_rate()
        print(f"Decay rate of Lyapunov potential: {lya_decay_rate}")
        result = check_lyapunov_roi(
            x_vars, dxdt_exprs, lya_expr,
            mod.X_ROI,
            lya_decay_rate,
            mod.ABS_X_LB, config=verifier._config)
        if result is None:
            print("Learned candidate is a valid Lyapunov function for ROI.")
        else:
            print("Learned candidate is NOT a Lyapunov function for ROI.")
            print(f"Counterexample:\n{result}")
        validation = (result is None)

        result = check_lyapunov_sublevel_set(
            x_vars, dxdt_exprs, lya_expr, lya_decay_rate,
            level_ub, mod.ABS_X_LB, abs_x_ub, config=verifier._config)
        if result is None:
            print("The Basin of Attraction can cover the entire ROI.")
        else:
            print("The Basin of Attraction cannot cover the entire ROI.")
            print(f"Counterexample:\n{result}")
        cover_roi = (result is None)

    if mod.X_DIM != 2:  # Support plotting 2D systems only
        return

    print(" Plotting verified regions ".center(80, "="))
    # Calculate the axis-aligned bounding box
    abs_x_ub = np.asfarray(abs_x_ub)
    # plt.gca().set_xlim(-1.125*abs_x_ub[0], +1.125*abs_x_ub[0])
    # plt.gca().set_ylim(-1.125*abs_x_ub[1], +1.125*abs_x_ub[1])
    plt.gca().set_xlim(*(1.125*mod.X_ROI[:, 0]))
    plt.gca().set_ylim(*(1.125*mod.X_ROI[:, 1]))

    plt.gca().set_title(
        f"CEGuS Status: {cegus_status}.\n"
        f"Is True Lyapunov: {str(validation)}. "
        f"BOA covers ROI: {str(cover_roi)}. \n"
        f"# epoch: {last_epoch}. "
        f"# total samples: {len(last_approx.x_values)}. "
        f"Time: {cegus_time_usage:.3f}s")
    add_valid_regions(
        plt.gca(), last_approx, cex_regions)
    plt.gca().add_patch(Rectangle(
        (-mod.ABS_X_LB, -mod.ABS_X_LB), 2*mod.ABS_X_LB, 2*mod.ABS_X_LB, color='b', fill=False))
    plt.gca().add_patch(Rectangle(
        (-abs_x_ub[0], -abs_x_ub[1]), 2*abs_x_ub[0], 2*abs_x_ub[1], color='b', fill=False))
    plt.gca().add_patch(Rectangle(
        (mod.X_ROI[0][0], mod.X_ROI[0][1]), mod.X_ROI[1][0]-mod.X_ROI[0][0], mod.X_ROI[1][1]-mod.X_ROI[0][1], color='r', fill=False))

    # add_level_sets(plt.gca(), learner.lya_values, level_ub=level_ub)

    plt.gca().set_aspect("equal")
    plt.tight_layout()
    f_name = f"cegus-{mod.__name__}-valid_regions-{'x'.join(str(n) for n in init_part)}.png"
    f_path = OUT_DIR / f_name
    plt.savefig(f_path)
    plt.clf()
    print(f'The plot is saved to "{f_path}".')


if __name__ == "__main__":
    main()
