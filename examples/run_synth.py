from datetime import date
from dreal import Config, Variable  # type: ignore
import numpy as np
from math import factorial
from matplotlib.patches import Circle, Ellipse, Rectangle
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist

from plot_utils_2d import CatchTime, add_level_sets, add_valid_regions, validate_lip_bbox
from pricely.approx.boxes import AxisAlignedBoxes
from pricely.approx.simplices import SimplicialComplex
from pricely.candidates import QuadraticLyapunov
from pricely.cegus_lyapunov import cegus_lyapunov
from pricely.learner_cvxpy import QuadraticLearner
from pricely.utils import cartesian_prod, check_lyapunov_roi, check_lyapunov_sublevel_set, gen_equispace_regions
from pricely.verifier_dreal import SMTVerifier, pretty_sub


OUT_DIR = Path(f"out/{str(date.today())}")
OUT_DIR.mkdir(exist_ok=True)


def viz_2d(ax, mod, last_approx, cex_regions, cand: QuadraticLyapunov):
    print(" Plotting verified regions ".center(80, "="))
    # Calculate the axis-aligned bounding box
    # ax.set_xlim(-1.125*abs_x_ub[0], +1.125*abs_x_ub[0])
    # ax.set_ylim(-1.125*abs_x_ub[1], +1.125*abs_x_ub[1])
    x_roi = mod.X_ROI
    abs_x_lb = mod.ABS_X_LB
    ax.set_xlim(*(1.0625*x_roi[:, 0]))
    ax.set_ylim(*(1.0625*x_roi[:, 1]))

    add_valid_regions(
        ax, last_approx, cex_regions)
    # ax.add_patch(Rectangle(
    #     (-abs_x_ub[0], -abs_x_ub[1]), 2*abs_x_ub[0], 2*abs_x_ub[1], color='b', fill=False))

    if hasattr(mod, "NORM_LB"):
        ax.add_patch(Circle((0, 0), mod.NORM_LB, color='r', fill=False))
    else:
        ax.add_patch(Rectangle(
            (-abs_x_lb, -abs_x_lb), 2*abs_x_lb, 2*abs_x_lb, color='r', fill=False))

    if hasattr(mod, "NORM_UB"):
        ax.add_patch(Circle((0, 0), mod.NORM_UB, color='r', fill=False))
        l_min = np.linalg.eigvalsh(cand._pd_mat)[0]
        level_ub = 0.5*l_min*(mod.NORM_UB**2)
    else:
        ax.add_patch(Rectangle(
            (x_roi[0][0], x_roi[0][1]), x_roi[1][0]-x_roi[0][0], x_roi[1][1]-x_roi[0][1], color='r', fill=False))
        p_inv = np.linalg.inv(cand._pd_mat)
        level_ub = 0.5*(x_roi[1]**2 / p_inv.diagonal()).min()
    # Draw Basin of Attraction
    add_level_sets(ax, cand.lya_values, level_ub, colors="b")


def viz_region_stats(x_roi, cand, approx, cex_regions):
    from pricely.candidates import QuadraticLyapunov

    x_dim = x_roi.shape[1]

    assert isinstance(cand, QuadraticLyapunov)
    assert isinstance(approx, SimplicialComplex)
    print(" Plotting volumes of all regions in descending order".center(80, "="))
    fig = plt.figure()
    fig.suptitle("Volumes of regions")
    ax = fig.add_subplot(211)

    tri = approx._triangulation

    # max_dists = [pdist(approx.x_values[simplex]).max() for simplex in tri.simplices]
    # max_dists.sort(reverse=True)
    # ax.bar(np.arange(len(max_dists)), max_dists, width=1.0)

    vols = []
    n_fact = factorial(x_dim)
    for simplex in tri.simplices:
        x_vertices = approx.x_values[simplex]
        vol = abs(np.linalg.det((x_vertices[1:] - x_vertices[0])))/n_fact
        vols.append(vol)
    vols = np.asfarray(vols)
    sorted_idx = np.flip(np.argsort(vols))

    print(f"Vol sum: {sum(vols)}")
    ax.bar(np.arange(len(vols)), vols[sorted_idx], color="b", width=1.0)

    ax1 = fig.add_subplot(212)
    ax1.scatter(vols[sorted_idx], approx._lips[sorted_idx], s=0.2)
    fig.tight_layout()
    return fig


def main(max_epochs: int=40, n_jobs: int=16):
    # import circle_following as mod
    # import lcss2020_eq5 as mod
    # import lcss2020_eq13 as mod
    # import lcss2020_eq14 as mod
    import lcss2020_eq15 as mod
    # import path_following_stanley as mod
    # import traj_tracking_wheeled as mod
    # import van_der_pol as mod
    # import inverted_pendulum as mod

    timer = CatchTime()

    init_part = [5]*mod.X_DIM
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
            approx = SimplicialComplex.from_autonomous(
                x_roi=mod.X_ROI,
                x_values=x_values,
                f_bbox=mod.f_bbox,
                lip_bbox=mod.calc_lip_bbox)

    # dReal configurations
    config = Config()
    config.use_polytope_in_forall = True
    config.use_local_optimization = True
    config.precision = 1e-7
    config.number_of_jobs = 1

    print(" Run CEGuS ".center(80, "="))
    with timer:
        learner = QuadraticLearner(mod.X_DIM)
        verifier = SMTVerifier(
            x_roi=mod.X_ROI, abs_x_lb=mod.ABS_X_LB,
            norm_lb=getattr(mod, "NORM_LB", 0.0),
            norm_ub=getattr(mod, "NORM_UB", np.inf), config=config)
        last_epoch, last_approx, cex_regions = \
            cegus_lyapunov(
                learner, verifier, approx,
                max_epochs=max_epochs, max_iter_learn=1, n_jobs=n_jobs)
    cegus_status = "Found" if not cex_regions else "Can't Find" if last_epoch < max_epochs else "Reach epoch limit"
    cegus_time_usage = timer.elapsed

    print(" Validate learned Lyapunov candidate ".center(80, "="))
    with timer:
        x_vars = [Variable(f"x{pretty_sub(i)}") for i in range(mod.X_DIM)]
        dxdt_exprs = mod.f_expr(x_vars)
        cand = learner.get_candidate()
        lya_expr = cand.lya_expr(x_vars)
        lya_decay_rate = cand.lya_decay_rate()
        level_ub, abs_x_ub = cand.find_sublevel_set_and_box(mod.X_ROI)
        print(f"Decay rate of Lyapunov potential: {lya_decay_rate}")
        result = check_lyapunov_roi(
            x_vars, dxdt_exprs, lya_expr,
            mod.X_ROI,
            lya_decay_rate,
            mod.ABS_X_LB,
            norm_lb=getattr(mod, "NORM_LB", 0.0),
            norm_ub=getattr(mod, "NORM_UB", np.inf),
            config=config)
        if result is None:
            print("Learned candidate is a valid Lyapunov function for ROI.")
        else:
            print("Learned candidate is NOT a Lyapunov function for ROI.")
            print(f"Counterexample:\n{result}")
        validation = (result is None)

        result = check_lyapunov_sublevel_set(
            x_vars, dxdt_exprs, lya_expr, lya_decay_rate,
            level_ub, mod.ABS_X_LB, abs_x_ub, config=config)
        if result is None:
            print("The Basin of Attraction can cover the entire ROI.")
        else:
            print("Cannot prove if the Basin of Attraction covers the entire ROI.")
        cover_roi = (result is None)


    fig_err = viz_region_stats(mod.X_ROI, cand, last_approx, cex_regions)
    f_name = f"err-{mod.__name__}-cover-{'x'.join(str(n) for n in init_part)}.png"
    f_path = OUT_DIR / f_name
    fig_err.savefig(f_path)
    plt.clf()
    print(f'The plot is saved to "{f_path}".')

    if mod.X_DIM != 2:  # Support plotting 2D systems only
        return

    fig_cover = plt.figure()
    fig_cover.suptitle(f"Cover of ROI for {mod.__name__}")
    ax = fig_cover.add_subplot()
    ax.set_title(
        f"CEGuS Status: {cegus_status}.\n"
        f"Is True Lyapunov: {str(validation)}. "
        f"BOA covers ROI: {str(cover_roi)}. \n"
        f"# epoch: {last_epoch}. "
        f"# total samples: {len(last_approx.x_values)}. "
        f"Time: {cegus_time_usage:.3f}s")
    viz_2d(ax, mod, last_approx, cex_regions, cand)
    # add_level_sets(ax, cand.lya_values, level_ub=level_ub)
    ax.set_aspect("equal")
    fig_cover.tight_layout()
    f_name = f"cegus-{mod.__name__}-valid_regions-{'x'.join(str(n) for n in init_part)}.png"
    f_path = OUT_DIR / f_name
    plt.savefig(f_path)
    plt.clf()
    print(f'The plot is saved to "{f_path}".')


if __name__ == "__main__":
    main()
