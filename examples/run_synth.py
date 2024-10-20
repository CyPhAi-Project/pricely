from datetime import date
from dreal import Config, Variable  # type: ignore
import numpy as np
from math import factorial
from matplotlib.patches import Circle, Ellipse, Rectangle
import matplotlib.pyplot as plt
from pathlib import Path

from plot_utils_2d import CatchTime, add_level_sets, add_valid_regions, validate_lip_bbox
from pricely.approx.simplices import SimplicialComplex
from pricely.candidates import QuadraticLyapunov
from pricely.cegus_lyapunov import cegus_lyapunov
from pricely.learner_cvxpy import QuadraticLearner
from pricely.utils import cartesian_prod, check_lyapunov_roi, check_lyapunov_sublevel_set
from pricely.verifier_dreal import SMTVerifier, pretty_sub

NCOLS = 120

OUT_DIR = Path(f"out/{str(date.today())}")
OUT_DIR.mkdir(exist_ok=True)


def viz_regions_2d(ax, mod, last_approx, cex_regions):
    print(" Plotting verified regions ".center(NCOLS, "="))
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
    else:
        ax.add_patch(Rectangle(
            (x_roi[0][0], x_roi[0][1]), x_roi[1][0]-x_roi[0][0], x_roi[1][1]-x_roi[0][1], color='r', fill=False))


def viz_basin_2d(ax, mod, cand: QuadraticLyapunov):
    x_roi = mod.X_ROI
    # Draw Basin of Attraction
    if hasattr(mod, "NORM_UB"):
        l_min = np.linalg.eigvalsh(cand._sym_mat)[0]
        level_ub = 0.5*l_min*(mod.NORM_UB**2)
    else:
        p_inv = np.linalg.inv(cand._sym_mat)
        level_ub = 0.5*(x_roi[1]**2 / p_inv.diagonal()).min()
    add_level_sets(ax, cand.lya_values, level_ub, colors="b")


def viz_region_stats(x_roi, approx, cex_regions):
    x_dim = x_roi.shape[1]

    assert isinstance(approx, SimplicialComplex)
    print(" Plotting volumes of all regions in descending order".center(NCOLS, "="))
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
    # import hscc2014_normalized_pendulum as mod
    # import fossil_nonpoly0 as mod
    # import fossil_nonpoly1 as mod
    # import fossil_nonpoly2 as mod
    # import fossil_nonpoly3 as mod
    # import fossil_poly1 as mod
    # import fossil_poly2 as mod
    # import fossil_poly3 as mod
    import fossil_poly4 as mod
    # import traj_tracking_wheeled as mod
    # import neurips2022_van_der_pol as mod
    # import neurips2022_unicycle_following as mod
    # import neurips2022_inverted_pendulum as mod
    # import path_following_stanley as mod

    timer = CatchTime()

    init_part = [5]*mod.X_DIM
    print(" Validate local Lipschitz constants ".center(NCOLS, "="))
    with timer:
        validate_lip_bbox(mod, init_part)

    print(" Generate initial samples and cover ".center(NCOLS, "="))
    with timer:
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

    print(" Run CEGuS ".center(NCOLS, "="))
    with timer:
        learner = QuadraticLearner(mod.X_DIM, v_max=1e8)
        verifier = SMTVerifier(
            x_roi=mod.X_ROI, abs_x_lb=mod.ABS_X_LB,
            norm_lb=getattr(mod, "NORM_LB", 0.0),
            norm_ub=getattr(mod, "NORM_UB", np.inf), config=config)
        status, last_epoch, last_approx, cex_regions = \
            cegus_lyapunov(
                learner, verifier, approx,
                eps=1e-3,
                max_epochs=max_epochs, max_iter_learn=1, n_jobs=n_jobs)
        print(f"\nCEGuS Status: {status}")
    if status != "NO_CANDIDATE":
        cand = learner.get_candidate()
        print("Last candidate (possibly with precision loss):\n", str(cand))
    cegus_status = status
    cegus_time_usage = timer.elapsed


    print(" Validate learned Lyapunov candidate ".center(NCOLS, "="))
    validation = "N/A"
    cover_roi = "N/A"
    if cegus_status == "NO_CANDIDATE":
        print("Skipped due to no feasible candidate.")
    else:
        x_vars = [Variable(f"x{pretty_sub(i)}") for i in range(mod.X_DIM)]
        dxdt_exprs = mod.f_expr(x_vars)
        cand = learner.get_candidate()
        lya_expr = cand.lya_expr(x_vars)
        lya_decay_rate = cand.lya_decay_rate()
        print(f"Check Lyapunov potential with decay rate: {lya_decay_rate}")
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

    fig_err = viz_region_stats(mod.X_ROI, last_approx, cex_regions)
    f_name = f"err-{mod.__name__}-cover-{'x'.join(str(n) for n in init_part)}.png"
    f_path = OUT_DIR / f_name
    fig_err.savefig(f_path)  # type: ignore
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
    viz_regions_2d(ax, mod, last_approx, cex_regions)
    if cegus_status == "FOUND":
        viz_basin_2d(ax, mod, learner.get_candidate())
    ax.set_aspect("equal")
    fig_cover.tight_layout()
    f_name = f"cegus-{mod.__name__}-valid_regions-{'x'.join(str(n) for n in init_part)}.png"
    f_path = OUT_DIR / f_name
    plt.savefig(f_path)
    plt.clf()
    print(f'The plot is saved to "{f_path}".')


if __name__ == "__main__":
    main()
