from dreal import Config, Variable  # type: ignore
import numpy as np
from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from scripts.utils_plotting_2d import CatchTime, add_level_sets, add_valid_regions
from pricely.approx.boxes import AxisAlignedBoxes
from pricely.approx.simplices import SimplicialComplex
from pricely.candidates import QuadraticLyapunov
from pricely.cegus_lyapunov import ROI, cegus_lyapunov
from pricely.learner.cvxpy import QuadraticLearner
from pricely.utils import cartesian_prod, check_lyapunov_roi, gen_equispace_regions
from pricely.verifier.smt_dreal import SMTVerifier, pretty_sub

NCOLS = 120


def viz_regions_2d(ax, mod, last_approx, cex_regions):
    print(" Plotting verified regions ".center(NCOLS, "="))
    x_lim = mod.X_LIM
    abs_x_lb = mod.ABS_X_LB
    ax.set_xlim(*(1.0625*x_lim[:, 0]))
    ax.set_ylim(*(1.0625*x_lim[:, 1]))

    add_valid_regions(
        ax, last_approx, cex_regions)

    if hasattr(mod, "X_NORM_LB"):
        ax.add_patch(Circle((0, 0), mod.X_NORM_LB, color='r', fill=False))
    else:
        ax.add_patch(Rectangle(
            (-abs_x_lb, -abs_x_lb), 2*abs_x_lb, 2*abs_x_lb, color='r', fill=False))

    if hasattr(mod, "X_NORM_UB"):
        ax.add_patch(Circle((0, 0), mod.X_NORM_UB, color='r', fill=False))
    else:
        ax.add_patch(Rectangle(
            (x_lim[0][0], x_lim[0][1]), x_lim[1][0]-x_lim[0][0], x_lim[1][1]-x_lim[0][1], color='r', fill=False))


def viz_basin_2d(ax, mod, cand: QuadraticLyapunov):
    x_lim = mod.X_LIM
    # Draw Basin of Attraction
    if hasattr(mod, "X_NORM_UB"):
        l_min = np.linalg.eigvalsh(cand._sym_mat)[0]
        level_ub = 0.5*l_min*(mod.X_NORM_UB**2)
    else:
        p_inv = np.linalg.inv(cand._sym_mat)
        level_ub = 0.5*(x_lim[1]**2 / p_inv.diagonal()).min()
    add_level_sets(ax, cand.lya_values, level_ub, colors="b")


def viz_region_stats(x_lim, approx, cex_regions):
    print(" Plotting diameters of all regions in descending order".center(NCOLS, "="))
    fig = plt.figure()
    fig.suptitle("Diameters of regions")
    ax = fig.add_subplot(211)

    diams = [approx[i].domain_diameter for i in range(len(approx))]
    diams = np.asfarray(diams)
    sorted_idx = np.flip(np.argsort(diams))

    ax.bar(np.arange(len(diams)), diams[sorted_idx], color="b", width=1.0)

    ax1 = fig.add_subplot(212)
    ax1.scatter(diams[sorted_idx], approx._lip_values[sorted_idx], s=0.2)
    fig.tight_layout()
    return fig


def main(mod, out_dir: Optional[Path]=None,
        max_epochs: int=40, max_num_samples: int=5*10**5,
        n_jobs: int=16) -> Optional[QuadraticLyapunov]:
    x_roi = ROI(
        x_lim=mod.X_LIM,
        abs_x_lb=mod.ABS_X_LB,
        x_norm_lim=(getattr(mod, "X_NORM_LB", 0.0),
                    getattr(mod, "X_NORM_UB", np.inf)))

    timer = CatchTime()

    init_part = [5]*mod.X_DIM

    print(" Generate initial samples and cover ".center(NCOLS, "="))
    with timer:
        if mod.X_DIM == 1:
            approx = AxisAlignedBoxes.from_autonomous(
                x_roi=x_roi,
                x_regions=gen_equispace_regions(init_part, x_roi.x_lim),
                f_bbox=mod.f_bbox,
                lip_bbox=mod.calc_lip_bbox)
        else:
            axis_cuts = [
                np.linspace(start=bnd[0], stop=bnd[1], num=cuts+1)
                for bnd, cuts in zip(mod.X_LIM.T, init_part)]
            x_values = cartesian_prod(*axis_cuts)
            approx = SimplicialComplex.from_autonomous(
                x_roi=x_roi,
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
        verifier = SMTVerifier(x_roi=x_roi, config=config)
        status, last_epoch, last_approx, cex_regions = \
            cegus_lyapunov(
                learner, verifier, approx,
                delta=1e-4,
                max_epochs=max_epochs, max_iter_learn=1,
                max_num_samples=max_num_samples,
                n_jobs=n_jobs)
        print(f"\nCEGuS Status: {status}")
    cand: Optional[QuadraticLyapunov] = None
    if status != "NO_CANDIDATE":
        cand = learner.get_candidate()
        print("Last candidate (possibly with precision loss):\n", str(cand))
    cegus_status = status
    cegus_time_usage = timer.elapsed


    print(" Validate learned Lyapunov candidate ".center(NCOLS, "="))
    validation = "N/A"
    if cand is None:
        print("Skipped due to no feasible candidate.")
    else:
        assert cand is not None
        x_vars = [Variable(f"x{pretty_sub(i)}") for i in range(mod.X_DIM)]
        dxdt_exprs = mod.f_expr(x_vars)
        lya_expr = cand.lya_expr(x_vars)
        lya_decay_rate = cand.lya_decay_rate()
        print(f"Check Lyapunov potential with decay rate: {lya_decay_rate}")
        result = check_lyapunov_roi(
            x_vars, dxdt_exprs, lya_expr,
            x_roi,
            lya_decay_rate=lya_decay_rate,
            config=config)
        if result is None:
            print("Learned candidate is a valid Lyapunov function for ROI.")
        else:
            print("Learned candidate is NOT a Lyapunov function for ROI.")
            print(f"Counterexample:\n{result}")
        validation = (result is None)

    print(" Statistics ".center(NCOLS, "="))
    print(f"CEGuS Status: {cegus_status}.\n"
        f"Is True Lyapunov: {validation}.\n"
        f"# epoch: {last_epoch}. "
        f"# total samples: {last_approx.num_samples}. "
        f"# total regions: {len(last_approx)}. "
        f"Time: {cegus_time_usage:.3f}s")

    if out_dir is None or len(last_approx) >= 2*10**4:  # Skip plotting
        return cand

    fig_err = viz_region_stats(mod.X_LIM, last_approx, cex_regions)
    f_name = f"err-cover-{'x'.join(str(n) for n in init_part)}.png"
    f_path = out_dir / f_name
    fig_err.savefig(f_path)  # type: ignore
    plt.clf()
    print(f'The plot is saved to "{f_path}".')

    if mod.X_DIM != 2:  # Support plotting 2D systems only
        return cand

    fig_cover = plt.figure()
    fig_cover.suptitle(f"Cover of ROI for {mod.__name__}")
    ax = fig_cover.add_subplot()
    ax.set_title(
        f"CEGuS Status: {cegus_status}.\n"
        f"Is True Lyapunov: {str(validation)}.\n"
        f"# epoch: {last_epoch}. "
        f"# total samples: {last_approx.num_samples}. "
        f"Time: {cegus_time_usage:.3f}s")
    viz_regions_2d(ax, mod, last_approx, cex_regions)
    if cand:
        viz_basin_2d(ax, mod, learner.get_candidate())
    ax.set_aspect("equal")
    fig_cover.tight_layout()
    f_name = f"cegus-valid_regions-{'x'.join(str(n) for n in init_part)}.png"
    f_path = out_dir / f_name
    plt.savefig(f_path)
    plt.clf()
    print(f'The plot is saved to "{f_path}".')

    return cand
