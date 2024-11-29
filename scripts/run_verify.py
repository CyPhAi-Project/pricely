import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from scripts.utils_plotting_2d import CatchTime, plot_cegar_result

from pricely.gen_cover import gen_init_cover
from pricely.cegus_lyapunov import verify_lyapunov
from pricely.learner.mock import MockQuadraticLearner


def main(mod, out_dir: Optional[Path]=None, max_epochs: int=200):
    assert hasattr(mod, "KNOWN_QUAD_LYA"), \
        "Must provide a known quadratic Lyapunov function."
    lip_cap = getattr(mod, "LIP_CAP", np.inf)

    timer = CatchTime()

    init_part = [40]*mod.X_DIM

    print(" Generate initial samples and cover ".center(80, "="))
    with timer:
        x_regions = gen_init_cover(
            abs_roi_ub=mod.X_LIM[1],
            f_bbox=mod.f_bbox,
            lip_bbox=mod.calc_lip_bbox,
            lip_cap=lip_cap,
            abs_lb=mod.ABS_X_LB,
            init_part=init_part)
    init_x_regions = x_regions.copy()

    print(" Run CEGAR verification ".center(80, "="))
    with timer:
        mock_learner = MockQuadraticLearner(mod.KNOWN_QUAD_LYA)
        # Set predefined Lyapunov candidate
        last_epoch, num_regions, cex_regions = \
            verify_lyapunov(
                mock_learner,
                mod.X_LIM,
                mod.ABS_X_LB,
                x_regions,
                mod.f_bbox, mod.calc_lip_bbox,
                max_epochs=max_epochs)
    cegar_status = "Found" if len(cex_regions) == 0 else "Can't Find" if last_epoch < max_epochs else "Reach epoch limit"
    cegar_time_usage = timer.elapsed

    if out_dir is None:  # Skip plotting
        return
    if mod.X_DIM != 2:  # Support plotting 2D systems only
        return
    print(" Plotting verified regions ".center(80, "="))
    plt.gca().set_xlim(*(1.125*mod.X_LIM[:, 0]))
    plt.gca().set_ylim(*(1.125*mod.X_LIM[:, 1]))

    plt.gca().set_title(
        f"CEGAR Status: {cegar_status}.\n"
        f"# epoch: {last_epoch}. "
        f"# total samples: {num_regions}. "
        f"Time: {cegar_time_usage:.3f}s")
    plot_cegar_result(plt.gca(), last_epoch, init_x_regions, cex_regions)

    plt.gca().set_aspect("equal")
    plt.tight_layout()
    cap_str = f"-cap_{int(lip_cap)}" if np.isfinite(lip_cap) else ""

    f_name = f"verify-valid_regions-{'x'.join(str(n) for n in init_part)}{cap_str}.png"
    f_path = out_dir / f_name
    plt.savefig(f_path)
    plt.clf()
    print(f'The plot is saved to "{f_path}".')
