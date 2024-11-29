from matplotlib.patches import Circle
import numpy as np
from pathlib import Path
from typing import Optional

from phaseportrait import PhasePortrait2D

from pricely.candidates import PLyapunovCandidate
from scripts.utils_plotting_2d import add_level_sets


def main(mod, cand: Optional[PLyapunovCandidate], out_dir: Path):
    def f_2d(x0, x1):
        x = np.array([[x0, x1]])
        dx = mod.f_bbox(x).squeeze()
        return dx[0], dx[1]

    XLIM = mod.X_NORM_UB*1.25
    Plot2D = PhasePortrait2D(f_2d, [[-XLIM, XLIM], [-XLIM, XLIM]], Title="", MeshDim=9, xlabel=r"", ylabel=r"", color="binary_r", odeint_method="euler")
    fig, ax = Plot2D.plot()
    ax.grid(False)
    ax.set_aspect("equal")

    for r in [mod.X_NORM_UB]:
        ax.add_patch(Circle((0, 0), r, color='r', fill=False))
        t = ax.text(r*np.cos(-2*np.pi/4), r*np.sin(-2*np.pi/4), f"{r}", horizontalalignment='center', verticalalignment='center')
    ax.add_patch(Circle((0, 0), mod.X_NORM_LB, color='r', fill=False))

    angs = np.linspace(0, 2*np.pi, 1000)
    x_values = mod.X_NORM_UB*np.column_stack((np.cos(angs), np.sin(angs)))

    if hasattr(mod, "nnet_lya"):
        level_ub = np.min(mod.nnet_lya(x_values))
        add_level_sets(ax, mod.nnet_lya, level_ub, colors="g", linestyles="dashed")

    if cand:
        level_ub = float(np.min(cand.lya_values(x_values)))
        add_level_sets(ax, cand.lya_values, level_ub, colors="b", linestyles="solid")

    f_name = f"phase_portrait.png"
    f_path = out_dir / f_name
    fig.savefig(f_path)  # type: ignore
