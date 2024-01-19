from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
import numpy as np
from typing import Callable


def add_level_sets(
        ax: Axes, lya_func: Callable[[np.ndarray], np.ndarray],
        x_roi: np.ndarray, norm_ub: float=np.inf,
        num_steps: int = 250,
        colors: str = "r"
        ):
    assert x_roi.shape == (2, 2)
    x_arr = np.linspace(x_roi[0], x_roi[1], num_steps).T
    X, Y = np.meshgrid(*x_arr)
    x_values = np.column_stack((X.ravel(), Y.ravel()))
    Z = lya_func(x_values).reshape(X.shape)
    if not np.isfinite(norm_ub):
        ax.contour(X, Y, Z, colors=colors)
    else:
        ang_arr = np.linspace(0.0, 2*np.pi, num_steps)
        sel_values = norm_ub*np.column_stack((np.cos(ang_arr), np.sin(ang_arr)))
        lya_values = lya_func(sel_values)
        ax.contour(X, Y, Z, levels=[np.min(lya_values), np.max(lya_values)], colors=colors)


def add_valid_regions(ax: Axes, num_iters: int, time_usage: float, regions: np.ndarray, cex_regions):
    # Contain ref value, lower bound, and upper bound
    assert regions.shape[1] == 3
    assert regions.shape[2] == 2   # Only for 2D states
    ax.set_title(f"# iteration: {num_iters}. "
                 f"# total samples: {len(regions)}. "
                 f"Time: {time_usage:.3f}s")
    x_values, x_lbs, x_ubs = \
        regions[:, 0], regions[:, 1], regions[:, 2]

    num_samples = len(x_values)
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
        w, h = x_ubs[j] - x_lbs[j]
        rect = Rectangle(x_lbs[j], w, h, fill=True,
                         edgecolor='black', facecolor=facecolor, alpha=0.3)
        ax.add_patch(rect)
