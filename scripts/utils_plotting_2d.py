from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
import numpy as np
from scipy.spatial import Delaunay
import time
from typing import Callable

from pricely.approx.boxes import AxisAlignedBoxes
from pricely.approx.simplices import SimplicialComplex
from pricely.cegus_lyapunov import PApproxDynamic


def add_level_sets(
        ax: Axes, lya_func: Callable[[np.ndarray], np.ndarray],
        level_ub: float=np.inf,
        num_steps: int = 250,
        colors: str = "r",
        linestyles: str = "solid"):
    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    x_arr = np.linspace(*x_lim, num_steps)
    y_arr = np.linspace(*y_lim, num_steps)
    X, Y = np.meshgrid(x_arr, y_arr)
    x_values = np.column_stack((X.ravel(), Y.ravel()))
    Z = lya_func(x_values).reshape(X.shape)
    if np.isfinite(level_ub):
        levels=[level_ub]
    else:
        sel_values = np.vstack([
            np.column_stack((x_arr, np.full_like(x_arr, y_lim[0]))),
            np.column_stack((np.full_like(y_arr, x_lim[0]), y_arr))])
        lya_values = lya_func(sel_values)
        levels=np.linspace(0.0, np.min(lya_values), 5)

    ax.contour(X, Y, Z, levels, colors=colors, linestyles=linestyles)


def add_valid_regions(ax: Axes, approx: PApproxDynamic, cex_regions):
    if isinstance(approx, AxisAlignedBoxes):
        _add_axis_aligned_boxes(ax, approx._x_regions, cex_regions)
    elif isinstance(approx, SimplicialComplex):
        _add_simplicial_complex(ax, approx._triangulation, cex_regions)
    else:
        raise NotImplementedError


def _add_simplicial_complex(ax: Axes, tri: Delaunay, cex_regions):
    mask = np.ones(len(tri.simplices), dtype=bool)
    unverified = [j for j, _ in cex_regions]
    mask[unverified] = False
    # Color unverifed regions
    ax.tripcolor(tri.points[:,0], tri.points[:,1], triangles=tri.simplices, mask=mask, facecolors=np.ones(len(tri.simplices)), cmap="Pastel1")
    ax.triplot(tri.points[:,0], tri.points[:,1], triangles=tri.simplices, lw=0.5, alpha=0.7)


def _add_axis_aligned_boxes(ax: Axes, regions: np.ndarray, cex_regions):
    # Contain ref value, lower bound, and upper bound
    assert regions.shape[1] == 3
    assert regions.shape[2] == 2   # Only for 2D states
    x_values, x_lbs, x_ubs = \
        regions[:, 0], regions[:, 1], regions[:, 2]

    num_regions = len(regions)
    sat_region_iter = (k for k, _ in cex_regions)
    k = next(sat_region_iter, None)
    for j in range(num_regions):
        if j == k:
            k = next(sat_region_iter, None)
            facecolor = "red"
        else:
            facecolor = "white"
        w, h = x_ubs[j] - x_lbs[j]
        rect = Rectangle(x_lbs[j], w, h, fill=True,
                         edgecolor='black', facecolor=facecolor, alpha=0.3)
        ax.add_patch(rect)


def plot_cegar_result(
        ax: Axes, num_iters: int,
        init_regions: np.ndarray, cex_regions: np.ndarray):
    x_values, x_lbs, x_ubs = \
        init_regions[:, 0], init_regions[:, 1], init_regions[:, 2]
    for j in range(len(init_regions)):
        w, h = x_ubs[j] - x_lbs[j]
        rect = Rectangle(x_lbs[j], w, h, fill=True,
                         edgecolor='black', facecolor="white", alpha=0.3)
        ax.add_patch(rect)
    
    x_values, x_lbs, x_ubs = \
        cex_regions[:, 0], cex_regions[:, 1], cex_regions[:, 2]
    for j in range(len(cex_regions)):
        w, h = x_ubs[j] - x_lbs[j]
        rect = Rectangle(x_lbs[j], w, h, fill=True,
                         edgecolor='black', facecolor="red", alpha=0.3)
        ax.add_patch(rect)


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
