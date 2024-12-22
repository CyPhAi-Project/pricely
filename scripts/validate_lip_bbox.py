import numpy as np
from typing import Sequence

from pricely.utils import cartesian_prod, gen_equispace_regions, pretty_sup
from scripts.utils_plotting_2d import CatchTime

NCOLS = 120


def est_lip_bound(mod, x_part: Sequence[int]) -> np.ndarray:
    x_lim = mod.X_LIM
    assert x_lim.ndim == 2 and x_lim.shape[0] == 2
    x_dim = mod.X_DIM

    # Generate evenly-spaced points
    axes_cuts = (np.linspace(
        x_lim[0, i], x_lim[1, i], x_part[i]+1) for i in range(x_dim))
    pts_shape = tuple(n+1 for n in x_part) + (x_dim,)
    x_pts = cartesian_prod(
        *axes_cuts).reshape(pts_shape)
    y_pts = mod.f_bbox(x_pts.reshape((-1, x_dim))).reshape(pts_shape)

    all_bvecs = cartesian_prod(*np.repeat([[0, 1]], x_dim, axis=0))
    slice_tups = [
        tuple(slice(1, None) if b else slice(0, -1) for b in bvec)
        for bvec in all_bvecs]
    x_ref_pts = x_pts[slice_tups[0]].reshape((-1, x_dim))
    y_ref_pts = y_pts[slice_tups[0]].reshape((-1, x_dim))

    lip_lbs = np.zeros(np.prod(x_part))
    for sel in slice_tups[1:]:
        x_other_pts = x_pts[sel].reshape((-1, x_dim))
        x_dists = np.linalg.norm(x_other_pts - x_ref_pts, axis=1)

        y_other_pts = y_pts[sel].reshape((-1, x_dim))
        y_dists = np.linalg.norm(y_other_pts - y_ref_pts, axis=1)
    
        tmp_lip_lbs = np.divide(
            y_dists, x_dists,
            out=np.full(x_dists.shape, np.nan),  # Default nan when Div By 0
            where=~np.isclose(x_dists, np.zeros_like(x_dists)))
        lip_lbs = np.maximum(lip_lbs, tmp_lip_lbs)
    assert np.all(np.isfinite(lip_lbs))
    return lip_lbs


def validate_lip_bbox(mod, parts: Sequence[int], n_jobs: int = 16):
    regions = gen_equispace_regions(parts, mod.X_LIM)

    lip_values = mod.calc_lip_bbox(regions)
    assert len(lip_values) == np.prod(parts)

    lip_lbs = est_lip_bound(mod, parts)
    assert lip_values.shape == lip_lbs.shape

    min_lb_idx = np.argmin(lip_lbs)
    max_lb_idx = np.argmax(lip_lbs)
    min_val_idx = np.argmin(lip_values)
    max_val_idx = np.argmax(lip_values)
    min_diff_idx = np.argmin(lip_values - lip_lbs)
    max_diff_idx = np.argmax(lip_values - lip_lbs)
    print(f"Min Estimated vs Provided: {lip_lbs[min_lb_idx]:.3f} <= {lip_values[min_lb_idx]:.3f}")
    print(f"Max Estimated vs Provided: {lip_lbs[max_lb_idx]:.3f} <= {lip_values[max_lb_idx]:.3f}")
    print(f"Estimated vs Min Provided: {lip_lbs[min_val_idx]:.3f} <= {lip_values[min_val_idx]:.3f}")
    print(f"Estimated vs Max Provided: {lip_lbs[max_val_idx]:.3f} <= {lip_values[max_val_idx]:.3f}")
    print(f"Best diff: {lip_lbs[min_diff_idx]:.3f} <= {lip_values[min_diff_idx]:.3f}")
    print(f"Worst diff: {lip_lbs[max_diff_idx]:.3f} <= {lip_values[max_diff_idx]:.3f}")
    assert np.all(lip_lbs <= lip_values)


def execute(mod, max_num_samples: int = 10**6, n_jobs: int = 16):
    num_cuts = int(np.floor(10**(np.log10(max_num_samples)/mod.X_DIM))) - 1
    print(f" Validate local Lipschitz bounds with {num_cuts+1}{pretty_sup(mod.X_DIM)} samples ".center(NCOLS, "="))
    with CatchTime():
        validate_lip_bbox(mod, [num_cuts]*mod.X_DIM)
