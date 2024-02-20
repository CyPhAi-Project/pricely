import numpy as np
from numpy.typing import ArrayLike
from typing import Callable, Optional, Sequence, Tuple

from pricely.cegus_lyapunov import NDArrayFloat
from pricely.utils import cartesian_prod


def gen_stars(x_roi: NDArrayFloat, hint_part: Sequence[int]) -> Tuple[NDArrayFloat, NDArrayFloat]:
    assert x_roi.shape == (2, len(hint_part))
    assert np.all(x_roi[0] < x_roi[1])
    x_dim = x_roi.shape[1]
    x_lb, x_ub = x_roi[0], x_roi[1]
    xi_cuts = [np.linspace(
        x_lb[i], x_ub[i], hint_part[i]+1) for i in range(x_dim)]
    # Mid point for each rectangle
    xi_pts = [(c_i[1:] + c_i[:-1]) / 2.0 for c_i in xi_cuts]
    x_centers = cartesian_prod(*xi_pts)
    x_vecs = (x_ub - x_lb) / (2*np.asarray(hint_part))
    return x_centers, x_vecs


def gen_init_cover(
        abs_roi_ub: NDArrayFloat,
        f_bbox: Callable[[NDArrayFloat], NDArrayFloat],
        lip_bbox: Callable[[NDArrayFloat], NDArrayFloat],
        lip_cap: float = np.inf,
        abs_lb: ArrayLike = 2**-10,
        init_part: Optional[Sequence[int]] = None) -> NDArrayFloat:
    """ Generate an initial cover in which each region is a subset of the theoretical upperbound. """
    x_dim = len(abs_roi_ub)
    abs_lb_arr = np.asfarray(abs_lb)
    assert np.all(abs_roi_ub > 0.0)
    assert np.all(abs_lb_arr > 0.0)
    assert np.all(abs_roi_ub > 2*abs_lb_arr)
    if init_part is None:
        init_part = [3]*x_dim
    assert len(init_part) == x_dim

    init_x_centers, init_x_vecs = gen_stars(
        np.row_stack((-abs_roi_ub, abs_roi_ub)), init_part)
    all_regions = []
    work_list = [(init_x_centers, init_x_vecs)]
    while work_list:
        x_centers, x_vecs = work_list.pop()
        if np.any(x_vecs <= 2**-20):
            # Found a region that is too small
            raise RuntimeError("A region in the initial cover is too small.")
        # Sample the outputs at selected reference points
        dxdt_values = f_bbox(x_centers)
        x_lbs = x_centers - x_vecs
        x_ubs = x_centers + x_vecs

        norm_y_values = np.linalg.norm(dxdt_values, ord=2, axis=1)
        radius: float = np.linalg.norm(x_vecs, ord=2).item()
        lip_ubs = lip_bbox(np.stack((x_centers, x_lbs, x_ubs), axis=1))

        # Ignore regions near equilibrium
        near_origin = np.logical_and(
            np.all(x_lbs >= -abs_lb_arr, axis=1),
            np.all(x_ubs <= +abs_lb_arr, axis=1))
        
        # Also igore when Lipschitz constant is too large and the region is small
        large_lip = (lip_ubs > lip_cap)
        may_ignore = np.logical_or(near_origin, large_lip)    

        may_prove = np.logical_and(
            norm_y_values - lip_ubs*radius >= 0.0,
            np.logical_not(may_ignore))
        # Add regions that may be provable and not ignored.
        new_regions = np.stack(
            (x_centers[may_prove], x_lbs[may_prove], x_ubs[may_prove]), axis=1)
        all_regions.append(new_regions)

        if np.all(np.logical_or(may_prove, may_ignore)):
            # The current regions may pass verification
            continue

        # else: refine the regions that is neither provable nor ignored
        must_refine = np.logical_not(np.logical_or(may_prove, may_ignore))
        for lb, ub in \
                zip(x_lbs[must_refine], x_ubs[must_refine]):
            roi = np.row_stack((lb, ub))
            # TODO Refine regions using Lipschitz constant and sampled values
            hint_part = [2]*x_dim  # Divide into 2^x_dim regions
            work_list.append(gen_stars(roi, hint_part))

    if len(all_regions) == 0:
        raise RuntimeError("The construted cover is an empty set.")
    return np.concatenate(all_regions)
