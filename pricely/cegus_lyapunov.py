import abc
from dreal import Expression as Expr, Variable  # type: ignore
import itertools
import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm
from typing import Callable, Optional, Protocol, Sequence, Tuple, Union
import warnings


NDArrayFloat = NDArray[np.float_]

class PLyapunovLearner(Protocol):
    @abc.abstractmethod
    def fit_loop(self, X: NDArrayFloat, y: NDArrayFloat, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def lya_expr(self, x_vars: Sequence[Variable]) -> Expr:
        raise NotImplementedError

    @abc.abstractmethod
    def lya_values(self, x_values: NDArrayFloat) -> NDArrayFloat:
        raise NotImplementedError

    @abc.abstractmethod
    def ctrl_exprs(self, x_vars: Sequence[Variable]) -> Sequence[Expr]:
        raise NotImplementedError

    @abc.abstractmethod
    def ctrl_values(self, x_values: NDArrayFloat) -> NDArrayFloat:
        raise NotImplementedError

    def find_level_ub(self, x_roi: NDArrayFloat) -> float:
        """ Find a sublevel set covering the region of interest
        Heuristically find the max value in the region of interest.
        List all vertices of ROI and pick the max level value.
        Provide it as the upper bound of the level value.
        """
        assert len(x_roi) == 2
        x_dim = x_roi.shape[1]
        x_lb, x_ub= x_roi

        # XXX This generates 2^x_dim vertices.
        if x_dim > 16:
            warnings.warn(f"Generating 2^{x_dim} = {2**x_dim} vertices of the unit cube."
                            "This may take a while.")
        unit_cube = np.fromiter(itertools.product((0.0, 1.0), repeat=x_dim),
                                dtype=np.dtype((np.float_, x_dim)))
        vertices = x_lb + unit_cube * (x_ub - x_lb)
        return float(np.max(self.lya_values(vertices)))
    
    @abc.abstractmethod
    def find_sublevel_set_and_box(self, x_roi: NDArrayFloat) -> Tuple[float, ArrayLike]:
        raise NotImplementedError


class PLyapunovVerifier(Protocol):
    @abc.abstractmethod
    def set_lyapunov_candidate(self, learner: PLyapunovLearner):
        raise NotImplementedError

    @abc.abstractmethod
    def find_cex(
            self, x_region_j: NDArrayFloat,
            u_j: NDArrayFloat, dxdt_j: NDArrayFloat,
            lip_expr: Union[float, Expr]) -> Optional[NDArrayFloat]:
        raise NotImplementedError


def cegus_lyapunov(
        learner: PLyapunovLearner,
        verifier: PLyapunovVerifier,
        x_regions: NDArrayFloat,
        f_bbox: Callable[[NDArrayFloat], NDArrayFloat],
        lip_bbox: float,
        max_epochs: int = 10,
        max_iter_learn: int = 10):
    null_arr = np.array([])
    def new_f_bbox(x, u):
        return f_bbox(x)

    return cegus_lyapunov_control(
        learner=learner,
        verifier=verifier,
        x_regions=x_regions,
        u_values=null_arr.reshape(len(x_regions), 0),
        f_bbox=new_f_bbox, lip_bbox=lip_bbox,
        max_epochs=max_epochs, max_iter_learn=max_iter_learn
    )


def cegus_lyapunov_control(
        learner: PLyapunovLearner,
        verifier: PLyapunovVerifier,
        x_regions: NDArrayFloat,
        u_values: NDArrayFloat,
        f_bbox: Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat],
        lip_bbox: ArrayLike,
        max_epochs: int = 10,
        max_iter_learn: int = 10):
    assert x_regions.shape[1] == 3
    assert len(x_regions) == len(u_values)
    assert max_epochs > 0
    # Initial sampled x, u values and the constructed set cover
    x_values = x_regions[:, 0]

    outer_pbar = tqdm(
        iter(range(1, max_epochs + 1)),
        desc="Outer", ascii=True, postfix={"#Valid": 0, "#Total": len(x_values)})
    cex_regions = []
    obj_values = []
    for epoch in outer_pbar:
        dxdt_values = f_bbox(x_values, u_values)

        objs = learner.fit_loop(x_values, dxdt_values,
                               max_epochs=max_iter_learn, copy=False)
        obj_values.extend(objs)

        # Verify Lyapunov condition
        assert len(x_regions) == len(dxdt_values)

        verifier.set_lyapunov_candidate(learner)
        cex_regions.clear()
        for j in tqdm(range(len(x_regions)),
                      desc=f"Verify at {epoch}", ascii=True, leave=False):
            result = verifier.find_cex(
                x_region_j=x_regions[j],
                u_j=u_values[j],
                dxdt_j=dxdt_values[j], lip_expr=lip_bbox)
            if result is not None:
                cex_regions.append((j, result))

        outer_pbar.set_postfix({"#Valid": len(x_regions)-len(cex_regions), "#Total": len(x_regions)})
        if len(cex_regions) == 0:
            # Lyapunov function candidate passed
            return epoch, x_regions, cex_regions
        # else:
        # NOTE splitting regions may also modified the input arrays
        x_regions = split_regions(x_regions, cex_regions)

        x_values = x_regions[:, 0]
        u_values = learner.ctrl_values(x_values)
    outer_pbar.close()

    tqdm.write(f"Cannot find a Lyapunov function in {max_epochs} iterations.")
    return max_epochs, x_regions, cex_regions


def split_regions(
        x_regions: NDArrayFloat,
        sat_regions: Sequence[Tuple[int, NDArrayFloat]]) -> NDArrayFloat:
    assert x_regions.shape[1] == 3
    x_values, x_lbs, x_ubs = x_regions[:, 0], x_regions[:, 1], x_regions[:, 2]
    new_cexs, new_lbs, new_ubs = [], [], []
    for j, box_j in sat_regions:
        res = split_region(x_regions[j], box_j)
        if res is None:
            continue
            raise RuntimeError("Sampled state is inside cex box")
        cex, cut_axis, cut_value = res
        # Shrink the bound for the existing sample
        # Copy the old bounds
        cex_lb, cex_ub = x_lbs[j].copy(), x_ubs[j].copy()
        if cex[cut_axis] < x_values[j][cut_axis]:
            # Increase lower bound for old sample
            x_lbs[j][cut_axis] = cut_value
            cex_ub[cut_axis] = cut_value  # Decrease upper bound for new sample
        else:
            assert cex[cut_axis] > x_values[j][cut_axis]
            # Decrease upper bound for old sample
            x_ubs[j][cut_axis] = cut_value
            cex_lb[cut_axis] = cut_value  # Increase lower bound for new sample
        new_cexs.append(cex)
        new_lbs.append(cex_lb)
        new_ubs.append(cex_ub)
    x_values = np.row_stack((x_values, *new_cexs))
    x_lbs = np.row_stack((x_lbs, *new_lbs))
    x_ubs = np.row_stack((x_ubs, *new_ubs))
    return np.stack((x_values, x_lbs, x_ubs), axis=1)


def split_region(
    region: NDArrayFloat,
    box: NDArrayFloat
) -> Optional[Tuple[NDArrayFloat, np.intp, float]]:
    assert region.shape[0] == 3
    cex_lb, cex_ub = box
    x, lb, ub = region
    if np.all(np.logical_and(cex_lb <= x, x <= cex_ub)):
        return None
        raise RuntimeError("Sampled state is inside cex box")
    # Clip the cex bounds to be inside the region.
    cex_lb = cex_lb.clip(min=lb, max=ub)
    cex_ub = cex_ub.clip(min=lb, max=ub)
    cex = (cex_lb + cex_ub) / 2.0

    # Decide the separator between the existing sample and the cex box
    # Choose the dimension with the max distance to cut
    axes_aligned_dist = (cex_lb - x).clip(min=0.0) + (x - cex_ub).clip(min=0.0)
    cut_axis = np.argmax(axes_aligned_dist)

    if x[cut_axis] < cex_lb[cut_axis]:
        box_edge = cex_lb[cut_axis]
    else:
        assert x[cut_axis] > cex_lb[cut_axis]
        box_edge = cex_ub[cut_axis]
    cut_value = (x[cut_axis] + box_edge) / 2.0
    return cex, cut_axis, cut_value
