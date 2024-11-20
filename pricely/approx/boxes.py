from dreal import Expression as Expr, Formula, sqrt as Sqrt, Variable, logical_and  # type:ignore
import numpy as np
from typing import Callable, Hashable, Sequence, Tuple, Union, overload

from pricely.cegus_lyapunov import NDArrayFloat, PLyapunovCandidate, PApproxDynamic, PLocalApprox, split_region
from pricely.utils import gen_equispace_regions


class ConstantApprox(PLocalApprox):
    def __init__(self, x_region: NDArrayFloat, u: NDArrayFloat, y: NDArrayFloat, lip: float) -> None:
        assert len(x_region) == 3
        assert len(y) == x_region.shape[1]
        assert lip >= 0.0 and np.isfinite(lip)
        self._x_region = x_region
        self._u = u
        self._y = y
        self._lip = lip

    @property
    def _x(self) -> NDArrayFloat:
        return self._x_region[0]
    @property
    def _lb(self) -> NDArrayFloat:
        return self._x_region[1]
    @property
    def _ub(self) -> NDArrayFloat:
        return self._x_region[2]
    
    @property
    def num_approxes(self) -> int:
        return 1
    
    @property
    def domain_diameter(self) -> float:
        return float(np.linalg.norm(self._ub - self._lb))
    
    @property
    def x_witness(self) -> NDArrayFloat:
        dist_ub = np.sum(self._ub - self._x)
        dist_lb = np.sum(self._x - self._lb)
        if dist_lb <= dist_ub:
            return (self._x + self._ub) / 2
        else:
            return (self._lb + self._x) / 2

    def in_domain_repr(self) -> Hashable:
        return "Box", self._lb.tobytes(), self._ub.tobytes()

    def in_domain_pred(self, x_vars: Sequence[Variable]) -> Formula:
        return logical_and(
            *(logical_and(x >= lb, x <= ub)
            for x, lb, ub in zip(x_vars, self._lb, self._ub)))

    def error_bound_expr(self, x_vars: Sequence[Variable], u_vars: Sequence[Variable], k: int) -> Expr:
        return self._lip * Sqrt(
            sum((x - v)**2 for x, v in zip(x_vars, self._x)) +
            sum((u - v)**2 for u, v in zip(u_vars, self._u)))

    def func_exprs(self, x_vars: Sequence[Variable], u_vars: Sequence[Variable], k: int) -> Sequence[Expr]:
        return [Expr(v) for v in self._y]  # Constant value approximation


class AxisAlignedBoxes(PApproxDynamic):
    def __init__(
            self,
            x_lim: NDArrayFloat, u_roi: NDArrayFloat,
            x_regions: NDArrayFloat, u_values: NDArrayFloat,
            f_bbox: Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat],
            lip_bbox: Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat]) -> None:
        assert x_lim.ndim == 2 and x_lim.shape[0] == 2
        assert u_roi.ndim == 2 and u_roi.shape[0] == 2
        assert x_regions.ndim == 3 and x_regions.shape[1] == 3 and x_regions.shape[2] == x_lim.shape[1]
        assert u_values.ndim == 2 and u_values.shape[1] == u_roi.shape[1]
        assert len(x_regions) == len(u_values)

        self._x_lim = x_lim
        self._u_roi = u_roi
        self._x_regions = x_regions
        self._u_values = u_values.reshape((len(x_regions), -1))  # ensure to be 2D
        self._f_bbox =  f_bbox
        self._lip_bbox = lip_bbox

        self._y_values = f_bbox(self.x_values, self.u_values)
        self._lip_values = lip_bbox(x_regions, u_roi)
        assert len(self._x_regions) == len(self._y_values) and len(self._x_regions) == len(self._lip_values)

    @property
    def x_values(self) -> NDArrayFloat:
        "Get sampled states"
        return self._x_regions[:, 0, :]

    @property
    def x_regions(self) -> NDArrayFloat:
        return self._x_regions

    @property
    def u_values(self) -> NDArrayFloat:
        "Get sampled inputs"
        return self._u_values

    @property
    def y_values(self) -> NDArrayFloat:
        "Get sampled outputs from black-box dynamics"
        return self._y_values

    @property
    def x_dim(self) -> int:
        return self._x_lim.shape[1]

    @property
    def u_dim(self) -> int:
        return self._u_roi.shape[1]

    def __len__(self) -> int:
        return len(self._x_regions)

    @overload
    def __getitem__(self, item: int) -> ConstantApprox: ...

    @overload
    def __getitem__(self, item: slice) -> Sequence[ConstantApprox]: ...

    def __getitem__(self, item: Union[int, slice]) -> Union[ConstantApprox, Sequence[ConstantApprox]]:
        if isinstance(item, slice):
            raise NotImplementedError("Subclass disallows slicing")
        assert isinstance(item, int)
        return ConstantApprox(
            x_region=self._x_regions[item],
            u=self.u_values[item],
            y=self.y_values[item],
            lip=self._lip_values[item])

    def add(self, cex_boxes: Sequence[Tuple[int, NDArrayFloat]], cand: PLyapunovCandidate) -> None:
        new_regions = self._split_regions(cex_boxes)

        new_x_values = new_regions[:, 0, :]
        new_u_values = cand.ctrl_values(new_x_values)
        new_y_values = self._f_bbox(new_x_values, new_u_values)
        new_lip_values = self._lip_bbox(new_regions, self._u_roi)

        self._x_regions = np.concatenate((self._x_regions, new_regions), axis=0)
        self._u_values = np.row_stack((self._u_values, new_u_values))
        self._y_values = np.row_stack((self._y_values, new_y_values))
        self._lip_values = np.concatenate((self._lip_values, new_lip_values))
        assert len(self._x_regions) == len(self._u_values) and len(self._x_regions) == len(self._y_values)


    def _split_regions(
            self,
            sat_regions: Sequence[Tuple[int, NDArrayFloat]]) -> NDArrayFloat:
        assert self._x_regions.shape[1] == 3
        x_values, x_lbs, x_ubs = self._x_regions[:, 0], self._x_regions[:, 1], self._x_regions[:, 2]
        new_cexs, new_lbs, new_ubs = [], [], []
        for j, box_j in sat_regions:
            res = split_region(self._x_regions[j], box_j)
            if res is None:
                # FIXME Why the cex is so close to the sampled value?
                # continue
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
        x_values = np.row_stack(new_cexs)
        x_lbs = np.row_stack(new_lbs)
        x_ubs = np.row_stack(new_ubs)
        return np.stack((x_values, x_lbs, x_ubs), axis=1)


def test_approx():
    X_LIM = np.array([
        [-1, -2, -3],
        [+1, +2, +3]])
    U_ROI = np.array([
        [-0.5, -2.5],
        [+0.5, +2.5]])

    def f_bbox(x: NDArrayFloat, u: NDArrayFloat) -> NDArrayFloat:
        return x

    def lip_bbox(x_regions: NDArrayFloat, u_roi: NDArrayFloat) -> NDArrayFloat:
        return np.ones((len(x_regions)))

    x_regions =gen_equispace_regions([2, 3, 4], X_LIM)
    u_values = np.zeros((len(x_regions), 2))
    approx = AxisAlignedBoxes(
        X_LIM, U_ROI, x_regions, u_values, f_bbox, lip_bbox)

if __name__ == "__main__":
    test_approx()
