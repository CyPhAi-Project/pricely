from dreal import Expression as Expr, Formula, sqrt as Sqrt, Variable, logical_and  # type: ignore
import numpy as np
from scipy.spatial import Delaunay, QhullError
from scipy.spatial.distance import pdist
from typing import Callable, Sequence, Tuple, Union, overload

from pricely.cegus_lyapunov import NDArrayFloat, PApproxDynamic, PLocalApprox, PLyapunovCandidate


class DatasetApproxBase(PLocalApprox):
    def __init__(
            self,
            x_values: NDArrayFloat, u_values: NDArrayFloat, y_values: NDArrayFloat,
            lip: float) -> None:
        self._x_values = x_values
        self._x_lbs = x_values.min(axis=0)
        self._x_ubs = x_values.max(axis=0)
        self._u_values = u_values
        self._y_values = y_values
        self._lip = lip

    @property
    def x_witness(self) -> NDArrayFloat:
        return self._x_values.mean(axis=0)

    def in_domain_pred(self, x_vars: Sequence[Variable]) -> Formula:
        return logical_and(
            *(x_i >= lb_i for x_i, lb_i in zip(x_vars, self._x_lbs)),
            *(x_i <= ub_i for x_i, ub_i in zip(x_vars, self._x_ubs)))


class BarycentricMixin:
    def __init__(
            self, trans_barycentric: NDArrayFloat, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        x_dim = trans_barycentric.shape[1]
        assert trans_barycentric.shape[0] == x_dim + 1
        assert np.alltrue(np.isfinite(trans_barycentric))

        self._t_mat = trans_barycentric[:x_dim, :x_dim]
        self._r_vec = trans_barycentric[x_dim, :x_dim]

    def _get_barycentric_exprs(self, x_vars: Sequence[Variable]) -> Sequence[Expr]:
        ret = (self._t_mat @ (x_vars - self._r_vec)).tolist()
        ret.append(1.0 - sum(ret))
        return ret

    def in_simplex_pred(self, x_vars: Sequence[Variable]) -> Formula:
        barycentric_exprs = self._get_barycentric_exprs(x_vars)
        return logical_and(
            *(lambda_k >= 0 for lambda_k in barycentric_exprs),
            *(lambda_k <= 1 for lambda_k in barycentric_exprs))


class LinearInterpolaton(BarycentricMixin, DatasetApproxBase):
    def __init__(
            self, trans_barycentric: NDArrayFloat,
            x_values: NDArrayFloat, u_values: NDArrayFloat, y_values: NDArrayFloat,
            lip: float) -> None:
        super().__init__(trans_barycentric, x_values, u_values, y_values, lip)

    @property
    def num_approxes(self) -> int:
        return 1

    def in_domain_pred(self, x_vars: Sequence[Variable]) -> Formula:
        return logical_and(
            self.in_simplex_pred(x_vars),
            super().in_domain_pred(x_vars)) 

    def func_exprs(self, x_vars: Sequence[Variable], u_vars: Sequence[Variable], k: int) -> Sequence[Expr]:
        barycentric_exprs = self._get_barycentric_exprs(x_vars)
        return np.sum(barycentric_exprs * self._y_values.T, axis=0)

    def error_bound_expr(self, x_vars: Sequence[Variable], u_vars: Sequence[Variable], k: int) -> Expr:
        barycentric_exprs = self._get_barycentric_exprs(x_vars)
        norm_exprs = [Sqrt(sum((x_vars - xv)**2) + sum((u_vars - uv)**2))
                      for xv, uv in zip(self._x_values, self._u_values)]
        return self._lip * sum(b*n for b, n in zip(barycentric_exprs, norm_exprs))


class AnyBoxConstant(DatasetApproxBase):
    """
    Use any of the sampled outputs for approximation
    Use a bounding box for the region covering the samples
    """
    def __init__(
            self,
            x_values: NDArrayFloat, u_values: NDArrayFloat, y_values: NDArrayFloat,
            lip: float) -> None:
        super().__init__(x_values, u_values, y_values, lip)

    @property
    def num_approxes(self) -> int:
        return len(self._x_values)
    
    @property
    def domain_diameter(self) -> float:
        return float(np.linalg.norm(self._x_ubs - self._x_lbs))

    def func_exprs(self, x_vars: Sequence[Variable], u_vars: Sequence[Variable], k: int) -> Sequence[Expr]:
        return [Expr(vi) for vi in self._y_values[k]]

    def error_bound_expr(self, x_vars: Sequence[Variable], u_vars: Sequence[Variable], k: int) -> Expr:
        def l2norm(var_seq, val_seq) -> Expr:
            return Sqrt(sum((x - v)**2 for x, v in zip(var_seq, val_seq)))
        xu_vars = list(x_vars) + list(u_vars)
        xu_values = np.concatenate((self._x_values[k], self._u_values[k]))
        return self._lip * l2norm(xu_vars, xu_values)


class AnySimplexConstant(BarycentricMixin, AnyBoxConstant):
    """ Add constraints of the simplex in addition to the bounding box """
    def __init__(
            self, trans_barycentric: NDArrayFloat,
            x_values: NDArrayFloat, u_values: NDArrayFloat, y_values: NDArrayFloat,
            lip: float) -> None:
        super().__init__(trans_barycentric, x_values, u_values, y_values, lip)

    @property
    def domain_diameter(self) -> float:
        return float(np.max(pdist(self._x_values)))

    def in_domain_pred(self, x_vars: Sequence[Variable]) -> Formula:
        return logical_and(
            self.in_simplex_pred(x_vars),
            super().in_domain_pred(x_vars))


class SimplicialComplex(PApproxDynamic):
    def __init__(
            self,
            x_roi: NDArrayFloat, u_roi: NDArrayFloat,
            x_values: NDArrayFloat, u_values: NDArrayFloat,
            f_bbox: Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat],
            lip_bbox: Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat]
        ) -> None:
        assert x_roi.ndim == 2 and x_roi.shape[0] == 2
        assert u_roi.ndim == 2 and u_roi.shape[0] == 2
        assert x_values.ndim == 2 and x_values.shape[1] == x_roi.shape[1]
        assert u_values.ndim == 2 and u_values.shape[1] == u_roi.shape[1]
        assert len(x_values) == len(u_values)

        self._x_roi = x_roi
        self._u_roi = u_roi
        self._triangulation = Delaunay(points=x_values, incremental=True)
        self._u_values = u_values
        self._f_bbox =  f_bbox
        self._lip_bbox = lip_bbox

        self._y_values = f_bbox(self.x_values, self.u_values)
        self._lips = self._calc_lipschitz()

    @classmethod
    def from_autonomous(
        cls,
        x_roi: NDArrayFloat, x_values: NDArrayFloat,
        f_bbox: Callable[[NDArrayFloat], NDArrayFloat],
        lip_bbox: Callable[[NDArrayFloat], NDArrayFloat]):
        return cls(
            x_roi=x_roi,
            u_roi=np.empty((2, 0)),
            x_values=x_values,
            u_values=np.empty((len(x_values), 0)),
            f_bbox=lambda x, u: f_bbox(x),
            lip_bbox=lambda x, u: lip_bbox(x))

    @property
    def x_values(self) -> NDArrayFloat:
        "Get sampled states"
        return self._triangulation.points

    @property
    def x_regions(self) -> NDArrayFloat:
        "Get regions of states"
        raise NotImplementedError
        # return self._triangulation.points[self._triangulation.simplices]

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
        return self._x_roi.shape[1]

    @property
    def u_dim(self) -> int:
        return self._u_roi.shape[1]

    def __len__(self) -> int:
        return len(self._triangulation.simplices)

    @overload
    def __getitem__(self, item: int) -> DatasetApproxBase: ...

    @overload
    def __getitem__(self, item: slice) -> Sequence[DatasetApproxBase]: ...

    def __getitem__(self, item: Union[int, slice]) -> Union[DatasetApproxBase, Sequence[DatasetApproxBase]]:
        if isinstance(item, slice):
            raise NotImplementedError("Subclass disallows slicing")
        assert isinstance(item, int)

        vertex_idxs = self._triangulation.simplices[item]
        trans = self._triangulation.transform[item]
        if np.any(np.isnan(trans)):  # A degenerate simplex
            return AnyBoxConstant(
                self.x_values[vertex_idxs],
                self.u_values[vertex_idxs],
                self.y_values[vertex_idxs],
                lip=self._lips[item])
        # else:
        return AnySimplexConstant(
            trans,
            self.x_values[vertex_idxs],
            self.u_values[vertex_idxs],
            self.y_values[vertex_idxs],
            lip=self._lips[item])

    def add(self, cex_boxes: Sequence[Tuple[int, NDArrayFloat]], cand: PLyapunovCandidate) -> None:
        cex_regions = np.asfarray([box for j, box in cex_boxes])
        assert cex_regions.shape[1] == 2 and cex_regions.shape[2] == self.x_dim
        new_x_values = cex_regions.mean(axis=1)
        assert np.alltrue(np.isfinite(new_x_values))

        # NOTE This assumes SciPy Delaunay class maintains the order of added points.
        try:
            self._triangulation.add_points(
                new_x_values,
                restart=10*len(new_x_values)>=len(self._triangulation.points))
        except QhullError as e:
            raise RuntimeError("Exception in triangulation.")
        
        new_u_values = cand.ctrl_values(new_x_values)
        new_y_values = self._f_bbox(new_x_values, new_u_values)
        self._u_values = np.row_stack((self._u_values, new_u_values))
        self._y_values = np.row_stack((self._y_values, new_y_values))

        # NOTE Lipschitz constants follow the number of regions instead of samples.
        self._lips = self._calc_lipschitz()

    def _calc_lipschitz(self) -> NDArrayFloat:
        x_simplices = self._triangulation.points[self._triangulation.simplices]
        assert x_simplices.ndim == 3 \
            and x_simplices.shape[1] == self.x_dim + 1 \
            and x_simplices.shape[2] == self.x_dim
        x_boxes = np.stack((
            x_simplices.mean(axis=1),
            x_simplices.min(axis=1),
            x_simplices.max(axis=1)), axis=1)
        return self._lip_bbox(x_boxes, self._u_roi)


def test_approx():
    from pricely.utils import cartesian_prod
    X_ROI = np.array([
        [-1, -2],
        [+1, +2]])
    U_ROI = np.array([
        [-0.5],
        [+0.5]])
    X_DIM = X_ROI.shape[1]

    def f_bbox(x: NDArrayFloat, u: NDArrayFloat) -> NDArrayFloat:
        dxdt = np.empty_like(x)
        dxdt[:, 0] = -x[:, 0]
        dxdt[:, 1] = -0.5*x[:, 1]**3
        return dxdt

    def lip_bbox(x_regions: NDArrayFloat, u_roi: NDArrayFloat) -> NDArrayFloat:
        max_abs_x: NDArrayFloat = abs(x_regions).max(axis=1)
        max_abs_x[:, 1] = 1.5 * max_abs_x[:, 1]**2
        return max_abs_x.max(axis=1)

    axis_cuts = np.linspace(start=X_ROI[0], stop=X_ROI[1], num=5).transpose()
    x_values = cartesian_prod(*axis_cuts)
    u_values = np.zeros((len(x_values), U_ROI.shape[1]))
    approx = SimplicialComplex(
        X_ROI, U_ROI, x_values, u_values, f_bbox, lip_bbox)

    item = 3
    local_approx = approx[item]
    x_vars = [Variable(f"x{i}") for i in range(X_ROI.shape[1])]
    u_vars = [Variable(f"u{i}") for i in range(U_ROI.shape[1])]
    print(local_approx._x_values)
    print(local_approx._y_values)
    print(local_approx.in_domain_pred(x_vars))
    print(local_approx.func_exprs(x_vars, u_vars, 1))
    print(local_approx.error_bound_expr(x_vars, u_vars, 1))


if __name__ == "__main__":
    test_approx()
