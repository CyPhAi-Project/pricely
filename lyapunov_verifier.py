import abc
from dreal import CheckSatisfiability, Config, Expression as Expr, Variable, logical_and
import numpy as np
from typing import NamedTuple, Optional, Protocol, Sequence, Tuple, Union


def pretty_sub(i: int) -> str:
    prefix = "" if i < 10 else pretty_sub(i // 10)
    return prefix + chr(0x2080 + (i % 10))


class DRealVars(NamedTuple):
    x: Variable
    der_lya: Variable
    lb: Variable
    ub: Variable
    x_s: Variable
    dxdt_s: Variable


class DRealInputs(NamedTuple):
    u: Variable
    u_s: Variable


class PLyapunovLearner(Protocol):
    @abc.abstractmethod
    def lya_expr(self, x_vars: Sequence[Variable]) -> Expr:
        raise NotImplementedError
    
    @abc.abstractmethod
    def lya_values(self, x_values):
        raise NotImplementedError

    @abc.abstractmethod
    def ctrl_exprs(self, x_vars: Sequence[Variable]) -> Sequence[Expr]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def ctrl_values(self, x_values):
        raise NotImplementedError


class LyapunovVerifier:
    def __init__(
            self,
            x_roi: np.ndarray,
            u_roi: np.ndarray,
            norm_lb: float = 0.0,
            norm_ub: float = np.inf,
            config: Config = None) -> None:
        assert x_roi.shape[0] == 2 and x_roi.shape[1] >= 1
        assert u_roi.shape[0] == 2

        x_dim = x_roi.shape[1]
        self._lya_var = Variable("V")
        self._lip_sq_var = Variable("L²")
        self._all_vars = [
            DRealVars(
                x=Variable(f"x{pretty_sub(i)}"),
                der_lya=Variable(f"∂V/∂x{pretty_sub(i)}"),
                lb=Variable(f"x̲{pretty_sub(i)}"),
                ub=Variable(f"x̄{pretty_sub(i)}"),
                x_s=Variable(f"x̃{pretty_sub(i)}"),
                dxdt_s=Variable(f"f{pretty_sub(i)}(x̃,ũ)")
            ) for i in range(x_dim)
        ]

        u_dim = u_roi.shape[1] if u_roi is not None else 0
        self._all_inputs = [
            DRealInputs(
                u=Variable(f"u{pretty_sub(i)}"),
                u_s=Variable(f"ũ{pretty_sub(i)}")
            ) for i in range(u_dim)
        ]

        self._smt_tpls = self._init_lyapunov_template(x_roi, u_roi, norm_lb, norm_ub)

        self._lya_cand_expr = None
        self._der_lya_cand_exprs = [None]*x_dim
        self._ctrl_exprs = [None]*u_dim

        if config is not None:
            self._config = config
        else:
            self._config = Config()
            self._config.use_polytope_in_forall = True
            self._config.use_local_optimization = True
            self._config.precision = 1e-6

    def set_lyapunov_candidate(self, lya: PLyapunovLearner):
        x_vars = [xi.x for xi in self._all_vars]
        self._lya_cand_expr = lya.lya_expr(x_vars)
        self._der_lya_cand_exprs = [
            self._lya_cand_expr.Differentiate(x) for x in x_vars]
        if len(self._all_inputs) > 0:
            self._ctrl_exprs = lya.ctrl_exprs(x_vars)

    def reset_lyapunov_candidate(self):
        self._lya_cand_expr = None
        self._der_lya_cand_exprs = [None]*len(self._all_vars)
        self._ctrl_exprs = [None]*len(self._all_inputs)

    def find_cex(
        self,
        x_region_j: Tuple[np.ndarray, np.ndarray, np.ndarray],
        u_j: np.ndarray,
        dxdt_j: np.ndarray,
        lip_expr: Union[float, Expr]
    ) -> Optional[np.ndarray]:
        assert self._lya_cand_expr is not None \
            and all(e is not None for e in self._der_lya_cand_exprs)
        x_dim, u_dim = len(self._all_vars), len(self._all_inputs)
        x_j, x_lb_j, x_ub_j = x_region_j
        assert len(x_j) == x_dim
        assert len(x_lb_j) == x_dim
        assert len(x_ub_j) == x_dim
        assert len(u_j) == u_dim
        assert not (u_dim == 0 and self._ctrl_exprs is None)

        if isinstance(lip_expr, float):
            lip_expr = Expr(lip_expr)

        sub_pairs = \
            [(self._lya_var, self._lya_cand_expr)] + \
            [(self._lip_sq_var, lip_expr**2)] + \
            [(xi.der_lya, ei) for xi, ei in zip(self._all_vars, self._der_lya_cand_exprs)] + \
            [(xi.lb, Expr(vi)) for xi, vi in zip(self._all_vars, x_lb_j)] + \
            [(xi.ub, Expr(vi)) for xi, vi in zip(self._all_vars, x_ub_j)] + \
            [(xi.x_s, Expr(vi)) for xi, vi in zip(self._all_vars, x_j)] + \
            [(xi.dxdt_s, Expr(vi)) for xi, vi in zip(self._all_vars, dxdt_j)] + \
            [(ui.u, ei) for ui, ei in zip(self._all_inputs, self._ctrl_exprs)] + \
            [(ui.u_s, Expr(vi)) for ui, vi in zip(self._all_inputs, u_j)]
        sub_dict = dict(sub_pairs)

        for query_tpl in self._smt_tpls:
            smt_query = query_tpl.Substitute(sub_dict)
            result = CheckSatisfiability(smt_query, self._config)
            if result:
                x_vars  = [var.x for var in self._all_vars]
                box_np = np.asfarray(
                    [[result[var].lb(), result[var].ub()] for var in x_vars],
                    dtype=np.float32).transpose()
                return box_np
        return None

    def _init_lyapunov_template(
            self,
            x_roi: np.ndarray,
            u_roi: np.ndarray,
            norm_lb: float = 0.0,
            norm_ub: float = np.inf
        ):
        assert x_roi.shape[1] >= 1

        x_vars = [var.x for var in self._all_vars]
        der_lya_vars = [var.der_lya for var in self._all_vars]
        u_vars = [var.u for var in self._all_inputs]

        radius_sq = sum(x*x for x in x_vars)
        in_roi_pred = logical_and(
            radius_sq >= norm_lb**2,
            radius_sq <= norm_ub**2,
            *(logical_and(x >= lb, x <= ub)
              for x, lb, ub in zip(x_vars, x_roi[0], x_roi[1])))

        in_nbr_pred = logical_and(
            *(logical_and(var.x >= var.lb, var.x <= var.ub)
              for var in self._all_vars))

        lie_der_lya = sum(var.der_lya*var.dxdt_s
                          for var in self._all_vars)

        der_lya_l2_sq = sum(e*e for e in der_lya_vars)
        dist_sq = \
            sum((var.x-var.x_s)**2 for var in self._all_vars) + \
            sum((var.u-var.u_s)**2 for var in self._all_inputs)
        trig_cond = der_lya_l2_sq*dist_sq*self._lip_sq_var >= lie_der_lya**2
        # Validity Cond: forall x in [lb, ub].
        #   V(x) > 0 /\ ∂V/∂x⋅f(x̃,ũ) < 0 /\ (|∂V/∂x||(x-x̃,u-ũ)|Lip)² < |∂V/∂x⋅f(x̃,ũ)|²
        # SMT Cond: exists x in [lb, ub].
        #   V(x)<= 0 \/ ∂V/∂x⋅f(x̃,ũ)>= 0 \/ (|∂V/∂x||(x-x̃,u-ũ)|Lip)²>= |∂V/∂x⋅f(x̃,ũ)|²
        return [
            logical_and(in_roi_pred, in_nbr_pred, cond)
            for cond in [self._lya_var <= 0, lie_der_lya >= 0, trig_cond]]


def split_regions(
    x_values: np.ndarray,
    x_lbs: np.ndarray,
    x_ubs: np.ndarray,
    sat_regions: Sequence[Tuple[int, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    new_cexs, new_lbs, new_ubs = [], [], []
    for j, box_j in sat_regions:
        res = split_region((x_values[j], x_lbs[j], x_ubs[j]), box_j)
        if res is None:
            continue  # Skip because sampled state is inside cex box.
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
    return x_values, x_lbs, x_ubs


def split_region(
    region: Tuple[np.ndarray, np.ndarray, np.ndarray],
    box: np.ndarray
) -> Optional[Tuple[np.ndarray, int, float]]:
    cex_lb, cex_ub = box
    x, lb, ub = region
    if np.all(np.logical_and(cex_lb <= x, x <= cex_ub)):
        return
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
