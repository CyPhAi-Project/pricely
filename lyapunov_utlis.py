from dreal import Box, CheckSatisfiability, Config, Expression as Expr, Variable, logical_and
import numpy as np
from typing import NamedTuple, Optional, Sequence, Tuple, Union
import warnings


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


class LyapunovVerifier:
    def __init__(
            self,
            x_roi: np.ndarray,
            u_roi: np.ndarray = None,
            config: Config = None) -> None:
        assert x_roi.shape[0] == 2 and x_roi.shape[1] >= 1
        assert u_roi is None or (u_roi.shape[0] == 2 and x_roi.shape[1] >= 1)
        if u_roi is not None:
            raise NotImplementedError(
                "System with input is not supported yet.")

        x_dim = x_roi.shape[1]
        self._lya_var = Variable("V")
        self._lip_sq_var = Variable("L²")
        self._all_vars = [
            DRealVars(
                x=Variable(f"x{pretty_sub(i)}"),
                der_lya=Variable(f"∂V/∂x{pretty_sub(i)}"),
                lb=Variable(f"lb{pretty_sub(i)}"),
                ub=Variable(f"ub{pretty_sub(i)}"),
                x_s=Variable(f"x̃{pretty_sub(i)}"),
                dxdt_s=Variable(f"f{pretty_sub(i)}(x̃)")
            ) for i in range(x_dim)
        ]
        self._smt_tpls = self._init_lyapunov_template(x_roi, u_roi)

        self._lya_cand_expr = None
        self._der_lya_cand_expr = None

        if config is not None:
            self._config = config
        else:
            self._config = Config()
            self._config.use_polytope_in_forall = True
            self._config.use_local_optimization = True
            self._config.precision = 1e-6

    def set_lyapunov_candidate(self, lya):
        x_vars = [xi.x for xi in self._all_vars]
        self._lya_cand_expr = lya.dreal_expr(x_vars)
        self._der_lya_cand_expr = [
            self._lya_cand_expr.Differentiate(x) for x in x_vars]

    def reset_lyapunov_candidate(self):
        self._lya_cand_expr = None
        self._der_lya_cand_expr = None

    def find_cex(
        self,
        lb_j: np.ndarray,
        ub_j: np.ndarray,
        x_j: np.ndarray,
        dxdt_j: np.ndarray,
        lip_expr: Union[float, Expr]
    ) -> Optional[Box]:
        assert self._lya_cand_expr is not None \
            and self._der_lya_cand_expr is not None
        if isinstance(lip_expr, float):
            lip_expr = Expr(lip_expr)

        sub_pairs = \
            [(self._lya_var, self._lya_cand_expr)] + \
            [(self._lip_sq_var, lip_expr**2)] + \
            [(xi.der_lya, ei) for xi, ei in zip(self._all_vars, self._der_lya_cand_expr)] + \
            [(xi.lb, Expr(vi)) for xi, vi in zip(self._all_vars, lb_j)] + \
            [(xi.ub, Expr(vi)) for xi, vi in zip(self._all_vars, ub_j)] + \
            [(xi.x_s, Expr(vi)) for xi, vi in zip(self._all_vars, x_j)] + \
            [(xi.dxdt_s, Expr(vi)) for xi, vi in zip(self._all_vars, dxdt_j)]
        sub_dict = dict(sub_pairs)

        for query_tpl in self._smt_tpls:
            smt_query = query_tpl.Substitute(sub_dict)
            result = CheckSatisfiability(smt_query, self._config)
            if result:
                return result
        return None

    def _init_lyapunov_template(
            self,
            x_roi: np.ndarray,
            u_roi: np.ndarray):
        assert x_roi.shape[1] >= 1

        x_vars = [var.x for var in self._all_vars]
        der_lya_vars = [var.der_lya for var in self._all_vars]
        lb_vars = [var.lb for var in self._all_vars]
        ub_vars = [var.ub for var in self._all_vars]
        x_s_vars = [var.x_s for var in self._all_vars]
        dxdt_s_vars = [var.dxdt_s for var in self._all_vars]

        # TODO remove hard coded ball (region of interest)
        ball_lb = 0.2
        ball_ub = 1.2
        warnings.warn(
            f"Warning: hardcoded region of interest {ball_lb} <= |x| <= {ball_ub}.")

        radius_sq = sum(x*x for x in x_vars)
        in_roi_pred = logical_and(
            radius_sq >= ball_lb**2,
            radius_sq <= ball_ub**2,
            *(logical_and(x >= lb, x <= ub)
              for x, lb, ub in zip(x_vars, x_roi[0], x_roi[1]))
        )

        in_nbr_pred = logical_and(
            *(logical_and(x >= lb, x <= ub)
              for x, lb, ub in zip(x_vars, lb_vars, ub_vars)))

        lie_der_lya = sum(der_lya_i*dxdt_i
                          for der_lya_i, dxdt_i in zip(der_lya_vars, dxdt_s_vars))

        der_lya_l2_sq = sum(e*e for e in der_lya_vars)
        dist_sq = sum((x-v)*(x-v) for x, v in zip(x_vars, x_s_vars))
        trig_cond = der_lya_l2_sq*dist_sq * \
            self._lip_sq_var >= lie_der_lya*lie_der_lya
        # Validity Cond: forall x in [lb, ub].
        #   V(x) > 0 /\ ∂V/∂x⋅f(x̃) < 0 /\ (|∂V/∂x||x-x̃|Lip)² < |∂V/∂x⋅f(x̃)|²
        # SMT Cond: exists x in [lb, ub].
        #   V(x)<= 0 \/ ∂V/∂x⋅f(x̃)>= 0 \/ (|∂V/∂x||x-x̃|Lip)²>= |∂V/∂x⋅f(x̃)|²
        return [
            logical_and(in_roi_pred, in_nbr_pred, cond)
            for cond in [self._lya_var <= 0, lie_der_lya >= 0, trig_cond]]


def split_regions(
    x_values: np.ndarray,
    x_lbs: np.ndarray,
    x_ubs: np.ndarray,
    sat_regions: Sequence[Tuple[int, Box]]
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
    box: Box
) -> Optional[Tuple[np.ndarray, int, float]]:
    cex_lb, cex_ub = np.asfarray(
        [[itvl.lb(), itvl.ub()] for itvl in box.values()],
        dtype=np.float32).transpose()
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
