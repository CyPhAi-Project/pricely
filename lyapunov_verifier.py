import dreal
from dreal import Box, CheckSatisfiability, Config, Expression as Expr, Variable, logical_and, logical_or  # type: ignore
import numpy as np
from typing import NamedTuple, Optional, Sequence, Tuple, Union

from cegus_lyapunov import PLyapunovLearner, PLyapunovVerifier


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


class SMTVerifier(PLyapunovVerifier):
    def __init__(
            self,
            x_roi: np.ndarray,
            u_roi: Optional[np.ndarray] = None,
            norm_lb: float = 0.0,
            norm_ub: float = np.inf,
            config: Config = None) -> None:
        assert x_roi.shape[0] == 2 and x_roi.shape[1] >= 1
        assert u_roi is None or u_roi.shape[0] == 2

        x_dim = x_roi.shape[1]
        self._lya_var = Variable("V")
        self._lip_var = Variable("Lip")
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

        self._smt_tpls = self._init_lyapunov_template(x_roi, norm_lb, norm_ub)

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
            [(self._lip_var, lip_expr)] + \
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
            norm_lb: float = 0.0,
            norm_ub: float = np.inf,
            use_l1: bool = False,
            use_l2: bool = True
        ):
        assert x_roi.shape[1] >= 1

        x_vars = [var.x for var in self._all_vars]
        der_lya_vars = [var.der_lya for var in self._all_vars]

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

        neg_trig_cond_list = []
        if use_l1:  #  Hölder's inequality for L1, Linf norms
            der_lya_l1 = sum(abs(e) for e in der_lya_vars)
            dist_linf = \
                dreal.Max(dreal.Max(*(abs(var.x-var.x_s) for var in self._all_vars)),
                          0.0) # dreal.Max(*(abs(var.u-var.u_s) for var in self._all_inputs)))
            neg_trig_cond_l1 = (der_lya_l1*dist_linf*self._lip_var + lie_der_lya >= 0.0)
            neg_trig_cond_list.append(neg_trig_cond_l1)

        if use_l2:  # Cauchy-Schwarz inequality
            der_lya_l2_sq = sum(e*e for e in der_lya_vars)
            dist_sq = \
                sum((var.x-var.x_s)**2 for var in self._all_vars) + \
                sum((var.u-var.u_s)**2 for var in self._all_inputs)
            # (|∂V/∂x||(x-x̃,u-ũ)|Lip)²>= |∂V/∂x⋅f(x̃,ũ)|²
            neg_trig_cond_l2 = der_lya_l2_sq*dist_sq*(self._lip_var**2) >= lie_der_lya**2
            neg_trig_cond_list.append(neg_trig_cond_l2)

        neg_trig_cond = logical_and(*neg_trig_cond_list)
        # Validity Cond: forall x in [lb, ub].
        #   V(x) > 0 /\ ∂V/∂x⋅f(x̃,ũ) < 0 /\ (|∂V/∂x||(x-x̃,u-ũ)|Lip < -∂V/∂x⋅f(x̃,ũ)
        # SMT Cond: exists x in [lb, ub].
        #   V(x)<= 0 \/ ∂V/∂x⋅f(x̃,ũ)>= 0 \/ |∂V/∂x||(x-x̃,u-ũ)|Lip >= -∂V/∂x⋅f(x̃,ũ)
        return [
            logical_and(in_roi_pred, in_nbr_pred, cond)
            for cond in [self._lya_var <= 0, lie_der_lya >= 0, neg_trig_cond]]


def check_exact_lyapunov(
        x_vars: Sequence[Variable],
        dxdt_exprs: Sequence[Expr],
        x_roi: np.ndarray,
        lya_expr: Expr,
        norm_lb: float=0.0,
        norm_ub: float=np.inf,
        config: dreal.Config = dreal.Config()
    ) -> Optional[Box]:
    radius_sq = sum(x*x for x in x_vars)
    in_roi_pred = logical_and(
        dreal.sqrt(radius_sq) >= norm_lb,
        dreal.sqrt(radius_sq) <= norm_ub,
        *(logical_and(x >= lb, x <= ub)
        for x, lb, ub in zip(x_vars, x_roi[0], x_roi[1]))
    )

    der_lya = [lya_expr.Differentiate(x) for x in x_vars]
    lie_der_lya = sum(
        der_lya_i*dxdt_i
        for der_lya_i, dxdt_i in zip(der_lya, dxdt_exprs))

    neg_lya_cond = logical_or(lya_expr <= 0.0, lie_der_lya >= 0.0)
    smt_query = logical_and(in_roi_pred, neg_lya_cond)
    result = CheckSatisfiability(smt_query, config)
    return result
