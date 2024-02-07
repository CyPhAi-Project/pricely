from dreal import CheckSatisfiability, Config, Expression as Expr, Max, Variable, logical_and  # type: ignore
import numpy as np
from numpy.typing import ArrayLike
from typing import NamedTuple, Optional, Tuple, Union

from pricely.cegus_lyapunov import NDArrayFloat, PLyapunovLearner, PLyapunovVerifier


def pretty_sub(i: int) -> str:
    prefix = "" if i < 10 else pretty_sub(i // 10)
    return prefix + chr(0x2080 + (i % 10))


class DRealVars(NamedTuple):
    x: Variable
    der_lya: Variable
    lb: Variable
    ub: Variable
    abs_ub: Variable
    x_s: Variable
    dxdt_s: Variable


class DRealInputs(NamedTuple):
    u: Variable
    u_s: Variable


class SMTVerifier(PLyapunovVerifier):
    def __init__(
            self,
            x_roi: NDArrayFloat,
            u_roi: Optional[NDArrayFloat] = None,
            abs_x_lb: ArrayLike = 2**-6,
            config: Config = None) -> None:
        assert x_roi.shape[0] == 2 and x_roi.shape[1] >= 1
        assert u_roi is None or u_roi.shape[0] == 2
        assert np.all(np.asfarray(abs_x_lb) > 0.0) and np.all(np.isfinite(abs_x_lb))

        self._x_roi = x_roi

        x_dim = x_roi.shape[1]
        self._lya_var = Variable("V")
        self._lya_level_var = Variable("c")
        self._lip_var = Variable("Lip")
        self._all_vars = [
            DRealVars(
                x=Variable(f"x{pretty_sub(i)}"),
                der_lya=Variable(f"∂V/∂x{pretty_sub(i)}"),
                lb=Variable(f"x̲{pretty_sub(i)}"),
                ub=Variable(f"x̄{pretty_sub(i)}"),
                abs_ub=Variable(f"b{pretty_sub(i)}"),
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

        self._smt_tpls = self._init_lyapunov_template(abs_x_lb)

        self._lya_cand_expr = Expr(0.0)
        self._lya_cand_level_ub = np.inf
        self._abs_x_ub = np.inf
        self._der_lya_cand_exprs = [Expr(0.0)]*x_dim
        self._ctrl_exprs = [Expr(0.0)]*u_dim

        if config is not None:
            self._config = config
        else:
            self._config = Config()
            self._config.use_polytope_in_forall = True
            self._config.use_local_optimization = True
            self._config.precision = 1e-6

    @property
    def x_dim(self) -> int:
        return len(self._all_vars)
    
    @property
    def u_dim(self) -> int:
        return len(self._all_inputs)

    def set_lyapunov_candidate(
            self, lya: PLyapunovLearner):
        x_vars = [xi.x for xi in self._all_vars]
        self._lya_cand_expr = lya.lya_expr(x_vars)
        self._lya_cand_level_ub, self._abs_x_ub = \
            lya.find_sublevel_set_and_box(self._x_roi)
        self._der_lya_cand_exprs = [
            self._lya_cand_expr.Differentiate(x) for x in x_vars]
        if self.u_dim > 0:
            self._ctrl_exprs = lya.ctrl_exprs(x_vars)

    def reset_lyapunov_candidate(self):
        self._lya_cand_expr = Expr(0.0)
        self._lya_cand_level_ub = np.inf
        self._abs_x_ub = np.inf
        self._der_lya_cand_exprs = [Expr(0.0)]*self.x_dim
        self._ctrl_exprs = [Expr(0.0)]*self.u_dim

    def find_cex(
        self,
        x_region_j: Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat],
        u_j: NDArrayFloat,
        dxdt_j: NDArrayFloat,
        lip_expr: Union[float, Expr]
    ) -> Optional[NDArrayFloat]:
        assert self._lya_cand_expr is not None \
            and all(e is not None for e in self._der_lya_cand_exprs)
        x_j, x_lb_j, x_ub_j = x_region_j
        assert len(x_j) == self.x_dim
        assert len(x_lb_j) == self.x_dim
        assert len(x_ub_j) == self.x_dim
        assert len(u_j) == self.u_dim

        if np.isscalar(lip_expr):
            lip_expr = Expr(lip_expr)

        sub_pairs = \
            [(self._lya_var, self._lya_cand_expr)] + \
            [(self._lya_level_var, Expr(self._lya_cand_level_ub))] + \
            [(self._lip_var, lip_expr)] + \
            [(xi.der_lya, ei) for xi, ei in zip(self._all_vars, self._der_lya_cand_exprs)] + \
            [(xi.lb, Expr(vi)) for xi, vi in zip(self._all_vars, x_lb_j)] + \
            [(xi.ub, Expr(vi)) for xi, vi in zip(self._all_vars, x_ub_j)] + \
            [(xi.x_s, Expr(vi)) for xi, vi in zip(self._all_vars, x_j)] + \
            [(xi.dxdt_s, Expr(vi)) for xi, vi in zip(self._all_vars, dxdt_j)] + \
            [(ui.u, ei) for ui, ei in zip(self._all_inputs, self._ctrl_exprs)] + \
            [(ui.u_s, Expr(vi)) for ui, vi in zip(self._all_inputs, u_j)]
        
        if np.isscalar(self._abs_x_ub):
            sub_pairs.extend((xi.abs_ub, Expr(self._abs_x_ub)) for xi in self._all_vars)
        else:
            abs_x_ub = np.asfarray(self._abs_x_ub)
            assert len(abs_x_ub) == len(self._all_vars)
            sub_pairs.extend((xi.abs_ub, Expr(ub)) for xi, ub in zip(self._all_vars, abs_x_ub))

        sub_dict = dict(sub_pairs)

        for query_tpl in self._smt_tpls:
            smt_query = query_tpl.Substitute(sub_dict)
            result = CheckSatisfiability(smt_query, self._config)
            if result:
                x_vars  = [var.x for var in self._all_vars]
                box_np = np.asfarray(
                    [[result[var].lb(), result[var].ub()] for var in x_vars]).transpose()
                return box_np
        return None

    def _init_lyapunov_template(
            self,
            abs_x_lb: ArrayLike,
            use_l1: bool = False,
            use_l2: bool = True
        ):
        x_vars = [var.x for var in self._all_vars]
        der_lya_vars = [var.der_lya for var in self._all_vars]

        if np.isscalar(abs_x_lb):
            abs_x_lb_conds = [abs(x) >= Expr(abs_x_lb) for x in x_vars]
        else:
            abs_x_lb = np.asfarray(abs_x_lb)
            assert len(abs_x_lb) == len(x_vars)
            abs_x_lb_conds = [abs(x) >= Expr(lb) for x, lb in zip(x_vars, abs_x_lb)]
        abs_x_ub_conds = [abs(xi.x) <= xi.abs_ub for xi in self._all_vars]
        sublevel_set_cond = (self._lya_var <= self._lya_level_var)

        in_omega_pred = logical_and(
            *abs_x_lb_conds,
            *abs_x_ub_conds,
            sublevel_set_cond)

        in_nbr_pred = logical_and(
            *(logical_and(var.x >= var.lb, var.x <= var.ub)
              for var in self._all_vars))

        lie_der_lya = sum(var.der_lya*var.dxdt_s
                          for var in self._all_vars)

        neg_trig_cond_list = []
        if use_l1:  #  Hölder's inequality for L1, Linf norms
            der_lya_l1 = sum(abs(e) for e in der_lya_vars)
            dist_linf = \
                Max(Max(*(abs(var.x-var.x_s) for var in self._all_vars)),
                          0.0) # Max(*(abs(var.u-var.u_s) for var in self._all_inputs)))
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
            logical_and(in_omega_pred, in_nbr_pred, cond)
            for cond in [self._lya_var <= 0, lie_der_lya >= 0, neg_trig_cond]]

def test_smt_verifier():
    X_ROI = np.asfarray([
        [-1, -2, -3],
        [+1, +2, +3]
    ])
    verifier = SMTVerifier(X_ROI, abs_x_lb=2**-4)
    print(*verifier._smt_tpls, sep="\n")


if __name__ == "__main__":
    test_smt_verifier()
