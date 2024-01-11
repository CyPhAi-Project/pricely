from dreal import Box, CheckSatisfiability, Config, Expression as Expr, Formula, Variable, logical_and, logical_or  # type: ignore
import numpy as np
from typing import Optional, Sequence


def check_exact_lyapunov(
    x_vars: Sequence[Variable],
    dxdt_exprs: Sequence[Expr],
    x_roi: np.ndarray,
    lya_expr: Expr,
    norm_lb: float = 0.0,
    norm_ub: float = np.inf,
    config: Config = Config()
) -> Optional[Box]:
    radius_sq = sum(x*x for x in x_vars)
    norm_lb_cond = radius_sq >= norm_lb**2 if np.isfinite(
        norm_lb) and norm_lb > 0 else Formula.TRUE()
    norm_ub_cond = radius_sq <= norm_ub**2 if np.isfinite(
        norm_ub) else Formula.TRUE()

    in_roi_pred = logical_and(
        norm_lb_cond,
        norm_ub_cond,
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


def cartesian_prod(*arrays) -> np.ndarray:
    n = 1
    for x in arrays:
        n *= x.size
    out = np.zeros((n, len(arrays)))

    for i in range(len(arrays)):
        m = int(n / arrays[i].size)
        out[:n, i] = np.repeat(arrays[i], m)
        n //= arrays[i].size

    n = arrays[-1].size
    for k in range(len(arrays)-2, -1, -1):
        n *= arrays[k].size
        m = int(n / arrays[k].size)
        for j in range(1, arrays[k].size):
            out[j*m:(j+1)*m, k+1:] = out[0:m, k+1:]
    return out


def gen_equispace_regions(
        x_part: Sequence[int],
        x_roi: np.ndarray) -> np.ndarray:
    assert x_roi.shape == (2, len(x_part))
    x_dim = len(x_part)
    # generate dataset (values of x):
    axes_cuts = (np.linspace(
        x_roi[0, i], x_roi[1, i], x_part[i]+1) for i in range(x_dim))
    bound_pts = cartesian_prod(
        *axes_cuts).reshape(tuple(n+1 for n in x_part) + (x_dim,))
    lb_pts = bound_pts[(slice(0, -1),)*x_dim].reshape((-1, x_dim))
    ub_pts = bound_pts[(slice(1, None),)*x_dim].reshape((-1, x_dim))
    x = (lb_pts + ub_pts) / 2
    return np.stack((x, lb_pts, ub_pts), axis=1)
