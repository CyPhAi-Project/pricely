from dreal import Box, CheckSatisfiability, Config, Expression as Expr, Variable, logical_and, logical_or  # type: ignore
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Sequence


def check_exact_lyapunov(
    x_vars: Sequence[Variable],
    dxdt_exprs: Sequence[Expr],
    lya_expr: Expr,
    level_ub: float = np.inf,
    abs_x_lb: ArrayLike = 2**-6,
    abs_x_ub: ArrayLike = 2**6,
    config: Config = Config()
) -> Optional[Box]:
    assert np.all(0.0 < np.asfarray(abs_x_lb))
    assert np.all(np.isfinite(abs_x_ub))
    assert np.all(np.asfarray(abs_x_lb) < np.asfarray(abs_x_ub))

    sublevel_set_cond = lya_expr <= Expr(level_ub)

    # Add the range on the absolute values of x to limit the search space.
    abs_range_conds = []
    if np.isscalar(abs_x_lb):
        abs_range_conds.extend(abs(x) >= Expr(abs_x_lb) for x in x_vars)
    else:
        abs_x_lb = np.asfarray(abs_x_lb)
        assert len(abs_x_lb) == len(x_vars)
        abs_range_conds.extend(
            abs(x) >= Expr(lb) for x, lb in zip(x_vars, abs_x_lb))

    if np.isscalar(abs_x_ub):
        abs_range_conds.extend(abs(x) <= Expr(abs_x_ub) for x in x_vars)
    else:
        abs_x_ub = np.asfarray(abs_x_ub)
        assert len(abs_x_ub) == len(x_vars)
        abs_range_conds.extend(
            abs(x) <= Expr(ub) for x, ub in zip(x_vars, abs_x_ub))

    in_omega_pred = logical_and(
        *abs_range_conds,
        sublevel_set_cond)

    der_lya = [lya_expr.Differentiate(x) for x in x_vars]
    lie_der_lya = sum(
        der_lya_i*dxdt_i
        for der_lya_i, dxdt_i in zip(der_lya, dxdt_exprs))

    neg_lya_cond = logical_or(lya_expr <= 0.0, lie_der_lya >= 0.0)
    smt_query = logical_and(in_omega_pred, neg_lya_cond)
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
    """ Generate reference values and bounds of x inside ROI """
    assert x_roi.shape == (2, len(x_part))
    x_dim = len(x_part)
    axes_cuts = (np.linspace(
        x_roi[0, i], x_roi[1, i], x_part[i]+1) for i in range(x_dim))
    bound_pts = cartesian_prod(
        *axes_cuts).reshape(tuple(n+1 for n in x_part) + (x_dim,))
    lb_pts = bound_pts[(slice(0, -1),)*x_dim].reshape((-1, x_dim))
    ub_pts = bound_pts[(slice(1, None),)*x_dim].reshape((-1, x_dim))
    x = (lb_pts + ub_pts) / 2
    return np.stack((x, lb_pts, ub_pts), axis=1)
