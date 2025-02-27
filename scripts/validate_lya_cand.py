from dreal import Config, Variable  # type: ignore
import numpy as np

from pricely.cegus_lyapunov import NCOLS, ROI, PLyapunovCandidate
from pricely.utils import check_lyapunov_roi, pretty_sub


def execute(mod, cand: PLyapunovCandidate, dreal_config: Config = Config()):
    print(" Validate learned Lyapunov candidate ".center(NCOLS, "="))

    x_roi = ROI(
    x_lim=mod.X_LIM,
        abs_x_lb=getattr(mod, "ABS_X_LB", 0.0),
        x_norm_lim=(getattr(mod, "X_NORM_LB", 0.0),
                    getattr(mod, "X_NORM_UB", np.inf)))

    x_vars = [Variable(f"x{pretty_sub(i)}") for i in range(mod.X_DIM)]
    dxdt_exprs = mod.f_expr(x_vars)
    lya_expr = cand.lya_expr(x_vars)
    lya_decay_rate = cand.lya_decay_rate()
    print(f"Check Lyapunov potential with decay rate: {lya_decay_rate}")
    result = check_lyapunov_roi(
        x_vars, dxdt_exprs, lya_expr,
        x_roi,
        lya_decay_rate=lya_decay_rate,
        config=dreal_config)
    if result is None:
        print("Learned candidate is a valid Lyapunov function for ROI.")
    else:
        print("Learned candidate is NOT a Lyapunov function for ROI.")
        print(f"Counterexample:\n{result}")
