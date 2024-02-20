import numpy as np
from plot_utils_2d import CatchTime, validate_lip_bbox

from pricely.gen_cover import gen_init_cover
from pricely.cegus_lyapunov import cegar_verify_lyapunov
from pricely.learner_mock import MockQuadraticLearner
from pricely.verifier_dreal import SMTVerifier


def main(max_epochs: int=200):
    # import circle_following as mod
    # import inverted_pendulum as mod
    # import lcss2020_eq14 as mod
    # import lcss2020_eq15 as mod
    import traj_tracking_wheeled as mod

    assert hasattr(mod, "KNOWN_QUAD_LYA"), \
        "Must provide a known quadratic Lyapunov function."

    timer = CatchTime()

    init_part = [40]*mod.X_DIM

    print(" Validate local Lipschitz constants ".center(80, "="))
    with timer:
        validate_lip_bbox(mod, init_part)

    print(" Generate initial samples and cover ".center(80, "="))
    with timer:
        x_regions = gen_init_cover(
            abs_roi_ub=mod.X_ROI[1],
            f_bbox=mod.f_bbox,
            lip_bbox=mod.calc_lip_bbox,
            lip_cap=getattr(mod, "LIP_CAP", np.inf),
            abs_lb=mod.ABS_X_LB,
            init_part=init_part)

    print(" Run CEGAR verification ".center(80, "="))
    with timer:
        mock_learner = MockQuadraticLearner(mod.KNOWN_QUAD_LYA)
        # Set predefined Lyapunov candidate
        last_epoch, last_regions = \
            cegar_verify_lyapunov(
                mock_learner,
                mod.X_ROI,
                mod.ABS_X_LB,
                x_regions,
                mod.f_bbox, mod.calc_lip_bbox,
                max_epochs=max_epochs)
    cegar_status = "Found" if len(last_regions) == 0 else "Can't Find" if last_epoch < max_epochs else "Reach epoch limit"
    cegar_time_usage = timer.elapsed

    if mod.X_DIM != 2:  # Support plotting 2D systems only
        return    
    # TODO

if __name__ == "__main__":
    main()
