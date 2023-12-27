import torch
from tqdm import tqdm
from typing import Callable, Tuple

from lyapunov_verifier import LyapunovVerifier, split_regions
from lyapunov_learner_nnet import LyapunovNetRegressor
from nnet_utils import DEVICE


def cegus_lyapunov(
        lya: LyapunovNetRegressor,
        x_roi: torch.Tensor,
        x_regions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        f_bbox: Callable[[torch.Tensor], torch.Tensor],
        lip_bbox: float,
        norm_lb: float = 0.0,
        norm_ub: float = torch.inf,
        max_epochs: int = 10,
        max_iter_learn: int = 10):
    null_tensor = torch.tensor([], device=DEVICE)
    null_roi = null_tensor.reshape(2, 0)
    def new_f_bbox(x, u):
        return f_bbox(x)

    return cegus_lyapunov_control(
        lya=lya, x_roi=x_roi, u_roi=null_roi,
        x_cover=x_regions,
        u_values=null_tensor.reshape(len(x_regions[0]), 0),
        f_bbox=new_f_bbox, lip_bbox=lip_bbox,
        norm_lb=norm_lb, norm_ub=norm_ub,
        max_epochs=max_epochs, max_iter_learn=max_iter_learn
    )


def cegus_lyapunov_control(
        lya: LyapunovNetRegressor,
        x_roi: torch.Tensor,
        u_roi: torch.Tensor,
        x_cover: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        u_values: torch.Tensor,
        f_bbox: Callable[[torch.Tensor], torch.Tensor],
        lip_bbox: float,
        norm_lb: float = 0.0,
        norm_ub: float = torch.inf,
        max_epochs: int = 10,
        max_iter_learn: int = 10):
    x_dim, u_dim = x_roi.shape[1], u_roi.shape[1]
    verifier = LyapunovVerifier(
        x_roi=x_roi.cpu().numpy(),
        u_roi=u_roi.cpu().numpy(),
        norm_lb=norm_lb, norm_ub=norm_ub
    )

    lya_risks = []
    x_values, x_lbs, x_ubs = x_cover

    x_values.requires_grad = True
    # Initial sampled x, u values and the constructed set cover
    assert x_values.device == DEVICE
    x_lbs_np = x_lbs.detach().cpu().numpy()
    x_ubs_np = x_ubs.detach().cpu().numpy()
    del x_lbs, x_ubs  # Only use numpy for bounds afterwards

    outer_pbar = tqdm(
        iter(range(1, max_epochs + 1)),
        desc="Outer", ascii=True, postfix={"#Valid": 0, "#Total": len(x_values)})
    for epoch in outer_pbar:
        dxdt_values = f_bbox(x_values, u_values)

        assert x_values.requires_grad, f"at iteration {epoch}."
        new_lya_risks = lya.fit_loop(x_values, dxdt_values,
                                     max_epoch=max_iter_learn, copy=False)
        lya_risks.extend(new_lya_risks)

        # Verify Lyapunov condition
        x_values_np = x_values.detach().cpu().numpy()
        u_values_np = u_values.detach().cpu().numpy()
        dxdt_values_np = dxdt_values.detach().cpu().numpy()
        assert len(x_values_np) == len(dxdt_values_np)

        cex_regions = []
        verifier.set_lyapunov_candidate(lya)
        for j in tqdm(range(len(x_values_np)),
                      desc=f"Verify at {epoch}", ascii=True, leave=False):
            result = verifier.find_cex(
                x_region_j=(x_values_np[j], x_lbs_np[j], x_ubs_np[j]),
                u_j=u_values_np[j],
                dxdt_j=dxdt_values_np[j], lip_expr=lip_bbox)
            if result is not None:
                cex_regions.append((j, result))

        outer_pbar.set_postfix({"#Valid": len(x_values_np)-len(cex_regions), "#Total": len(x_values_np)})
        if len(cex_regions) == 0:
            break  # Lyapunov function candidate passed
        # else:
        # NOTE splitting regions may also modified the input arrays
        x_values_np, x_lbs_np, x_ubs_np = \
            split_regions(x_values_np, x_lbs_np, x_ubs_np, cex_regions)

        # Convert the new sample set to PyTorch
        del x_values, u_values, dxdt_values  # Release memory first
        x_values = torch.tensor(
            x_values_np, dtype=torch.float32,
            requires_grad=True, device=DEVICE)
        u_values = lya.ctrl_values(x_values)
    outer_pbar.close()

    if len(cex_regions) > 0:
        tqdm.write(f"Cannot find a Lyapunov function in {epoch} iterations.")
    return x_values_np, x_lbs_np, x_ubs_np, cex_regions
