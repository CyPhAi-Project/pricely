import torch
from tqdm import tqdm
from typing import Callable, Sequence, Tuple

from lyapunov_utlis import LyapunovVerifier, split_regions
from nnet_utils import DEVICE, NeuralNetRegressor


def cegus_lyapunov(
        lya: NeuralNetRegressor,
        x_roi: torch.Tensor,
        x_regions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        f_bbox: Callable[[torch.Tensor], torch.Tensor],
        lip_bbox: float,
        max_epochs: int = 10,
        max_iter_learn: int = 10
) -> Sequence[float]:
    verifier = LyapunovVerifier(x_roi=x_roi.cpu().numpy())

    lya_risks = []
    x_values, x_lbs, x_ubs = x_regions
    x_values.requires_grad = True
    # Initial sampled x values and the constructed set cover
    assert x_values.device == DEVICE
    x_lbs_np = x_lbs.detach().cpu().numpy()
    x_ubs_np = x_ubs.detach().cpu().numpy()
    del x_lbs, x_ubs  # Only use numpy for bounds afterwards

    outer_pbar = tqdm(
        iter(range(1, max_epochs + 1)),
        desc="Outer", ascii=True, postfix={"#Valid": 0, "#Total": len(x_values)})
    for epoch in outer_pbar:
        dxdt_values = f_bbox(x_values)
        assert x_values.requires_grad, f"at iteration {epoch}."

        new_lya_risks = lya.fit_loop(x_values, dxdt_values,
                                     max_epoch=max_iter_learn, copy=False)
        lya_risks.extend(new_lya_risks)

        # Verify Lyapunov condition
        x_values_np = x_values.detach().cpu().numpy()
        dxdt_values_np = dxdt_values.detach().cpu().numpy()
        assert len(x_values_np) == len(dxdt_values_np)

        cex_regions = []
        verifier.set_lyapunov_candidate(lya)
        for j in tqdm(range(len(x_values_np)),
                      desc=f"Verify at {epoch}", ascii=True, leave=False):
            result = verifier.find_cex(
                lb_j=x_lbs_np[j], ub_j=x_ubs_np[j],
                x_j=x_values_np[j], dxdt_j=dxdt_values_np[j], lip_expr=lip_bbox)
            if result:
                cex_regions.append((j, result))

        outer_pbar.set_postfix({"#Valid": len(x_values)-len(cex_regions), "#Total": len(x_values)})
        if len(cex_regions) == 0:
            break  # Lyapunov function candidate passed
        # else:
        # NOTE splitting regions may also modified the input arrays
        x_values_np, x_lbs_np, x_ubs_np = \
            split_regions(x_values_np, x_lbs_np, x_ubs_np, cex_regions)

        # Convert the new sample set to PyTorch
        del x_values, dxdt_values  # Release memory first
        x_values = torch.tensor(
            x_values_np, dtype=torch.float32,
            requires_grad=True, device=DEVICE)
        assert x_values.requires_grad
    outer_pbar.close()

    if len(cex_regions) > 0:
        tqdm.write(f"Cannot find a Lyapunov function in {epoch} iterations.")
    return x_values_np, x_lbs_np, x_ubs_np, cex_regions
