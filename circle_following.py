from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
import torch

from cegus_lyapunov import cegus_lyapunov_control
from lyapunov_learner_nnet import KnownControl, KnownLyapunovNet
from nnet_utils import DEVICE, gen_equispace_regions


# Known Lyapunov net from the NeuRIPS 2022 paper
KNOWN_CTRL = KnownControl(
    C=torch.tensor([[5.0]], device=DEVICE),
    K=torch.tensor([-5.9553876, -4.0342584], device=DEVICE),
    b=torch.tensor([0.1974], device=DEVICE))

KNOWN_LYA = KnownLyapunovNet(
    W1=torch.tensor([
        [-2.1378724575042725, +1.0794919729232788],
        [+4.9804959297180176, +0.11680498719215393],
        [+2.8365912437438965, +0.69793730974197388],
        [-3.338552713394165, -2.2363924980163574],
        [-0.027711296454071999, -0.25035503506660461],
        [+0.61321312189102173, -1.6286146640777588]], device=DEVICE),
    b1=torch.tensor([
        -1.9072647094726562,
        -0.79773855209350586,
        +0.18891614675521851,
        +0.73854517936706543,
        +0.87543833255767822,
        +1.0984361171722412], device=DEVICE),
    W2=torch.tensor([[
        -1.2369513511657715,
        +1.4756215810775757,
        -2.1383264064788818,
        -0.76876986026763916,
        +1.0839570760726929,
        -0.84737318754196167]], device=DEVICE),
    b2=torch.tensor(0.59095233678817749, device=DEVICE),
    ctrl=KNOWN_CTRL)


def main():
    VEL = 1.0  # m/s
    X_DIM = 2
    X_ROI = torch.Tensor([
        [-0.8, -0.8],  # Lower bounds
        [0.8, 0.8]  # Upper bounds
    ])
    U_DIM = 1
    U_ROI = torch.Tensor([
        [-5.0],  # Lower bounds
        [5.0]  # Upper bounds
    ])

    def f_bbox(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == X_DIM
        assert u.shape[1] == U_DIM

        dxdt = torch.zeros_like(x)
        dxdt[:, 0] = VEL*torch.sin(x[:, 1])
        dxdt[:, 1] = u[:, 0] - VEL*torch.cos(x[:, 1])/(1-x[:, 0])
        return dxdt

    LIP_BB = 45.0  # Manually derived for ROI

    x_part = (2, 2)
    part = x_part
    x, x_lb, x_ub = gen_equispace_regions(part, X_ROI)
    x = x.to(DEVICE)

    if True:
        lya = KNOWN_LYA
        u = KNOWN_CTRL.apply(x)
    else:
        u = torch.zeros(len(x), U_DIM, device=DEVICE)
        lya_net = LyapunovNet(n_input=X_DIM, n_hidden=6)
        lya = LyapunovNetRegressor(
            module=lya_net,
            optimizer=torch.optim.Adam(lya_net.parameters(), lr=10E-5)
        )

    x_values_np, x_lbs_np, x_ubs_np, cex_regions = \
        cegus_lyapunov_control(
            lya, X_ROI, U_ROI, (x, x_lb, x_ub), u,
            f_bbox, LIP_BB,
            norm_lb=0.1, norm_ub=0.8,
            max_epochs=10, max_iter_learn=1)

    num_samples = len(x_values_np)
    assert X_ROI.shape[1] == 2
    sat_region_iter = (k for k, _ in cex_regions)
    k = next(sat_region_iter, None)
    for j in range(num_samples):
        if j == k:
            k = next(sat_region_iter, None)
            facecolor = "white"
        elif j >= num_samples - len(cex_regions):
            facecolor = "gray"
        else:
            facecolor = "green"
        w, h = x_ubs_np[j] - x_lbs_np[j]
        rect = Rectangle(x_lbs_np[j], w, h, fill=True,
                         edgecolor='black', facecolor=facecolor, alpha=0.3)
        plt.gca().add_patch(rect)

    plt.gca().add_patch(Circle((0, 0), 0.1, color='r', fill=False))
    plt.gca().add_patch(Circle((0, 0), 0.8, color='r', fill=False))
    plt.gca().set_xlim(*X_ROI[:, 0])
    plt.gca().set_ylim(*X_ROI[:, 1])
    plt.gca().set_aspect("equal")
    plt.savefig("out/CircleFollowing_valid_regions.png")
    plt.clf()


if __name__ == "__main__":
    main()
