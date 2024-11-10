import abc
from dreal import Expression as Expr, Formula, Variable  # type: ignore
from multiprocessing import Pool, TimeoutError
import numpy as np
from numpy.typing import ArrayLike, NDArray
from signal import signal, SIGINT
from tqdm import tqdm
from typing import Callable, Hashable, Literal, NamedTuple, Optional, Protocol, Sequence, Tuple

NCOLS = 120

NDArrayFloat = NDArray[np.float_]
NDArrayIndex = NDArray[np.int_]

class PLyapunovCandidate(Protocol):
    @abc.abstractmethod
    def __copy__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def lya_expr(self, x_vars: Sequence[Variable]) -> Expr:
        raise NotImplementedError

    @abc.abstractmethod
    def lya_values(self, x_values: NDArrayFloat) -> NDArrayFloat:
        raise NotImplementedError

    def lya_decay_rate(self) -> float:
        return 0.0  # default no decay

    @abc.abstractmethod
    def lie_der_values(self, x_values: NDArrayFloat, y_values: NDArrayFloat) -> NDArrayFloat:
        raise NotImplementedError

    @abc.abstractmethod
    def ctrl_exprs(self, x_vars: Sequence[Variable]) -> Sequence[Expr]:
        raise NotImplementedError

    @abc.abstractmethod
    def ctrl_values(self, x_values: NDArrayFloat) -> NDArrayFloat:
        raise NotImplementedError

    @abc.abstractmethod
    def find_level_ub(self, x_roi: NDArrayFloat) -> float:
        raise NotImplementedError


class PLyapunovLearner(Protocol):
    @abc.abstractmethod
    def fit_loop(self, x: NDArrayFloat, u: NDArrayFloat, y: NDArrayFloat, **kwargs):
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_candidate(self) -> PLyapunovCandidate:
        raise NotImplementedError


class PLocalApprox(Protocol):
    @property
    @abc.abstractmethod
    def num_approxes(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def x_witness(self) -> NDArrayFloat:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def domain_diameter(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def in_domain_repr(self) -> Hashable:
        raise NotImplementedError

    @abc.abstractmethod
    def in_domain_pred(self, x_vars: Sequence[Variable]) -> Formula:
        raise NotImplementedError

    @abc.abstractmethod
    def func_exprs(self, x_vars: Sequence[Variable], u_vars: Sequence[Variable], k: int) -> Sequence[Expr]:
        raise NotImplementedError

    @abc.abstractmethod
    def error_bound_expr(self, x_vars: Sequence[Variable], u_vars: Sequence[Variable], k: int) -> Expr:
        raise NotImplementedError


class PApproxDynamic(Sequence[PLocalApprox]):
    @property
    @abc.abstractmethod
    def x_values(self) -> NDArrayFloat:
        "Get sampled states"
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def x_regions(self) -> NDArrayFloat:
        "Get regions of states"
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def u_values(self) -> NDArrayFloat:
        "Get sampled inputs"
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def y_values(self) -> NDArrayFloat:
        "Get sampled outputs from black-box dynamics"
        raise NotImplementedError

    @abc.abstractmethod
    def add(self, cex_boxes: Sequence[Tuple[int, NDArrayFloat]], cand: PLyapunovCandidate) -> None:
        raise NotImplementedError


class PLyapunovVerifier(Protocol):
    @property
    @abc.abstractmethod
    def x_dim(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def filter_idx(self, x_values: NDArrayFloat) -> NDArrayIndex:
        raise NotImplementedError

    @abc.abstractmethod
    def set_lyapunov_candidate(self, cand: PLyapunovCandidate):
        raise NotImplementedError

    @abc.abstractmethod
    def find_cex(self, approx_j: PLocalApprox) -> Optional[NDArrayFloat]:
        raise NotImplementedError


def handler(signumber, frame):
    raise RuntimeError("Handle Ctrl+C for child processes")


def parallelizable_verify(
        verifier: PLyapunovVerifier,
        f_approx_j: PLocalApprox):
    signal(SIGINT, handler)
    reg_repr = f_approx_j.in_domain_repr()
    return reg_repr, verifier.find_cex(f_approx_j), f_approx_j.domain_diameter


class CEGuSResult(NamedTuple):
    status: Literal["FOUND", "NO_CANDIDATE", "EPOCH_LIMIT", "SAMPLE_LIMIT", "PRECISION_LIMIT"]
    epoch: int
    approx: PApproxDynamic
    cex_regions: Sequence[Tuple[int, NDArrayFloat]]


def cegus_lyapunov(
        learner: PLyapunovLearner,
        verifier: PLyapunovVerifier,
        init_approx: PApproxDynamic,
        eps: float = 1e-6,
        max_epochs: int = 10,
        max_iter_learn: int = 10,
        max_num_samples: int = 5*10**5,
        n_jobs: int = 1,
        timeout_per_job: Optional[float] = 30.0) -> CEGuSResult:
    assert max_epochs > 0

    diam_lb = eps * np.sqrt(2.0 + 2.0 / verifier.x_dim) / 2.0
    # Initial set cover and sampled values
    curr_approx = init_approx

    outer_pbar = tqdm(
        iter(range(max_epochs)),
        desc="CEGuS Loop", ascii=True, leave=True, position=0, ncols=NCOLS,
        postfix={"#Samples": len(curr_approx.x_values)})
    cex_regions = []
    obj_values = []

    ## Prepare dataset for learning
    filter_idx = verifier.filter_idx(curr_approx.x_values)
    x_values = curr_approx.x_values[filter_idx]
    u_values = curr_approx.u_values[filter_idx]
    y_values = curr_approx.y_values[filter_idx]
    for epoch in outer_pbar:
        # NOTE x_values is filtered 
        outer_pbar.set_postfix({"#Samples for learner": len(x_values)})
        # Learn a new candidate
        objs = learner.fit_loop(
            x_values, u_values, y_values,
            max_epochs=max_iter_learn, copy=False)
        if not objs:
            tqdm.write("No Lyapunov candidate in learner's hypothesis space.")
            return CEGuSResult("NO_CANDIDATE", epoch, curr_approx, cex_regions)

        obj_values.extend(objs)
        cand = learner.get_candidate()
        verifier.set_lyapunov_candidate(cand)
        stop_refine = False
        verified_regions = set()

        refinement_pbar = tqdm(
            desc=f"Refinement Loop",
            ascii=True, leave=None, position=1, ncols=NCOLS)
        while True:
            if stop_refine:
                refinement_pbar.set_postfix({
                    "#Valid Regions": "?",
                    "#Total Regions": len(curr_approx),
                    "#Samples": len(curr_approx.x_values)})
                tqdm.write("Current candidate is neither δ-provable nor falsified.")
                return CEGuSResult("PRECISION_LIMIT", epoch, curr_approx, cex_regions)

            num_timeouts = 0
            cex_regions.clear()
            with Pool(n_jobs) as p:
                new_verified_regions = set()
                if len(verified_regions) <= len(curr_approx) // 4:
                    future_list = [(j, p.apply_async(
                            func=parallelizable_verify,
                            args=((verifier, curr_approx[j]))))
                        for j in range(len(curr_approx))]
                else:
                    # Reuse verified regions only when it is worthwhile.
                    future_list = []
                    for j in tqdm(range(len(curr_approx)),
                                  desc="Check cache", ascii=True, leave=None, position=2, ncols=NCOLS):
                        reg_repr = curr_approx[j].in_domain_repr()
                        if reg_repr in verified_regions:
                            new_verified_regions.add(reg_repr)
                        else:
                            future_list.append((j, p.apply_async(
                                func=parallelizable_verify,
                                args=((verifier, curr_approx[j])))))

                for j, future in tqdm(future_list,
                                   desc=f"Verify", ascii=True, leave=None, position=2, ncols=NCOLS):
                    try:
                        reg_repr, box, diam = future.get(timeout_per_job)
                        if box is None:
                            new_verified_regions.add(reg_repr)
                        else:
                            stop_refine = stop_refine or (diam <= diam_lb)
                            cex_regions.append((j, box))
                    except TimeoutError:
                        num_timeouts += 1
                        cex = curr_approx[j].x_witness
                        box = np.row_stack((cex, cex))
                        cex_regions.append((j, box))
                verified_regions.clear()
                verified_regions = new_verified_regions

            if num_timeouts > 0:
                tqdm.write(f'Regional verifier times out for {num_timeouts} regions at {epoch}it')
            refinement_pbar.set_postfix({
                "#Total Regions": len(curr_approx),
                "#Valid Regions": len(curr_approx) - len(cex_regions),
                "#VC Queries": len(future_list),
                "#Samples": len(curr_approx.x_values)})
            if len(cex_regions) == 0:
                assert num_timeouts == 0
                # Lyapunov function candidate passed
                return CEGuSResult("FOUND", epoch, curr_approx, cex_regions)
            if len(curr_approx.x_values) + len(cex_regions) >= max_num_samples:
                tqdm.write(f"Exceeding max number of samples {max_num_samples} in next iteration.")
                return CEGuSResult("SAMPLE_LIMIT", epoch, curr_approx, cex_regions)

            # Update the cover with counterexamples
            curr_approx.add(cex_regions, cand)

            filter_idx = verifier.filter_idx(curr_approx.x_values)
            x_values = curr_approx.x_values[filter_idx]
            u_values = curr_approx.u_values[filter_idx]
            y_values = curr_approx.y_values[filter_idx]
            if np.any(cand.lya_values(x_values) <= 0.0) or \
                    np.any(cand.lie_der_values(x_values, y_values) >= 0.0):
                # Found true counterexamples. Break to learn a new candidate.
                break
            refinement_pbar.update(1)

    outer_pbar.close()

    tqdm.write(f"Cannot find a Lyapunov function in {max_epochs} iterations.")
    return CEGuSResult("EPOCH_LIMIT", max_epochs, curr_approx, cex_regions)


def verify_lyapunov(
        learner: PLyapunovLearner,
        x_roi: NDArrayFloat,
        abs_x_lb: ArrayLike,
        init_x_regions: NDArrayFloat,
        f_bbox: Callable[[NDArrayFloat], NDArrayFloat],
        lip_bbox: Callable[[NDArrayFloat], NDArrayFloat],
        max_epochs: int = 10,
        max_num_samples: int = 10**7):
    def new_f_bbox(x, u):
        return f_bbox(x)
    
    def new_lip_bbox(x, u):
        return lip_bbox(x)

    return verify_lyapunov_control(
        learner=learner,
        x_roi=x_roi,
        abs_x_lb=abs_x_lb,
        init_x_regions=init_x_regions,
        f_bbox=new_f_bbox, lip_bbox=new_lip_bbox,
        max_epochs=max_epochs,
        max_num_samples=max_num_samples)


def verify_lyapunov_control(
        learner: PLyapunovLearner,
        x_roi: NDArrayFloat,
        abs_x_lb: ArrayLike,
        init_x_regions: NDArrayFloat,
        f_bbox: Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat],
        lip_bbox: Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat],
        max_epochs: int = 10,
        max_num_samples: int = 10**7):
    """
    assert init_x_regions.shape[1] == 3
    assert max_epochs > 0
    x_dim = x_roi.shape[1]
    # Initial sampled x values and the constructed set cover
    x_regions = init_x_regions

    def parallelizable_verify(
            x_region_j: Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat],
            u_j: NDArrayFloat,
            dxdt_j: NDArrayFloat,
            lip_ub: float):
        u_j = np.atleast_1d(u_j)
        from pricely.verifier_dreal import SMTVerifier
        verifier = SMTVerifier(x_roi=x_roi, u_dim=len(u_j), abs_x_lb=abs_x_lb)
        verifier.set_lyapunov_candidate(learner)
        return verifier.find_cex(
            x_region_j=x_region_j,
            f_approx_j=u_j,
            err_j=dxdt_j,
            lip_expr=lip_ub)

    verified = 0
    outer_pbar = tqdm(
        iter(range(1, max_epochs + 1)), leave=True,
        desc="Outer", ascii=True, postfix={"#Not Verified": len(x_regions), "#Verified": verified})
    cex_regions = []
    for epoch in outer_pbar:
        x_values = x_regions[:, 0]
        u_values = learner.ctrl_values(x_values)
        dxdt_values = f_bbox(x_values, u_values)
        # Verify Lyapunov condition
        assert len(x_regions) == len(dxdt_values)

        cex_regions.clear()
        lip_ubs = lip_bbox(x_regions)

        results = tqdm(enumerate(
            Parallel(n_jobs=16, return_as="generator", prefer="processes")(
            delayed(parallelizable_verify)(
                    x_region_j=x_regions[j],
                    u_j=u_values[j],
                    dxdt_j=dxdt_values[j],
                    lip_ub=lip_ubs[j]) for j in range(len(x_regions)))),
            desc=f"Verify at {epoch}", 
            total=len(x_regions), ascii=True, leave=False)
        cex_regions = [(j, result) for j, result in results if result is not None]

        verified += len(x_regions) - len(cex_regions)
        outer_pbar.set_postfix({"#Not Verified": len(cex_regions), "#Verified": verified})
        if len(cex_regions) == 0:
            # Lyapunov function candidate passed
            return epoch, verified, np.zeros(shape=(0, 3, x_dim))
        if verified + len(cex_regions) > max_num_samples:
            tqdm.write(f"Exceed max # samples {max_num_samples}.")
            return epoch, verified + len(cex_regions), x_regions
        # else:
        # NOTE splitting regions may also modified the input arrays
        x_regions = get_unverified_regions(x_regions, cex_regions)

    outer_pbar.close()

    tqdm.write(f"Cannot verify the given Lyapunov function in {max_epochs} iterations.")
    return max_epochs, verified + len(cex_regions), x_regions
    """
    return 0, 0, np.array([]).reshape((0, 3, x_roi.shape[1]))


def get_unverified_regions(
        x_regions: NDArrayFloat,
        sat_regions: Sequence[Tuple[int, NDArrayFloat]],
        n_jobs: int = 16) -> NDArrayFloat:
    assert x_regions.shape[1] == 3

    def parallelizable_split(x_region_j: NDArrayFloat, box_j: NDArrayFloat):
        res = split_region(x_region_j, box_j)
        if res is None:
            # FIXME Why the cex is so close to the sampled value?
            raise RuntimeError("Sampled state is inside cex box")
        x_j, x_lb_j, x_ub_j = x_region_j
        cex, cut_axis, cut_value = res
        cex_lb, cex_ub = x_lb_j.copy(), x_ub_j.copy()

        if cex[cut_axis] < x_j[cut_axis]:
            # Increase lower bound for old sample
            x_lb_j[cut_axis] = cut_value
            cex_ub[cut_axis] = cut_value  # Decrease upper bound for new sample
        else:
            assert cex[cut_axis] > x_j[cut_axis]
            # Decrease upper bound for old sample
            x_ub_j[cut_axis] = cut_value
            cex_lb[cut_axis] = cut_value  # Increase lower bound for new sample
        return np.row_stack((x_j, x_lb_j, x_ub_j)), \
            np.row_stack((cex, cex_lb, cex_ub))

    new_regions = []
    with Pool(n_jobs) as p:
        result_iter = tqdm(
            p.starmap(parallelizable_split,
                      ((x_regions[j], box_j) for j, box_j in sat_regions)),
            desc=f"Validate Lipshitz Constants",
            total=len(sat_regions), ascii=True, leave=False, ncols=NCOLS)

        for old_region, cex_region in result_iter:  # type: ignore
            new_regions.append(old_region)
            new_regions.append(cex_region)
    new_regions = np.stack(new_regions, axis=0)
    return np.asfarray(new_regions)


def split_region(
    region: NDArrayFloat,
    box: NDArrayFloat
) -> Optional[Tuple[NDArrayFloat, np.intp, float]]:
    assert region.shape[0] == 3
    cex_lb, cex_ub = box
    x, lb, ub = region
    if np.all(np.logical_and(cex_lb <= x, x <= cex_ub)):
        return None
        raise RuntimeError("Sampled state is inside cex box")
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
