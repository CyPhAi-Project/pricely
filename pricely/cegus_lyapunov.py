import abc
from multiprocessing import Pool, TimeoutError
import numpy as np
from numpy.typing import NDArray
from signal import signal, SIGINT
from tqdm import tqdm
from typing import Hashable, List, Literal, MutableSet, NamedTuple, Optional, Protocol, Sequence, Tuple, TypeVar

NCOLS = 120

NDArrayFloat = NDArray[np.floating]
NDArrayIndex = NDArray[np.integer]
Variable = TypeVar('Variable', contravariant=True)
Expr = TypeVar('Expr', covariant=True)
Formula = TypeVar('Formula', covariant=True)


class ROI(NamedTuple):
    """
    Define a region of interest by constraining the vector x as below:
    x_lim[0, i] <= x[i] <= x_lim[1, i]
    x_norm_lim[0] <= ‖x‖ <= x_norm_lim[1]
    abs_x_lb <= max(abs(x[i]))
    """
    x_lim: NDArrayFloat
    abs_x_lb: float
    x_norm_lim: Tuple[float, float] = (0.0, np.inf)

    def contains(self, x_values: NDArrayFloat) -> np.ndarray:
        norm_x_values = np.linalg.norm(x_values, axis=1)
        return np.all(x_values >= self.x_lim[0], axis=1) & np.all(x_values <= self.x_lim[1], axis=1) \
            & (norm_x_values >= self.x_norm_lim[0]) \
            & (norm_x_values <= self.x_norm_lim[1]) \
            & (np.max(np.abs(x_values), axis=1) >= self.abs_x_lb)


class PLyapunovCandidate(Protocol[Variable, Expr]):
    @abc.abstractmethod
    def __copy__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def lya_expr(self, x_vars: Sequence[Variable]) -> Expr:
        """Return the Lyapunov function as an expression"""
        raise NotImplementedError

    @abc.abstractmethod
    def lya_values(self, x_values: NDArrayFloat) -> NDArrayFloat:
        """Compute values of the Lyapunov function for an array of states"""
        raise NotImplementedError

    def lya_decay_rate(self) -> float:
        return 0.0  # default no decay

    @abc.abstractmethod
    def lie_der_values(self, x_values: NDArrayFloat, y_values: NDArrayFloat) -> NDArrayFloat:
        """Compute the Lie derivative values of the Lyapunov function for an array of states"""
        raise NotImplementedError

    @abc.abstractmethod
    def ctrl_exprs(self, x_vars: Sequence[Variable]) -> Sequence[Expr]:
        raise NotImplementedError

    @abc.abstractmethod
    def ctrl_values(self, x_values: NDArrayFloat) -> NDArrayFloat:
        raise NotImplementedError


class PLyapunovLearner(Protocol):
    @abc.abstractmethod
    def fit_loop(self, x: NDArrayFloat, u: NDArrayFloat, y: NDArrayFloat, **kwargs):
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_candidate(self) -> PLyapunovCandidate:
        raise NotImplementedError


class PLocalApprox(Protocol[Expr, Formula]):
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
    def num_samples(self) -> int:
        "Number of all samples. Samples may include states out of ROI."
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def samples(self) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        "All samples as three arrays of the same length in the order of (x, u, y)"
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_samples_in_roi(self) -> int:
        "Number of samples of which states are in ROI."
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def samples_in_roi(self) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        "Samples with states in ROI as three arrays of the same length in the order of (x, u, y)"
        raise NotImplementedError

    @abc.abstractmethod
    def add(self, cex_boxes: Sequence[Tuple[int, NDArrayFloat]], cand: PLyapunovCandidate)\
            -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        raise NotImplementedError


class PLyapunovVerifier(Protocol):
    @property
    @abc.abstractmethod
    def x_dim(self) -> int:
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
    return verifier.find_cex(f_approx_j)


class DeltaProverResult(NamedTuple):
    status: Literal["FOUND", "FALSIFIED", "SAMPLE_LIMIT", "PRECISION_LIMIT"]
    cex_regions: Sequence[Tuple[int, NDArrayFloat]]


class CEGuSResult(NamedTuple):
    status: Literal["FOUND", "NO_CANDIDATE", "EPOCH_LIMIT", "SAMPLE_LIMIT", "PRECISION_LIMIT"]
    epoch: int
    approx: PApproxDynamic
    cex_regions: Sequence[Tuple[int, NDArrayFloat]]


def verify_delta_provability(
        verifier: PLyapunovVerifier,
        curr_approx: PApproxDynamic,
        cand: PLyapunovCandidate,
        diam_lb: float,
        max_num_samples: int = 5*10**5,
        n_jobs: int = 1,
        timeout_per_job: Optional[float] = 30.0):
    refinement_pbar = tqdm(
        desc=f"Refinement Loop", initial=-1,
        ascii=True, leave=None, position=1, ncols=NCOLS)
    verified_regions: MutableSet[Hashable] = set()
    stop_refine = False
    cex_regions: List[Tuple[int, NDArrayFloat]] = []
    while not stop_refine:
        refinement_pbar.update(1)

        new_verified_regions = set()
        cex_regions.clear()
        with Pool(n_jobs) as p:
            # XXX Need to store and maintain the index j for retriving the same region again
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

            num_timeouts = 0
            for j, future in tqdm(future_list,
                                  desc=f"Verify", ascii=True, leave=None, position=2, ncols=NCOLS):
                try:
                    box = future.get(timeout_per_job)
                except TimeoutError:
                    num_timeouts += 1
                    cex = curr_approx[j].x_witness
                    box = np.vstack((cex, cex))

                if box is None:
                    new_verified_regions.add(curr_approx[j].in_domain_repr())
                else:
                    stop_refine = stop_refine or (curr_approx[j].domain_diameter <= diam_lb)
                    cex_regions.append((j, box))
        verified_regions.clear()
        verified_regions = new_verified_regions

        if num_timeouts > 0:
            tqdm.write(f'Regional verifier times out for {num_timeouts} regions')
        refinement_pbar.set_postfix({
            "#Total Regions": len(curr_approx),
            "#Valid Regions": len(curr_approx) - len(cex_regions),
            "#VC Queries": len(future_list),
            "#Samples": curr_approx.num_samples})
        if len(cex_regions) == 0:
            assert num_timeouts == 0
            # Lyapunov function candidate passed
            return DeltaProverResult("FOUND", cex_regions)
        if curr_approx.num_samples + len(cex_regions) >= max_num_samples:
            tqdm.write(f"Exceeding max number of samples {max_num_samples} in next iteration.")
            return DeltaProverResult("SAMPLE_LIMIT", cex_regions)

        # Update the cover with counterexamples
        x_values, u_values, y_values = curr_approx.add(cex_regions, cand)

        if np.any(cand.lya_values(x_values) <= 0.0) or \
                np.any(cand.lie_der_values(x_values, y_values) >= 0.0):
            # Found true counterexamples. Break to learn a new candidate.
            return DeltaProverResult("FALSIFIED", cex_regions)
        # End of the while-loop

    # stop_refine == True
    refinement_pbar.set_postfix({
        "#Valid Regions": "?",
        "#Total Regions": len(curr_approx),
        "#Samples": curr_approx.num_samples})
    tqdm.write("Current candidate is neither δ-provable nor falsified.")
    return DeltaProverResult("PRECISION_LIMIT", cex_regions)


def cegus_lyapunov(
        learner: PLyapunovLearner,
        verifier: PLyapunovVerifier,
        init_approx: PApproxDynamic,
        delta: float = 1e-4,
        max_epochs: int = 10,
        max_iter_learn: int = 10,
        max_num_samples: int = 5*10**5,
        n_jobs: int = 1,
        timeout_per_job: Optional[float] = 30.0) -> CEGuSResult:
    assert max_epochs > 0

    # TODO the diameter threshold can be larger for a cover using AxisAlignedBoxes
    diam_lb = delta * np.sqrt(2.0 + 2.0 / verifier.x_dim) / 2.0
    # Initial set cover and sampled values
    curr_approx = init_approx

    outer_pbar = tqdm(
        iter(range(max_epochs)),
        desc="CEGuS Loop", ascii=True, leave=True, position=0, ncols=NCOLS,
        postfix={"#Samples": curr_approx.num_samples})
    cex_regions: Sequence[Tuple[int, NDArrayFloat]] = []
    obj_values = []

    for epoch in outer_pbar:
        ## Prepare dataset for learning
        x_values, u_values, y_values = curr_approx.samples_in_roi
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

        res = verify_delta_provability(
            verifier, curr_approx, cand,
            diam_lb, max_num_samples, n_jobs, timeout_per_job)
        cex_regions = res.cex_regions
        if res.status != "FALSIFIED":
            return CEGuSResult(res.status, epoch, curr_approx, cex_regions)
        # else: continue

    outer_pbar.close()

    tqdm.write(f"Cannot find a Lyapunov function in {max_epochs} iterations.")
    return CEGuSResult("EPOCH_LIMIT", max_epochs, curr_approx, cex_regions)
