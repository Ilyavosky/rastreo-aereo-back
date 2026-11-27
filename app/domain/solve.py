import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.gauss import GaussianMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination

from app.domain.problem import RouteProblem, RouteRepair
from app.domain.simulator import Evaluator

HYPERVOLUME_REFERENCE = 1.1


@dataclass
class GenerationStats:
    gen: int
    n_feasible: int
    best_r_de: float | None
    best_t_mission_min: float | None
    best_e_total_wh: float | None
    hypervolume: float | None
    evaluations: int = 0
    elapsed_s: float = 0.0
    n_front: int = 0
    mean_r_de: float | None = None
    mean_t_mission_min: float | None = None
    mean_e_total_wh: float | None = None


@dataclass
class OptimizationResult:
    X: np.ndarray
    F: np.ndarray
    G: np.ndarray
    feasible: np.ndarray
    generations_done: int


class HypervolumeTracker:
    def __init__(self) -> None:
        self.low: np.ndarray | None = None
        self.span: np.ndarray | None = None

    def __call__(self, F: np.ndarray) -> float | None:
        if len(F) == 0:
            return None
        if self.low is None:
            self.low = F.min(axis=0)
            high = F.max(axis=0)
            self.span = np.where(high > self.low, high - self.low, 1.0)
        normalized = (F - self.low) / self.span
        indicator = HV(ref_point=np.full(3, HYPERVOLUME_REFERENCE))
        return float(indicator(normalized))


def optimize(
    evaluator: Evaluator,
    on_generation: Callable[[GenerationStats], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> OptimizationResult | None:
    params = evaluator.params
    problem = RouteProblem(evaluator)
    algorithm = NSGA2(
        pop_size=params.population_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=params.crossover_prob, eta=params.crossover_eta),
        mutation=GaussianMutation(sigma=params.mutation_sigma),
        repair=RouteRepair(evaluator),
        eliminate_duplicates=True,
    )
    algorithm.setup(
        problem,
        termination=get_termination("n_gen", params.generations),
        seed=params.seed,
        verbose=False,
    )
    tracker = HypervolumeTracker()
    started = time.perf_counter()
    while algorithm.has_next():
        algorithm.next()
        if on_generation is not None:
            on_generation(_stats(algorithm, tracker, time.perf_counter() - started))
        if should_stop is not None and should_stop():
            return None
    optimum = algorithm.opt
    return OptimizationResult(
        X=optimum.get("X"),
        F=optimum.get("F"),
        G=optimum.get("G"),
        feasible=optimum.get("FEAS")[:, 0],
        generations_done=algorithm.n_gen - 1,
    )


def _stats(algorithm, tracker: HypervolumeTracker, elapsed_s: float) -> GenerationStats:
    pop = algorithm.pop
    gen = algorithm.n_gen - 1
    evaluations = int(algorithm.evaluator.n_eval)
    feasible = pop.get("FEAS")[:, 0]
    F = pop.get("F")[feasible]
    if len(F) == 0:
        return GenerationStats(gen, 0, None, None, None, None, evaluations, elapsed_s, 0)
    rank = pop.get("rank")[feasible]
    return GenerationStats(
        gen=gen,
        n_feasible=int(feasible.sum()),
        best_r_de=float(-F[:, 0].min()),
        best_t_mission_min=float(F[:, 1].min()),
        best_e_total_wh=float(F[:, 2].min()),
        hypervolume=tracker(F),
        evaluations=evaluations,
        elapsed_s=elapsed_s,
        n_front=int((rank == 0).sum()),
        mean_r_de=float(-F[:, 0].mean()),
        mean_t_mission_min=float(F[:, 1].mean()),
        mean_e_total_wh=float(F[:, 2].mean()),
    )
