from pymoo.core.problem import Problem
from pymoo.core.repair import Repair

from app.domain.simulator import Evaluator


class RouteProblem(Problem):
    def __init__(self, evaluator: Evaluator) -> None:
        layout = evaluator.layout
        super().__init__(
            n_var=layout.n_var,
            n_obj=3,
            n_ieq_constr=evaluator.n_constr,
            xl=layout.xl,
            xu=layout.xu,
        )
        self.evaluator = evaluator

    def _evaluate(self, X, out, *args, **kwargs):
        F, G = self.evaluator.evaluate_many(X)
        out["F"] = F
        out["G"] = G


class RouteRepair(Repair):
    def __init__(self, evaluator: Evaluator) -> None:
        super().__init__()
        self.layout = evaluator.layout

    def _do(self, problem, X, **kwargs):
        return self.layout.repair(X)
