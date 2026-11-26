from pathlib import Path
from time import perf_counter

import highspy
import numpy as np


def build_primal(data):
    cost = np.asarray(data["cost"], dtype=float)
    m, n = cost.shape

    matrix = np.zeros((m + n, m * n))
    for i in range(m):
        matrix[i, i * n:(i + 1) * n] = 1
    for j in range(n):
        matrix[m + j, j::n] = 1
    lower = np.r_[np.full(m, -np.inf), data["demand"]]
    upper = np.r_[data["capacity"], np.full(n, np.inf)]
    return cost.ravel(), matrix, lower, upper, False


def build_dual(data):
    cost = np.asarray(data["cost"], dtype=float)
    m, n = cost.shape


    matrix = np.zeros((m * n, m + n))
    for i in range(m):
        for j in range(n):
            matrix[i * n + j, i] = -1
            matrix[i * n + j, m + j] = 1
    objective = np.r_[-np.asarray(data["capacity"]), data["demand"]]
    return objective, matrix, np.full(m * n, -np.inf), cost.ravel(), True


def solve_lp(model, seed, log_path=None):
    objective, matrix, lower, upper, maximize = model
    lp = highspy.HighsLp()
    lp.num_col_ = len(objective)
    lp.num_row_ = len(lower)
    lp.col_cost_ = np.asarray(objective, dtype=float)
    lp.col_lower_ = np.zeros(len(objective))
    lp.col_upper_ = np.full(len(objective), highspy.kHighsInf)
    lp.row_lower_ = lower
    lp.row_upper_ = upper
    lp.sense_ = highspy.ObjSense.kMaximize if maximize else highspy.ObjSense.kMinimize
    rows, cols = np.nonzero(matrix)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = np.r_[0, np.cumsum(np.bincount(rows, minlength=len(lower)))]
    lp.a_matrix_.index_ = cols.astype(np.int32)
    lp.a_matrix_.value_ = matrix[rows, cols]

    solver = highspy.Highs()
    options = {"solver": "simplex", "simplex_strategy": 1, "presolve": "on",
               "threads": 1, "random_seed": seed, "time_limit": 30.0,
               "primal_feasibility_tolerance": 1e-9,
               "dual_feasibility_tolerance": 1e-9, "log_to_console": False,
               "output_flag": log_path is not None}
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
        options["log_file"] = str(log_path)
    for name, value in options.items():
        if solver.setOptionValue(name, value) != highspy.HighsStatus.kOk:
            raise RuntimeError(f"Не удалось задать параметр HiGHS {name}.")
    if solver.passModel(lp) != highspy.HighsStatus.kOk:
        raise RuntimeError("HiGHS не принял модель.")
    if log_path is not None:
        solver.writeModel(str(log_path.with_suffix(".lp")))
    start = perf_counter()
    run_status = solver.run()
    elapsed = perf_counter() - start
    if run_status == highspy.HighsStatus.kError:
        raise RuntimeError("Ошибка выполнения HiGHS; см. лог.")
    status = solver.getModelStatus()
    names = {highspy.HighsModelStatus.kOptimal: "optimal",
             highspy.HighsModelStatus.kInfeasible: "infeasible",
             highspy.HighsModelStatus.kUnbounded: "unbounded",
             highspy.HighsModelStatus.kTimeLimit: "time_limit"}
    result = {"status": names.get(status, solver.modelStatusToString(status)),
              "seconds": elapsed, "iterations": solver.getInfo().simplex_iteration_count,
              "highs_version": solver.version(), "objective": None}

    if status == highspy.HighsModelStatus.kOptimal:
        solution = solver.getSolution()
        result.update(x=list(solution.col_value), row_dual=list(solution.row_dual),
                      objective=solver.getObjectiveValue())
    return result
