from copy import deepcopy

import numpy as np

from model import build_primal, solve_lp
from verification import check_primal

SCENARIOS = {"base": "Базовый", "deficit": "Дефицитный", "stress": "Стрессовый"}


def scenario_data(data, name):
    changed = deepcopy(data)
    if name in ("deficit", "stress"):
        changed["capacity"][0] *= 0.8
    if name == "stress":
        changed["demand"] = [value * 1.2 for value in data["demand"]]
    return changed


def solve_and_check(data, log_path):
    result = solve_lp(build_primal(data), data["seed"], log_path)
    if result["status"] == "optimal":
        result["check"] = check_primal(data, result["x"], result["objective"])
        if not result["check"]["ok"]:
            raise RuntimeError("Независимая проверка решения не пройдена.")
    return result


def sensitivity_experiments(data, base_result, output):
    m = len(data["suppliers"])

    prices = np.asarray(base_result["row_dual"])
    rows = []
    for field, names, offset in [("capacity", data["suppliers"], 0),
                                  ("demand", data["products"], m)]:
        for index, name in enumerate(names):
            changed = deepcopy(data)
            changed[field][index] += 1.0
            result = solve_and_check(changed, output / "logs" / f"{field}_{index + 1}_plus1.log")
            forecast = float(prices[offset + index])
            actual = (result["objective"] - base_result["objective"]
                      if result["status"] == "optimal" else None)
            rows.append({"parameter": field, "name": name, "change": 1.0,
                         "shadow_price": forecast, "predicted_change": forecast,
                         "actual_change": actual,
                         "prediction_error": None if actual is None else actual - forecast,
                         "status": result["status"]})
    curve = []
    for factor in np.linspace(0.6, 1.4, 17):
        changed = deepcopy(data)
        changed["capacity"][0] = data["capacity"][0] * float(factor)
        result = solve_and_check(changed, output / "logs" / f"capacity1_{factor:.2f}.log")
        prediction = (base_result["objective"]
                      + prices[0] * (changed["capacity"][0] - data["capacity"][0]))
        curve.append({"factor": float(factor), "capacity": changed["capacity"][0],
                      "status": result["status"], "objective": result["objective"],
                      "linear_prediction": float(prediction)})
    return rows, curve
