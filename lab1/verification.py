import numpy as np

TOL = 1e-7


def check_primal(data, x, solver_objective):
    cost = np.asarray(data["cost"], dtype=float)
    x = np.asarray(x, dtype=float).reshape(cost.shape)
    if not np.isfinite(x).all() or not np.isfinite(solver_objective):
        raise ValueError("В решении есть нечисловые или бесконечные значения.")
    capacity, demand = np.asarray(data["capacity"]), np.asarray(data["demand"])
    used, received = x.sum(axis=1), x.sum(axis=0)
    objective = float(np.sum(cost * x))
    rows = []
    for kind, names, lhs, rhs in [
        ("capacity", data["suppliers"], used, capacity),
        ("demand", data["products"], received, demand),
    ]:
        for name, left, right in zip(names, lhs, rhs):
            slack = right - left if kind == "capacity" else left - right
            rows.append({"type": kind, "name": name, "lhs": float(left),
                         "sense": "<=" if kind == "capacity" else ">=",
                         "rhs": float(right), "slack": float(slack),
                         "violation": float(max(0, -slack)), "ok": bool(slack >= -TOL)})
    for i, supplier in enumerate(data["suppliers"]):
        for j, product in enumerate(data["products"]):
            rows.append({"type": "nonnegative", "name": f"{supplier} / {product}",
                         "lhs": float(x[i, j]), "sense": ">=", "rhs": 0.0,
                         "slack": float(x[i, j]), "violation": float(max(0, -x[i, j])),
                         "ok": bool(x[i, j] >= -TOL)})
    error = abs(objective - solver_objective)
    return {"ok": all(row["ok"] for row in rows) and error <= TOL,
            "tolerance": TOL, "objective_recalculated": objective,
            "objective_error": error, "max_violation": max(r["violation"] for r in rows),
            "constraints": rows}


def check_dual(data, x, dual_x, primal_objective, dual_objective):
    cost = np.asarray(data["cost"], dtype=float)
    x = np.asarray(x, dtype=float).reshape(cost.shape)
    m, n = cost.shape
    z = np.asarray(dual_x, dtype=float)
    if z.shape != (m + n,) or not np.isfinite(z).all():
        raise ValueError("Некорректное решение двойственной задачи.")
    u, v = z[:m], z[m:]
    capacity, demand = np.asarray(data["capacity"]), np.asarray(data["demand"])
    reduced = cost + u[:, None] - v[None, :]
    value = float(demand @ v - capacity @ u)
    products = np.r_[x.ravel() * reduced.ravel(),
                     u * (capacity - x.sum(axis=1)),
                     v * (x.sum(axis=0) - demand)]
    gap = abs(primal_objective - value)
    error = abs(dual_objective - value)
    violation = float(max(0.0, -float(z.min()), -float(reduced.min())))
    complementarity = float(np.max(np.abs(products)))
    return {"ok": max(gap, error, violation, complementarity) <= TOL,
            "tolerance": TOL, "u": u.tolist(), "v": v.tolist(),
            "dual_objective_recalculated": value, "duality_gap": gap,
            "dual_objective_error": error, "max_dual_violation": violation,
            "max_complementarity_error": complementarity,
            "reduced_costs": reduced.tolist()}


def capacity_conflict(data):

    capacity, demand = float(sum(data["capacity"])), float(sum(data["demand"]))
    return {"total_capacity": capacity, "total_demand": demand,
            "shortage": max(0.0, demand - capacity),
            "infeasibility_proved": demand > capacity + TOL,
            "reason": "Сумма закупок должна быть не меньше общего спроса "
                      "и не больше суммы квот поставщиков."}
