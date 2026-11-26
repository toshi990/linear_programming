import argparse
import csv
import os
from pathlib import Path
import platform
import sys

from data_io import ROOT, load_data, save_json
from experiments import SCENARIOS, scenario_data, sensitivity_experiments, solve_and_check
from model import build_dual, solve_lp
from verification import capacity_conflict, check_dual


def save_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_plan(data, result):
    n = len(data["products"])
    print("Объёмы закупки, тыс. шт. за неделю:")
    print(f"{'Поставщик':<12}" + "".join(f"{name:>12}" for name in data["products"]))
    for i, supplier in enumerate(data["suppliers"]):
        print(f"{supplier:<12}" + "".join(f"{v:12.3f}" for v in result["x"][i*n:(i+1)*n]))


def save_solution(data, result, folder):
    n = len(data["products"])
    plan = [{"supplier": supplier, "product": product,
             "quantity_thousand": result["x"][i*n+j],
             "cost_per_thousand": data["cost"][i][j],
             "cost_thousand_rub": result["x"][i*n+j] * data["cost"][i][j]}
            for i, supplier in enumerate(data["suppliers"])
            for j, product in enumerate(data["products"])]
    save_csv(folder / "solution.csv", plan)
    save_csv(folder / "constraints.csv", result["check"]["constraints"])
    save_json(folder / "verification.json", result["check"])


def draw_plots(data, results, curve, output):

    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output / ".cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    feasible = [(name, value) for name, value in results.items() if value["status"] == "optimal"]
    if feasible:
        fig, axes = plt.subplots(1, len(feasible), figsize=(5 * len(feasible), 3.6), squeeze=False)
        colors = ["#365a7c", "#729a8a", "#c3a162", "#b77c73"]
        for ax, (name, result) in zip(axes.ravel(), feasible):
            x = np.asarray(result["x"]).reshape(len(data["suppliers"]), -1)
            bottom = np.zeros(len(data["suppliers"]))
            for j, product in enumerate(data["products"]):
                ax.bar(data["suppliers"], x[:, j], bottom=bottom, label=product, color=colors[j % 4])
                bottom += x[:, j]
            ax.set_title(SCENARIOS[name])
            ax.set_ylabel("Закупка, тыс. шт. за неделю")
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
        fig.tight_layout(rect=(0, 0.10, 1, 1))
        fig.savefig(figures / "purchase_plan.png", dpi=180)
        plt.close(fig)
    if curve:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        valid = [row for row in curve if row["status"] == "optimal"]
        ax.plot([r["capacity"] for r in valid], [r["objective"] for r in valid],
                "o-", color="#365a7c", label="Повторное решение")
        ax.plot([r["capacity"] for r in curve], [r["linear_prediction"] for r in curve],
                "--", color="#a65b4f", label="Прогноз по базовой теневой цене")
        ax.axvline(data["capacity"][0], color="gray", alpha=0.4)
        ax.set_xlabel("Квота П1, тыс. шт. за неделю")
        ax.set_ylabel("Затраты, тыс. руб. за неделю")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figures / "shadow_price.png", dpi=180)
        plt.close(fig)


def interactive_scenario():
    print("0 — все сценарии; 1 — базовый; 2 — дефицитный; 3 — стрессовый")
    choices = {"0": "all", "1": "base", "2": "deficit", "3": "stress"}
    while True:
        choice = input("Выберите сценарий: ").strip()
        if choice in choices:
            return choices[choice]
        print("Введите 0, 1, 2 или 3.")


def run(data, scenario, output):
    output.mkdir(parents=True, exist_ok=True)
    names = list(SCENARIOS) if scenario == "all" else [scenario]
    print(f"ЛР 1. Вариант {data['variant']}; N={data['student']['N']}; seed={data['seed']}.")
    print("Цель: покрыть недельную потребность в упаковке с минимальными затратами.")
    results, summary = {}, []
    for name in names:
        changed = scenario_data(data, name)
        folder = output / name
        save_json(folder / "input.json", changed)
        result = solve_and_check(changed, folder / "solver.log")
        results[name] = result
        if result["status"] != "optimal":
            for filename in ("solution.csv", "constraints.csv", "verification.json",
                             "shadow_prices.csv", "dual_result.json", "dual_verification.json",
                             "dual_solver.log", "dual_solver.lp"):
                (folder / filename).unlink(missing_ok=True)
        else:
            (folder / "infeasibility.json").unlink(missing_ok=True)
        print(f"\n{SCENARIOS[name]}: {result['status']}")
        if result["status"] == "optimal":
            print(f"Минимальные затраты: {result['objective']:.3f} тыс. руб. за неделю.")
            print_plan(changed, result)
            print(f"Проверка: пройдена; максимальное нарушение "
                  f"{result['check']['max_violation']:.2e}, "
                  f"ошибка цели {result['check']['objective_error']:.2e}.")
            save_solution(changed, result, folder)
            dual = solve_lp(build_dual(changed), changed["seed"], folder / "dual_solver.log")
            if dual["status"] != "optimal":
                raise RuntimeError("Не найден оптимум двойственной задачи.")
            certificate = check_dual(changed, result["x"], dual["x"],
                                     result["objective"], dual["objective"])
            if not certificate["ok"]:
                raise RuntimeError("Проверка двойственности не пройдена.")
            result["dual_check"] = certificate
            save_json(folder / "dual_verification.json", certificate)
            save_json(folder / "dual_result.json", dual)
            m = len(data["suppliers"])
            save_csv(folder / "shadow_prices.csv", [
                {"type": "capacity", "name": supplier,
                 "shadow_price": result["row_dual"][i], "unit": "тыс. руб. / тыс. шт."}
                for i, supplier in enumerate(data["suppliers"])
            ] + [
                {"type": "demand", "name": product,
                 "shadow_price": result["row_dual"][m+j], "unit": "тыс. руб. / тыс. шт."}
                for j, product in enumerate(data["products"])
            ])
            print(f"Цель двойственной задачи: {dual['objective']:.3f}; "
                  f"разрыв {certificate['duality_gap']:.2e}.")
        elif result["status"] == "infeasible":
            conflict = capacity_conflict(changed)
            result["conflict"] = conflict
            save_json(folder / "infeasibility.json", conflict)
            print("Допустимого плана закупок нет.")
            if conflict["infeasibility_proved"]:
                print(f"Общий спрос {conflict['total_demand']:.3f} > "
                      f"общая квота {conflict['total_capacity']:.3f} тыс. шт.; "
                      f"нехватка {conflict['shortage']:.3f} тыс. шт.")
        elif result["status"] == "unbounded":
            print("Цель не ограничена. Проверьте знаки цен и ограничений.")
        elif result["status"] == "time_limit":
            print("Истекло время решения; оптимальность не подтверждена.")
        else:
            raise RuntimeError(f"Решение остановлено: {result['status']}")
        save_json(folder / "result.json", result)
        summary.append({"scenario": name, "status": result["status"],
                        "capacity_total": sum(changed["capacity"]),
                        "demand_total": sum(changed["demand"]),
                        "objective_thousand_rub": result["objective"],
                        "iterations": result["iterations"], "seconds": result["seconds"]})
    curve = []
    if "base" in results and results["base"]["status"] == "optimal":
        marginal, curve = sensitivity_experiments(data, results["base"], output)
        save_csv(output / "marginal_checks.csv", marginal)
        save_csv(output / "capacity_curve.csv", curve)
    save_csv(output / "scenarios.csv", summary)
    save_json(output / "summary.json", results)
    save_json(output / "environment.json", {"python": platform.python_version(),
              "highs": next(iter(results.values()))["highs_version"]})
    draw_plots(data, results, curve, output)
    print(f"\nРешения, проверки, сценарии, графики и логи сохранены: {output}")


def main():
    parser = argparse.ArgumentParser(description="Закупка упаковки у пяти поставщиков")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "variant11.json",
                        help="JSON с ценами, потребностями и квотами")
    parser.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--interactive", action="store_true", help="Выбор сценария с клавиатуры")
    args = parser.parse_args()
    try:
        data = load_data(args.data)
        scenario = interactive_scenario() if args.interactive else args.scenario
        run(data, scenario, args.output.resolve())
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nВвод завершён.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
