import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def generate_data(number=21, group=517, year=26):
    group_indices = {507: 0, 517: 1, 527: 2}
    if group not in group_indices or number < 1:
        raise ValueError("Укажите группу 507, 517 или 527 и положительный номер.")
    q = group_indices[group]
    variant = 1 + (17 * number + 7 * q + year) % 20
    if variant != 11:
        raise ValueError(f"Получен вариант {variant}; программа описывает вариант 11.")
    seed = 10000 + 100 * q + number
    rng = np.random.default_rng(seed)
    demand = rng.integers(45, 86, size=4)

    weights = rng.integers(85, 116, size=5)
    capacity = np.ceil(1.10 * demand.sum() * weights / weights.sum()).astype(int)

    typical_cost = np.array([
        [19.5, 8.1, 4.9, 6.4],
        [20.3, 7.3, 5.4, 6.0],
        [22.0, 8.0, 4.6, 5.7],
        [24.5, 8.7, 5.8, 7.1],
        [27.0, 9.3, 6.3, 7.8],
    ])
    cost = np.round(typical_cost + rng.uniform(-0.3, 0.3, size=(5, 4)), 2)
    return {
        "student": {"group": group, "N": number, "Q": q, "Y": year},
        "variant": variant,
        "seed": seed,
        "suppliers": [f"П{i + 1}" for i in range(5)],
        "products": ["Коробки", "Пакеты", "Лотки", "Стаканы"],
        "quantity_unit": "тыс. шт. за неделю",
        "cost_unit": "тыс. руб. за тыс. шт.",
        "demand": demand.tolist(),
        "capacity": capacity.tolist(),
        "cost": cost.tolist(),
    }


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n", encoding="utf-8")


def load_data(path):
    with Path(path).open(encoding="utf-8") as file:
        data = json.load(file)
    suppliers, products = data["suppliers"], data["products"]
    m, n = len(suppliers), len(products)
    if m < 5 or n < 4 or len(set(suppliers)) != m or len(set(products)) != n:
        raise ValueError("Нужны не менее пяти поставщиков и четырёх видов с разными именами.")
    for key, shape in [("cost", (m, n)), ("capacity", (m,)), ("demand", (n,))]:
        value = np.asarray(data[key], dtype=float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Некорректные размеры или числа в поле {key}.")
        if (value < 0).any() or (key == "cost" and (value <= 0).any()):
            raise ValueError("Цены должны быть положительными, объёмы — неотрицательными.")
    seed = data["seed"]
    if not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("Seed должен быть целым числом от 0 до 2^31 - 1.")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генерация данных варианта 11")
    parser.add_argument("--number", type=int, default=21)
    parser.add_argument("--group", type=int, default=517)
    parser.add_argument("--year", type=int, default=26)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "variant11.json")
    args = parser.parse_args()
    try:
        result = generate_data(args.number, args.group, args.year)
        save_json(args.output, result)
    except ValueError as error:
        parser.error(str(error))
    print(f"Вариант {result['variant']}, seed {result['seed']}: {args.output}")
