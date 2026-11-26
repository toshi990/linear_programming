from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from data_io import generate_data, load_data, save_json
from experiments import scenario_data
from model import build_dual, build_primal, solve_lp
from verification import TOL, capacity_conflict, check_dual, check_primal


class Lab1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = generate_data()
        cls.primal = solve_lp(build_primal(cls.data), cls.data["seed"])
        cls.dual = solve_lp(build_dual(cls.data), cls.data["seed"])

    def test_variant_and_reproducibility(self):
        self.assertEqual(self.data["variant"], 11)
        self.assertEqual(self.data["seed"], 10121)
        self.assertEqual(self.data, generate_data())

    def test_optimum_has_independent_certificate(self):
        self.assertEqual(self.primal["status"], "optimal")
        self.assertEqual(self.dual["status"], "optimal")
        self.assertTrue(check_primal(self.data, self.primal["x"],
                                     self.primal["objective"])["ok"])
        proof = check_dual(self.data, self.primal["x"], self.dual["x"],
                           self.primal["objective"], self.dual["objective"])
        self.assertTrue(proof["ok"])
        self.assertLessEqual(proof["duality_gap"], TOL)

    def test_checker_rejects_wrong_objective(self):
        result = check_primal(self.data, self.primal["x"], self.primal["objective"] + 1)
        self.assertFalse(result["ok"])

    def test_checker_rejects_infeasible_vectors(self):
        for x in [np.zeros(20), np.full(20, 1000.0), np.full(20, -1.0)]:
            with self.subTest(x=x[0]):
                value = float(np.asarray(self.data["cost"]).ravel() @ x)
                self.assertFalse(check_primal(self.data, x, value)["ok"])

    def test_checker_rejects_wrong_dual_solution(self):
        z = list(self.dual["x"])
        z[5] += 1
        proof = check_dual(self.data, self.primal["x"], z,
                           self.primal["objective"], self.dual["objective"])
        self.assertFalse(proof["ok"])

    def test_deficit_reallocates_and_raises_cost(self):
        data = scenario_data(self.data, "deficit")
        result = solve_lp(build_primal(data), data["seed"])
        self.assertEqual(result["status"], "optimal")
        self.assertGreater(result["objective"], self.primal["objective"])
        self.assertTrue(check_primal(data, result["x"], result["objective"])["ok"])
        self.assertFalse(np.allclose(result["x"], self.primal["x"]))

    def test_stress_infeasibility_is_proved_by_data(self):
        data = scenario_data(self.data, "stress")
        result = solve_lp(build_primal(data), data["seed"])
        self.assertEqual(result["status"], "infeasible")
        self.assertNotIn("x", result)
        proof = capacity_conflict(data)
        self.assertTrue(proof["infeasibility_proved"])
        self.assertAlmostEqual(proof["shortage"], 40.6)

    def test_shadow_price_for_small_change(self):
        data = deepcopy(self.data)
        data["capacity"][0] += 1
        result = solve_lp(build_primal(data), data["seed"])
        difference = result["objective"] - self.primal["objective"]
        self.assertAlmostEqual(difference, self.primal["row_dual"][0], places=7)

    def test_alternative_optimum_is_also_valid(self):

        x = np.asarray(self.primal["x"]).reshape(5, 4).copy()
        x[3, 2] -= 1
        x[4, 2] += 1
        self.assertTrue(check_primal(self.data, x, self.primal["objective"])["ok"])
        self.assertTrue(check_dual(self.data, x, self.dual["x"],
                                   self.primal["objective"], self.dual["objective"])["ok"])

    def test_data_validation(self):
        bad_values = [("cost", [[1]]), ("capacity", [-1, 1, 1, 1, 1]),
                      ("demand", [1, 2, 3, float("inf")])]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.json"
            save_json(path, self.data)
            self.assertEqual(load_data(path), self.data)
            for key, value in bad_values:
                with self.subTest(key=key):
                    data = deepcopy(self.data)
                    data[key] = value

                    import json
                    path.write_text(json.dumps(data), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_data(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
