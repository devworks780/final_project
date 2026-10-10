"""Unit-тесты inference без запуска HTTP-сервера."""

import unittest
from pathlib import Path

from app.recommender import PopularityRecommender

MODEL_PATH = Path("dvc_data/data/baseline/model.json")


class RecommenderTest(unittest.TestCase):
    """Проверить основные инварианты рекомендаций."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = PopularityRecommender(MODEL_PATH)

    def test_owned_products_are_excluded(self) -> None:
        owned = {"ind_recibo_ult1", "ind_cco_fin_ult1"}
        result = self.model.recommend("01", owned, k=7)
        self.assertEqual(len(result), 7)
        self.assertTrue(owned.isdisjoint(item.product for item in result))

    def test_unknown_segment_uses_global_fallback(self) -> None:
        unknown = self.model.recommend("NOT_A_SEGMENT", set(), k=3)
        fallback = self.model.recommend("UNKNOWN_GLOBAL", set(), k=3)
        self.assertEqual(unknown, fallback)

    def test_unknown_product_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown products"):
            self.model.recommend("01", {"not_a_product"}, k=7)


if __name__ == "__main__":
    unittest.main()
