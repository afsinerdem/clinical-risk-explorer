from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from modeling import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_patient_vs_cohort_explanation,
    compute_metrics,
    find_nearest_patients,
    load_artifacts,
    load_data,
    run_model_selection,
    save_training_artifacts,
    select_threshold,
    split_data,
    validate_data,
)


class ModelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_data(Path("framingham.csv"))

    def test_validate_data_removes_invalid_rows(self) -> None:
        sample = self.data.head(12).copy()
        sample.loc[0, "age"] = 5
        sample.loc[1, TARGET_COLUMN] = 3

        cleaned, report = validate_data(sample)

        self.assertEqual(report.dropped_rows, 2)
        self.assertEqual(len(cleaned), 10)
        self.assertGreaterEqual(report.issue_counts["age_out_of_range"], 1)
        self.assertGreaterEqual(report.issue_counts["target_not_binary"], 1)

    def test_split_data_keeps_stratified_ratio(self) -> None:
        cleaned, _ = validate_data(self.data)
        X_train, X_test, y_train, y_test = split_data(cleaned)

        self.assertEqual(len(X_train) + len(X_test), len(cleaned))
        self.assertEqual(len(X_test), int(round(len(cleaned) * 0.2)))
        self.assertAlmostEqual(float(y_train.mean()), float(y_test.mean()), delta=0.02)

    def test_select_threshold_returns_probability_cutoff(self) -> None:
        y_true = pd.Series([0, 0, 0, 1, 1, 1])
        probabilities = pd.Series([0.05, 0.2, 0.3, 0.35, 0.7, 0.9]).to_numpy()

        threshold = select_threshold(y_true, probabilities)

        self.assertGreaterEqual(threshold, 0.01)
        self.assertLessEqual(threshold, 0.99)

    def test_compute_metrics_includes_clinical_fields(self) -> None:
        y_true = pd.Series([0, 0, 1, 1])
        probabilities = pd.Series([0.1, 0.2, 0.7, 0.9]).to_numpy()

        metrics = compute_metrics(y_true, probabilities, threshold=0.5)

        for key in [
            "pr_auc",
            "roc_auc",
            "recall",
            "specificity",
            "ppv",
            "npv",
            "balanced_accuracy",
            "brier_score",
            "accuracy",
            "f2",
        ]:
            self.assertIn(key, metrics)

    def test_training_pipeline_writes_required_artifacts(self) -> None:
        sample = (
            self.data.groupby(TARGET_COLUMN, group_keys=False)
            .sample(n=180, random_state=42)
            .reset_index(drop=True)
        )

        artifacts = run_model_selection(sample)
        self.assertIn(artifacts.metadata["selected_model"], {"logistic_regression", "random_forest", "hist_gradient_boosting"})
        self.assertEqual(artifacts.metadata["feature_columns"], FEATURE_COLUMNS)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            save_training_artifacts(artifacts, path)
            loaded = load_artifacts(path)

            self.assertEqual(loaded.metadata["selected_model"], artifacts.metadata["selected_model"])
            self.assertEqual(loaded.metadata["feature_columns"], FEATURE_COLUMNS)

    def test_nearest_neighbors_returns_top50_and_closest(self) -> None:
        input_frame = pd.DataFrame([self.data.iloc[0][FEATURE_COLUMNS].to_dict()])

        result = find_nearest_patients(input_frame, self.data, top_k=50)

        self.assertEqual(len(result["neighbors"]), 50)
        self.assertIn("closest", result)
        self.assertGreaterEqual(float(result["closest"]["similarity_pct"]), float(result["neighbors"]["similarity_pct"].iloc[-1]))

    def test_patient_vs_cohort_explanation_filters_missing_indicators(self) -> None:
        sample_input = pd.DataFrame([self.data.iloc[0][FEATURE_COLUMNS].to_dict()])
        artifacts = run_model_selection(self.data.groupby(TARGET_COLUMN, group_keys=False).sample(n=180, random_state=42).reset_index(drop=True))

        explanation = build_patient_vs_cohort_explanation(artifacts.model, sample_input, self.data)

        self.assertTrue(explanation["display_rows"])
        self.assertTrue(explanation["technical_rows"])
        self.assertTrue(all("_missing" not in str(row["feature"]) for row in explanation["display_rows"]))
        self.assertTrue(any("_missing" in str(row["feature"]) for row in explanation["technical_rows"]))
        self.assertTrue(any(row["feature_label_key"] == "feature_smoking_exposure" for row in explanation["display_rows"]))
        self.assertTrue(all(row["relation_key"] in {"higher_than_cohort", "lower_than_cohort", "near_cohort"} for row in explanation["display_rows"]))


if __name__ == "__main__":
    unittest.main()
