from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "framingham.csv"
ARTIFACTS_DIR = APP_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
MODEL_METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
CV_METRICS_PATH = ARTIFACTS_DIR / "cv_metrics.json"

FEATURE_COLUMNS = [
    "male",
    "age",
    "education",
    "currentSmoker",
    "cigsPerDay",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]
TARGET_COLUMN = "TenYearCHD"
MISSING_INDICATOR_COLUMNS = ["glucose", "education", "BPMeds", "totChol", "cigsPerDay", "BMI"]
BINARY_COLUMNS = [
    "male",
    "currentSmoker",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
    TARGET_COLUMN,
]
CLINICAL_BOUNDS = {
    "age": (18, 110),
    "education": (1, 4),
    "cigsPerDay": (0, 100),
    "totChol": (80, 700),
    "sysBP": (70, 300),
    "diaBP": (30, 180),
    "BMI": (10, 80),
    "heartRate": (30, 220),
    "glucose": (40, 500),
}
RISK_BANDS = {
    "low": {"min": 0.0, "max": 0.10, "label_key": "risk_low"},
    "moderate": {"min": 0.10, "max": 0.20, "label_key": "risk_moderate"},
    "high": {"min": 0.20, "max": 1.01, "label_key": "risk_high"},
}
MODEL_PRIORITY = {"logistic_regression": 0, "random_forest": 1, "hist_gradient_boosting": 2}
MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
}
DISPLAY_EXPLANATION_FEATURES = [
    "age",
    "cigsPerDay",
    "currentSmoker",
    "sysBP",
    "totChol",
    "BMI",
    "glucose",
    "heartRate",
    "male",
    "prevalentHyp",
    "diabetes",
]
DISPLAY_FEATURE_LABEL_KEYS = {
    "age": "feature_age_baseline",
    "cigsPerDay": "feature_smoking_exposure",
    "currentSmoker": "feature_smoking_status",
    "sysBP": "feature_sys_bp_baseline",
    "totChol": "feature_tot_chol_baseline",
    "BMI": "feature_bmi_baseline",
    "glucose": "feature_glucose_baseline",
    "heartRate": "feature_heart_rate_baseline",
    "male": "feature_sex_baseline",
    "prevalentHyp": "feature_hypertension_baseline",
    "diabetes": "feature_diabetes_baseline",
}


@dataclass
class ValidationReport:
    rows_before: int
    rows_after: int
    dropped_rows: int
    issue_counts: dict[str, int]


@dataclass
class TrainingArtifacts:
    model: Any
    metadata: dict[str, Any]
    cv_metrics: dict[str, Any]


@dataclass(frozen=True)
class CandidateConfig:
    model_name: str
    params: dict[str, Any]


class SelectedMissingIndicatorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "SelectedMissingIndicatorTransformer":
        self.feature_names_in_ = list(self.columns)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.columns)
        return frame.isna().astype(float).to_numpy()

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        columns = input_features if input_features is not None else self.columns
        return np.asarray([f"{column}_missing" for column in columns], dtype=object)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}.")

    data = pd.read_csv(path, na_values=["NA"])
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError("Dataset is missing required columns: " + ", ".join(sorted(missing_columns)))

    return data


def validate_data(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    invalid_masks: dict[str, pd.Series] = {}

    invalid_masks["target_not_binary"] = ~df[TARGET_COLUMN].isin([0, 1])
    for column in BINARY_COLUMNS[:-1]:
        invalid_masks[f"{column}_not_binary"] = df[column].notna() & ~df[column].isin([0, 1])
    for column, (minimum, maximum) in CLINICAL_BOUNDS.items():
        invalid_masks[f"{column}_out_of_range"] = df[column].notna() & ~df[column].between(minimum, maximum)

    invalid_mask = pd.Series(False, index=df.index)
    issue_counts: dict[str, int] = {}
    for name, mask in invalid_masks.items():
        count = int(mask.sum())
        if count > 0:
            issue_counts[name] = count
            invalid_mask |= mask

    cleaned = df.loc[~invalid_mask].copy()
    cleaned.insert(0, "_source_row_id", cleaned.index.astype(int))
    cleaned = cleaned.reset_index(drop=True)
    report = ValidationReport(
        rows_before=len(df),
        rows_after=len(cleaned),
        dropped_rows=int(invalid_mask.sum()),
        issue_counts=issue_counts,
    )

    if cleaned.empty:
        raise ValueError("No rows remain after validation.")
    if cleaned[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Validated dataset must contain both target classes.")

    return cleaned, report


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("features", Pipeline(steps=numeric_steps), FEATURE_COLUMNS),
            ("missing_flags", SelectedMissingIndicatorTransformer(MISSING_INDICATOR_COLUMNS), MISSING_INDICATOR_COLUMNS),
        ],
        verbose_feature_names_out=False,
    )


def split_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    feature_frame = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN]
    class_counts = target.value_counts()
    if (class_counts < 2).any():
        raise ValueError("Need at least 2 rows in each target class for stratified train/test split.")

    return train_test_split(
        feature_frame,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )


def iter_candidate_configs() -> list[CandidateConfig]:
    return [
        CandidateConfig("logistic_regression", {"C": 0.5, "class_weight": None}),
        CandidateConfig("logistic_regression", {"C": 1.0, "class_weight": "balanced"}),
        CandidateConfig("logistic_regression", {"C": 2.0, "class_weight": "balanced"}),
        CandidateConfig("random_forest", {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 5, "class_weight": None}),
        CandidateConfig("random_forest", {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 3, "class_weight": "balanced"}),
        CandidateConfig("hist_gradient_boosting", {"learning_rate": 0.05, "max_depth": 3, "max_leaf_nodes": 15}),
        CandidateConfig("hist_gradient_boosting", {"learning_rate": 0.1, "max_depth": 4, "max_leaf_nodes": 31}),
    ]


def calibration_method_for_model(model_name: str) -> str:
    return "sigmoid" if model_name == "logistic_regression" else "isotonic"


def build_candidate_pipeline(candidate: CandidateConfig) -> Pipeline:
    if candidate.model_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=4000,
            solver="lbfgs",
            random_state=42,
            C=float(candidate.params["C"]),
            class_weight=candidate.params["class_weight"],
        )
        return Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numeric=True)),
                ("classifier", classifier),
            ]
        )

    if candidate.model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=int(candidate.params["n_estimators"]),
            max_depth=int(candidate.params["max_depth"]),
            min_samples_leaf=int(candidate.params["min_samples_leaf"]),
            class_weight=candidate.params["class_weight"],
            random_state=42,
            n_jobs=-1,
        )
        return Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numeric=False)),
                ("classifier", classifier),
            ]
        )

    classifier = HistGradientBoostingClassifier(
        learning_rate=float(candidate.params["learning_rate"]),
        max_depth=int(candidate.params["max_depth"]),
        max_leaf_nodes=int(candidate.params["max_leaf_nodes"]),
        random_state=42,
    )
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            ("classifier", classifier),
        ]
    )


def compute_threshold_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision),
        "ppv": float(precision),
        "npv": float(npv),
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
    }


def compute_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    metrics = compute_threshold_metrics(y_true, probabilities, threshold)
    metrics.update(
        {
            "pr_auc": float(average_precision_score(y_true, probabilities)),
            "roc_auc": float(roc_auc_score(y_true, probabilities)),
            "brier_score": float(brier_score_loss(y_true, probabilities)),
        }
    )
    return metrics


def select_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    candidate_thresholds = np.unique(np.round(probabilities, 4))
    thresholds = np.concatenate(([0.01], candidate_thresholds, [0.99]))

    best_threshold = 0.5
    best_score = -1.0
    for threshold in sorted(set(float(value) for value in thresholds)):
        score = compute_threshold_metrics(y_true, probabilities, threshold)["f2"]
        if score > best_score or (np.isclose(score, best_score) and threshold < best_threshold):
            best_score = score
            best_threshold = threshold

    return float(best_threshold)


def summarize_metric_list(metric_rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    keys = metric_rows[0].keys()
    for key in keys:
        values = [row[key] for row in metric_rows]
        summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return summary


def evaluate_candidate_cv(
    candidate: CandidateConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, Any]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    validation_probabilities = np.zeros(len(X_train), dtype=float)
    fold_records: list[dict[str, Any]] = []

    for fold_index, (subtrain_idx, validation_idx) in enumerate(splitter.split(X_train, y_train), start=1):
        estimator = build_candidate_pipeline(candidate)
        calibrated_model = CalibratedClassifierCV(
            estimator=estimator,
            method=calibration_method_for_model(candidate.model_name),
            cv=3,
            ensemble=False,
        )
        X_subtrain = X_train.iloc[subtrain_idx]
        y_subtrain = y_train.iloc[subtrain_idx]
        X_validation = X_train.iloc[validation_idx]
        y_validation = y_train.iloc[validation_idx]

        calibrated_model.fit(X_subtrain, y_subtrain)
        probabilities = calibrated_model.predict_proba(X_validation)[:, 1]
        validation_probabilities[validation_idx] = probabilities
        fold_records.append(
            {
                "fold_index": fold_index,
                "y_true": y_validation.to_numpy(),
                "probabilities": probabilities,
            }
        )

    threshold = select_threshold(y_train, validation_probabilities)
    fold_metrics = [
        compute_metrics(pd.Series(record["y_true"]), record["probabilities"], threshold)
        for record in fold_records
    ]
    summary = summarize_metric_list(fold_metrics)

    return {
        "candidate": candidate,
        "threshold": threshold,
        "fold_metrics": fold_metrics,
        "summary": summary,
    }


def candidate_sort_key(result: dict[str, Any]) -> tuple[float, float, float, int]:
    summary = result["summary"]
    return (
        -summary["pr_auc"]["mean"],
        -summary["roc_auc"]["mean"],
        -summary["balanced_accuracy"]["mean"],
        MODEL_PRIORITY[result["candidate"].model_name],
    )


def fit_final_model(candidate: CandidateConfig, X_train: pd.DataFrame, y_train: pd.Series) -> CalibratedClassifierCV:
    calibrated_model = CalibratedClassifierCV(
        estimator=build_candidate_pipeline(candidate),
        method=calibration_method_for_model(candidate.model_name),
        cv=5,
        ensemble=False,
    )
    calibrated_model.fit(X_train, y_train)
    return calibrated_model


def evaluate_on_test(model: Any, X_test: pd.DataFrame, y_test: pd.Series, threshold: float) -> dict[str, Any]:
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, probabilities, threshold)
    observed, predicted = calibration_curve(y_test, probabilities, n_bins=10, strategy="quantile")
    metrics["calibration_curve"] = {
        "predicted_probability": [float(value) for value in predicted],
        "observed_rate": [float(value) for value in observed],
    }
    return metrics


def get_calibrated_base_estimator(model: Any) -> Pipeline:
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        estimator = model.calibrated_classifiers_[0].estimator
        if isinstance(estimator, Pipeline):
            return estimator
    if hasattr(model, "estimator") and isinstance(model.estimator, Pipeline):
        return model.estimator
    raise ValueError("Unable to access fitted base estimator for explainability.")


def extract_global_explainability(model: Any, selected_model: str) -> dict[str, Any]:
    if selected_model == "logistic_regression":
        base_estimator = get_calibrated_base_estimator(model)
        preprocessor = base_estimator.named_steps["preprocessor"]
        classifier = base_estimator.named_steps["classifier"]
        feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
        coefficients = classifier.coef_[0]
        rows = [
            {
                "feature": feature,
                "coefficient": float(coefficient),
                "abs_coefficient": float(abs(coefficient)),
            }
            for feature, coefficient in zip(feature_names, coefficients, strict=True)
        ]
        rows.sort(key=lambda row: row["abs_coefficient"], reverse=True)
        return {
            "mode": "logistic_coefficients",
            "intercept": float(classifier.intercept_[0]),
            "rows": rows,
        }

    base_estimator = get_calibrated_base_estimator(model)
    preprocessor = base_estimator.named_steps["preprocessor"]
    classifier = base_estimator.named_steps["classifier"]
    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    importances = getattr(classifier, "feature_importances_", None)
    rows = []
    if importances is not None:
        rows = [
            {
                "feature": feature,
                "importance": float(importance),
            }
            for feature, importance in zip(feature_names, importances, strict=True)
        ]
        rows.sort(key=lambda row: row["importance"], reverse=True)
    return {"mode": "limited", "rows": rows}


def explain_logistic_prediction(model: Any, input_frame: pd.DataFrame) -> list[dict[str, float | str]]:
    base_estimator = get_calibrated_base_estimator(model)
    preprocessor = base_estimator.named_steps["preprocessor"]
    classifier = base_estimator.named_steps["classifier"]
    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    transformed = preprocessor.transform(input_frame)
    contributions = transformed[0] * classifier.coef_[0]
    rows = [
        {
            "feature": feature,
            "contribution": float(contribution),
            "abs_contribution": float(abs(contribution)),
        }
        for feature, contribution in zip(feature_names, contributions, strict=True)
    ]
    rows.sort(key=lambda row: row["abs_contribution"], reverse=True)
    return rows


def build_patient_vs_cohort_explanation(
    model: Any,
    input_frame: pd.DataFrame,
    reference_data: pd.DataFrame,
) -> dict[str, list[dict[str, float | str]]]:
    technical_rows = explain_logistic_prediction(model, input_frame)
    technical_map = {
        str(row["feature"]): row
        for row in technical_rows
        if str(row["feature"]) in FEATURE_COLUMNS
    }
    validated_reference, _ = validate_data(reference_data)
    feature_reference = validated_reference[FEATURE_COLUMNS]
    medians = feature_reference.median(numeric_only=True)
    ranges = (feature_reference.max(numeric_only=True) - feature_reference.min(numeric_only=True)).replace(0, 1.0)

    display_rows: list[dict[str, float | str]] = []
    patient_row = input_frame.iloc[0]
    for feature in DISPLAY_EXPLANATION_FEATURES:
        contribution = float(technical_map[feature]["contribution"])
        abs_contribution = float(abs(contribution))
        patient_value = float(patient_row[feature])
        cohort_value = float(medians[feature])
        feature_range = float(ranges[feature])

        if feature in {"male", "currentSmoker", "prevalentHyp", "diabetes"}:
            if patient_value == cohort_value:
                relation_key = "near_cohort"
            elif patient_value > cohort_value:
                relation_key = "higher_than_cohort"
            else:
                relation_key = "lower_than_cohort"
        else:
            tolerance = max(feature_range * 0.05, 0.01)
            if abs(patient_value - cohort_value) <= tolerance:
                relation_key = "near_cohort"
            elif patient_value > cohort_value:
                relation_key = "higher_than_cohort"
            else:
                relation_key = "lower_than_cohort"

        if contribution > 0.005:
            score_direction_key = "raises_score"
        elif contribution < -0.005:
            score_direction_key = "lowers_score"
        else:
            score_direction_key = "neutral_score"

        display_rows.append(
            {
                "feature": feature,
                "feature_label_key": DISPLAY_FEATURE_LABEL_KEYS[feature],
                "patient_value": patient_value,
                "cohort_value": cohort_value,
                "relation_key": relation_key,
                "score_direction_key": score_direction_key,
                "contribution": contribution,
                "abs_contribution": abs_contribution,
            }
        )

    display_rows.sort(key=lambda row: float(row["abs_contribution"]), reverse=True)
    return {
        "technical_rows": technical_rows,
        "display_rows": display_rows,
    }


def risk_band_label_key(probability: float) -> str:
    if probability < RISK_BANDS["moderate"]["min"]:
        return RISK_BANDS["low"]["label_key"]
    if probability < RISK_BANDS["high"]["min"]:
        return RISK_BANDS["moderate"]["label_key"]
    return RISK_BANDS["high"]["label_key"]


def similarity_group_key(similarity: float) -> str:
    if similarity >= 90:
        return "very_high"
    if similarity >= 80:
        return "high"
    if similarity >= 70:
        return "moderate"
    return "broad"


def find_nearest_patients(
    input_frame: pd.DataFrame,
    data: pd.DataFrame,
    top_k: int = 50,
) -> dict[str, Any]:
    validated_data, _ = validate_data(data)
    reference = validated_data[["_source_row_id", *FEATURE_COLUMNS, TARGET_COLUMN]].copy()
    feature_reference = reference[FEATURE_COLUMNS].copy()
    medians = feature_reference.median(numeric_only=True)
    feature_reference = feature_reference.fillna(medians)

    feature_ranges = feature_reference.max() - feature_reference.min()
    feature_ranges = feature_ranges.replace(0, 1.0)

    input_features = input_frame[FEATURE_COLUMNS].copy().fillna(medians)
    diffs = feature_reference.subtract(input_features.iloc[0], axis=1).abs().div(feature_ranges, axis=1).clip(0, 1)
    similarity = (1.0 - diffs.mean(axis=1)).clip(0, 1) * 100

    neighbors = reference.copy()
    neighbors["similarity_pct"] = similarity.round(2)
    neighbors["similarity_group"] = neighbors["similarity_pct"].apply(similarity_group_key)
    neighbors = neighbors.sort_values(["similarity_pct", "_source_row_id"], ascending=[False, True]).head(top_k).reset_index(drop=True)

    group_summary = []
    for group_key, group_frame in neighbors.groupby("similarity_group", sort=False):
        group_summary.append(
            {
                "group_key": group_key,
                "count": int(len(group_frame)),
                "share_pct": float(round((len(group_frame) / len(neighbors)) * 100, 1)),
                "avg_similarity_pct": float(round(group_frame["similarity_pct"].mean(), 2)),
            }
        )

    closest = neighbors.iloc[0].to_dict()
    return {
        "neighbors": neighbors,
        "group_summary": group_summary,
        "closest": closest,
    }


def flatten_summary(summary: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        f"{metric}_{stat}": value
        for metric, stats in summary.items()
        for stat, value in stats.items()
    }


def to_json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_json_ready(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [to_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [to_json_ready(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if dataclass_is_instance(value):
        return to_json_ready(asdict(value))
    return value


def dataclass_is_instance(value: Any) -> bool:
    return hasattr(value, "__dataclass_fields__")


def dataset_version(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:12]


def run_model_selection(data: pd.DataFrame, data_path: Path = DATA_PATH) -> TrainingArtifacts:
    validated_data, validation_report = validate_data(data)
    X_train, X_test, y_train, y_test = split_data(validated_data)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    evaluated_candidates = [evaluate_candidate_cv(candidate, X_train, y_train) for candidate in iter_candidate_configs()]

    best_by_model: dict[str, dict[str, Any]] = {}
    for result in evaluated_candidates:
        model_name = result["candidate"].model_name
        if model_name not in best_by_model or candidate_sort_key(result) < candidate_sort_key(best_by_model[model_name]):
            best_by_model[model_name] = result

    selected_result = sorted(best_by_model.values(), key=candidate_sort_key)[0]
    selected_candidate: CandidateConfig = selected_result["candidate"]
    final_model = fit_final_model(selected_candidate, X_train, y_train)
    test_metrics = evaluate_on_test(final_model, X_test, y_test, selected_result["threshold"])
    explainability = extract_global_explainability(final_model, selected_candidate.model_name)

    comparison_rows = []
    for result in sorted(best_by_model.values(), key=candidate_sort_key):
        summary = result["summary"]
        comparison_rows.append(
            {
                "model_key": result["candidate"].model_name,
                "model_name": MODEL_DISPLAY_NAMES[result["candidate"].model_name],
                "threshold": float(result["threshold"]),
                "params": result["candidate"].params,
                **flatten_summary(summary),
            }
        )

    metadata = {
        "data_version": dataset_version(data_path),
        "train_timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "selected_model": selected_candidate.model_name,
        "selected_model_display_name": MODEL_DISPLAY_NAMES[selected_candidate.model_name],
        "threshold": float(selected_result["threshold"]),
        "risk_bands": RISK_BANDS,
        "cv_summary": comparison_rows,
        "test_metrics": test_metrics,
        "calibration_method": calibration_method_for_model(selected_candidate.model_name),
        "validation_report": asdict(validation_report),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "explainability": explainability,
    }
    cv_metrics = {
        "selected_model": selected_candidate.model_name,
        "selected_params": selected_candidate.params,
        "candidates": [
            {
                "model_name": result["candidate"].model_name,
                "display_name": MODEL_DISPLAY_NAMES[result["candidate"].model_name],
                "params": result["candidate"].params,
                "threshold": float(result["threshold"]),
                "summary": result["summary"],
                "fold_metrics": result["fold_metrics"],
            }
            for result in evaluated_candidates
        ],
    }

    return TrainingArtifacts(model=final_model, metadata=metadata, cv_metrics=cv_metrics)


def save_training_artifacts(artifacts: TrainingArtifacts, artifacts_dir: Path = ARTIFACTS_DIR) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.model, artifacts_dir / "model.joblib")
    (artifacts_dir / "model_metadata.json").write_text(
        json.dumps(to_json_ready(artifacts.metadata), indent=2),
        encoding="utf-8",
    )
    (artifacts_dir / "cv_metrics.json").write_text(
        json.dumps(to_json_ready(artifacts.cv_metrics), indent=2),
        encoding="utf-8",
    )


def load_artifacts(artifacts_dir: Path = ARTIFACTS_DIR) -> TrainingArtifacts:
    model_path = artifacts_dir / "model.joblib"
    metadata_path = artifacts_dir / "model_metadata.json"
    cv_metrics_path = artifacts_dir / "cv_metrics.json"

    missing = [path.name for path in [model_path, metadata_path, cv_metrics_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Model artifacts not found: "
            + ", ".join(missing)
            + ". Run `python train_model.py` to generate offline artifacts."
        )

    return TrainingArtifacts(
        model=joblib.load(model_path),
        metadata=json.loads(metadata_path.read_text(encoding="utf-8")),
        cv_metrics=json.loads(cv_metrics_path.read_text(encoding="utf-8")),
    )
