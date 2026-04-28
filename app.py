from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from modeling import (
    ARTIFACTS_DIR,
    DATA_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_patient_vs_cohort_explanation,
    validate_data,
    find_nearest_patients,
    load_artifacts,
    load_data,
    explain_logistic_prediction,
    risk_band_label_key,
)


TRANSLATIONS = {
    "en": {
        "page_title": "Clinical Risk Explorer",
        "caption": "A cleaner, calibrated demo for 10-year coronary heart disease risk exploration.",
        "language": "Language",
        "sidebar_header": "Patient Inputs",
        "sex": "Sex",
        "female": "Female",
        "male": "Male",
        "age": "Age",
        "education": "Education level",
        "education_help": "Framingham coding: 1 to 4",
        "current_smoker": "Current smoker",
        "no": "No",
        "yes": "Yes",
        "cigs_per_day": "Cigarettes per day",
        "bp_meds": "On blood pressure medication",
        "stroke": "History of stroke",
        "prevalent_hyp": "Prevalent hypertension",
        "diabetes": "Diabetes",
        "tot_chol": "Total cholesterol",
        "sys_bp": "Systolic BP",
        "dia_bp": "Diastolic BP",
        "bmi": "BMI",
        "heart_rate": "Heart rate",
        "glucose": "Glucose",
        "predicted_risk": "Calibrated 10-year CHD risk",
        "risk_band": "Risk band",
        "decision_threshold": "Decision threshold",
        "risk_low": "Low",
        "risk_moderate": "Moderate",
        "risk_high": "High",
        "model_not_ready": "Model artifacts could not be loaded",
        "runtime_note": "This app loads a pre-trained calibrated model artifact and does not retrain at runtime.",
        "metadata_title": "Model snapshot",
        "selected_model": "Selected model",
        "train_rows": "Train rows",
        "test_rows": "Test rows",
        "trained_at": "Trained at",
        "data_version": "Data version",
        "performance_title": "Final test performance",
        "cv_title": "Cross-validation summary",
        "comparison_title": "Best candidate per model family",
        "explanation_title": "Explainability",
        "local_explanation": "Technical local contributions",
        "global_explanation": "Technical global coefficients",
        "limited_explanation": "Limited explainability",
        "limited_explanation_text": "The selected model is tree-based, so this panel shows global feature importances instead of patient-level signed contributions.",
        "dataset_preview": "Dataset preview",
        "input_summary": "Input summary",
        "rows_loaded": "Rows loaded",
        "validation_removed": "Rows removed during validation",
        "test_metric_note": "Final metrics below come from the untouched 20% test split. Risk bands use calibrated demo thresholds, not a clinical protocol.",
        "cv_metric_note": "Cross-validation metrics are reported as mean ± std on the 80% training split only.",
        "pr_auc": "PR-AUC",
        "roc_auc": "ROC-AUC",
        "recall": "Recall / Sensitivity",
        "specificity": "Specificity",
        "ppv": "PPV",
        "npv": "NPV",
        "balanced_accuracy": "Balanced accuracy",
        "brier_score": "Brier score",
        "accuracy": "Accuracy",
        "f2": "F2",
        "threshold": "Threshold",
        "model_name": "Model",
        "params": "Parameters",
        "calibration_curve": "Calibration curve",
        "predicted_probability": "Predicted probability",
        "observed_rate": "Observed event rate",
        "feature": "Feature",
        "coefficient": "Coefficient",
        "importance": "Importance",
        "contribution": "Contribution",
        "nearest_title": "Most similar 50 patients",
        "nearest_groups": "Similarity groups",
        "closest_patient": "Closest single patient",
        "similarity_pct": "Similarity %",
        "group_label": "Group",
        "group_very_high": "Very high similarity",
        "group_high": "High similarity",
        "group_moderate": "Moderate similarity",
        "group_broad": "Broader match",
        "group_share": "Share of top 50",
        "avg_similarity": "Average similarity",
        "actual_outcome": "Observed CHD outcome",
        "patient_id": "Reference row",
        "outcome_positive": "Positive (CHD)",
        "outcome_negative": "Negative (No CHD)",
        "top50_note": "These 50 rows come from the validated reference cohort and are ranked by normalized feature similarity.",
        "summary_tab": "Summary",
        "neighbors_tab": "Similar patients",
        "insights_tab": "Model insights",
        "risk_overview": "Risk overview",
        "screening_signal": "Screening signal",
        "closest_match": "Closest match",
        "final_quality": "Final model quality",
        "high_attention": "Closer follow-up suggested",
        "medium_attention": "Borderline profile",
        "low_attention": "Lower-risk pattern",
        "advanced_details": "Advanced details",
        "compact_note": "This version keeps the dashboard focused on decision-relevant outputs and hides lower-signal diagnostics behind tabs.",
        "model_card_note": "Offline-trained, calibrated, and evaluated on a locked test set.",
        "key_metrics": "Key metrics",
        "secondary_metrics": "Secondary metrics",
        "top_matches": "Top matches",
        "top_matches_note": "The table is sorted by normalized feature similarity, highest first.",
        "outcome_rate_top50": "Positive outcome in top 50",
        "risk_distribution_chart": "Risk distribution",
        "risk_percentile": "Risk percentile",
        "risk_distribution_note": "Your risk is placed inside the model-estimated cohort distribution.",
        "similar_patients_chart": "Top 10 similar patients",
        "contribution_chart": "Feature contribution chart",
        "relationship_graph": "Similarity graph",
        "selected_bin": "Selected patient bin",
        "cohort_count": "Cohort count",
        "top10_similarity_note": "Bars show the 10 closest matches ranked by normalized feature similarity.",
        "contribution_note": "Positive values push risk up; negative values pull risk down.",
        "graph_note": "Drag, pan, zoom, and click nodes to explore the similarity neighborhood around the selected patient.",
        "selected_patient": "Selected patient",
        "reference_patient": "Reference patient",
        "graph_fallback_note": "If the graph area stays empty, reload the page once. The similarity table below remains the source of truth.",
        "graph_legend_selected": "Selected patient",
        "graph_legend_highlighted": "Top 50 highlighted",
        "graph_legend_positive": "Positive outcome",
        "graph_legend_negative": "Negative outcome",
        "graph_legend_background": "Other cohort nodes",
        "graph_legend_strong": "Strong link",
        "graph_legend_weak": "Weak link",
        "graph_node_id": "ID",
        "contribution_chart": "Patient profile vs cohort baseline",
        "contribution_note": "Negative bars do not mean a factor is protective in general. They mean this patient's value lowers the model score relative to the cohort baseline.",
        "patient_value": "Patient value",
        "cohort_baseline": "Cohort baseline",
        "relative_position": "Relative position",
        "score_effect": "Model score effect",
        "higher_than_cohort": "Higher than cohort",
        "lower_than_cohort": "Lower than cohort",
        "near_cohort": "Close to cohort",
        "raises_score": "Raises score",
        "lowers_score": "Lowers score",
        "neutral_score": "Little effect",
        "risk_band_note": "Risk band is a demo model output and should not be interpreted as a standalone clinical decision.",
        "similarity_note": "Similarity describes profile resemblance, not outcome causality.",
        "feature_age_baseline": "Age vs cohort",
        "feature_smoking_exposure": "Smoking exposure vs cohort",
        "feature_smoking_status": "Smoking status vs cohort",
        "feature_sys_bp_baseline": "Systolic BP vs cohort",
        "feature_tot_chol_baseline": "Total cholesterol vs cohort",
        "feature_bmi_baseline": "BMI vs cohort",
        "feature_glucose_baseline": "Glucose vs cohort",
        "feature_heart_rate_baseline": "Heart rate vs cohort",
        "feature_sex_baseline": "Sex profile vs cohort",
        "feature_hypertension_baseline": "Hypertension history vs cohort",
        "feature_diabetes_baseline": "Diabetes status vs cohort",
    },
    "tr": {
        "page_title": "Klinik Risk Gezgini",
        "caption": "10 yıllık koroner kalp hastalığı riskini daha sade ve kalibre edilmiş şekilde gösteren bir demo.",
        "language": "Dil",
        "sidebar_header": "Hasta Girdileri",
        "sex": "Cinsiyet",
        "female": "Kadın",
        "male": "Erkek",
        "age": "Yaş",
        "education": "Eğitim seviyesi",
        "education_help": "Framingham kodlaması: 1 ile 4 arası",
        "current_smoker": "Aktif sigara kullanımı",
        "no": "Hayır",
        "yes": "Evet",
        "cigs_per_day": "Günde içilen sigara",
        "bp_meds": "Tansiyon ilacı kullanımı",
        "stroke": "İnme öyküsü",
        "prevalent_hyp": "Hipertansiyon varlığı",
        "diabetes": "Diyabet",
        "tot_chol": "Toplam kolesterol",
        "sys_bp": "Sistolik tansiyon",
        "dia_bp": "Diyastolik tansiyon",
        "bmi": "Vücut kitle indeksi",
        "heart_rate": "Nabız",
        "glucose": "Glukoz",
        "predicted_risk": "Kalibre edilmiş 10 yıllık KKH riski",
        "risk_band": "Risk düzeyi",
        "decision_threshold": "Karar eşiği",
        "risk_low": "Düşük",
        "risk_moderate": "Orta",
        "risk_high": "Yüksek",
        "model_not_ready": "Model artefact'ları yüklenemedi",
        "runtime_note": "Bu uygulama önceden eğitilmiş kalibre edilmiş model artefact'ını yükler; runtime sırasında yeniden eğitim yapmaz.",
        "metadata_title": "Model özeti",
        "selected_model": "Seçilen model",
        "train_rows": "Eğitim satırı",
        "test_rows": "Test satırı",
        "trained_at": "Eğitim zamanı",
        "data_version": "Veri sürümü",
        "performance_title": "Final test performansı",
        "cv_title": "Çapraz doğrulama özeti",
        "comparison_title": "Model ailesi başına en iyi aday",
        "explanation_title": "Açıklanabilirlik",
        "local_explanation": "Teknik yerel katkılar",
        "global_explanation": "Teknik global katsayılar",
        "limited_explanation": "Sınırlı açıklanabilirlik",
        "limited_explanation_text": "Seçilen model ağaç tabanlı olduğu için bu panel hasta bazlı işaretli katkılar yerine global feature importance gösterir.",
        "dataset_preview": "Veri kümesi önizlemesi",
        "input_summary": "Girdi özeti",
        "rows_loaded": "Yüklenen satır",
        "validation_removed": "Doğrulamada çıkarılan satır",
        "test_metric_note": "Aşağıdaki final metrikler dokunulmamış %20 test split'inden gelir. Risk bantları klinik protokol değil, kalibre edilmiş demo eşikleridir.",
        "cv_metric_note": "Çapraz doğrulama metrikleri yalnızca eğitim için ayrılan %80 bölümde mean ± std olarak raporlanır.",
        "pr_auc": "PR-AUC",
        "roc_auc": "ROC-AUC",
        "recall": "Recall / Duyarlılık",
        "specificity": "Spesifisite",
        "ppv": "PPV",
        "npv": "NPV",
        "balanced_accuracy": "Dengeli doğruluk",
        "brier_score": "Brier skoru",
        "accuracy": "Accuracy",
        "f2": "F2",
        "threshold": "Eşik",
        "model_name": "Model",
        "params": "Parametreler",
        "calibration_curve": "Kalibrasyon eğrisi",
        "predicted_probability": "Tahmin edilen olasılık",
        "observed_rate": "Gözlenen olay oranı",
        "feature": "Özellik",
        "coefficient": "Katsayı",
        "importance": "Önem",
        "contribution": "Katkı",
        "nearest_title": "En benzer 50 kişi",
        "nearest_groups": "Benzerlik grupları",
        "closest_patient": "En yakın tekil kişi",
        "similarity_pct": "Benzerlik %",
        "group_label": "Grup",
        "group_very_high": "Çok yüksek benzerlik",
        "group_high": "Yüksek benzerlik",
        "group_moderate": "Orta benzerlik",
        "group_broad": "Daha geniş eşleşme",
        "group_share": "İlk 50 içindeki pay",
        "avg_similarity": "Ortalama benzerlik",
        "actual_outcome": "Gözlenen KKH sonucu",
        "patient_id": "Referans satır",
        "outcome_positive": "Pozitif (KKH)",
        "outcome_negative": "Negatif (KKH yok)",
        "top50_note": "Bu 50 kayıt doğrulanmış referans kohorttan gelir ve normalize özellik benzerliğine göre sıralanır.",
        "summary_tab": "Özet",
        "neighbors_tab": "Benzer kişiler",
        "insights_tab": "Model içgörüleri",
        "risk_overview": "Risk özeti",
        "screening_signal": "Tarama sinyali",
        "closest_match": "En yakın eşleşme",
        "final_quality": "Final model kalitesi",
        "high_attention": "Daha yakın takip önerilir",
        "medium_attention": "Sınırda profil",
        "low_attention": "Daha düşük risk paterni",
        "advanced_details": "İleri detaylar",
        "compact_note": "Bu sürüm dashboard'u karar açısından önemli çıktılara odaklar ve düşük sinyalli detayları sekmelerin arkasına taşır.",
        "model_card_note": "Offline eğitilmiş, kalibre edilmiş ve kilitli test setinde değerlendirilmiştir.",
        "key_metrics": "Ana metrikler",
        "secondary_metrics": "İkincil metrikler",
        "top_matches": "En yakın eşleşmeler",
        "top_matches_note": "Tablo normalize özellik benzerliğine göre en yüksekten düşüğe sıralanır.",
        "outcome_rate_top50": "İlk 50 içindeki pozitif sonuç oranı",
        "risk_distribution_chart": "Risk dağılımı",
        "risk_percentile": "Risk persentili",
        "risk_distribution_note": "Riskiniz modelin tahmin ettiği kohort dağılımı içindeki konumuyla gösterilir.",
        "similar_patients_chart": "En yakın 10 benzer kişi",
        "contribution_chart": "Özellik katkı grafiği",
        "relationship_graph": "Benzerlik grafiği",
        "selected_bin": "Seçili kişi bandı",
        "cohort_count": "Kohort sayısı",
        "top10_similarity_note": "Barlar normalize özellik benzerliğine göre en yakın 10 eşleşmeyi gösterir.",
        "contribution_note": "Pozitif değerler riski artırır, negatif değerler riski aşağı çeker.",
        "graph_note": "Seçili kişinin benzerlik komşuluğunu görmek için düğümleri sürükleyin, yakınlaştırın, kaydırın ve tıklayın.",
        "selected_patient": "Seçili kişi",
        "reference_patient": "Referans kişi",
        "graph_fallback_note": "Graph alanı boş kalırsa sayfayı bir kez yenileyin. Aşağıdaki benzerlik tablosu esas referans olmaya devam eder.",
        "graph_legend_selected": "Seçili kişi",
        "graph_legend_highlighted": "İlk 50 vurgulu",
        "graph_legend_positive": "Pozitif sonuç",
        "graph_legend_negative": "Negatif sonuç",
        "graph_legend_background": "Diğer kohort düğümleri",
        "graph_legend_strong": "Güçlü bağlantı",
        "graph_legend_weak": "Zayıf bağlantı",
        "graph_node_id": "ID",
        "contribution_chart": "Bu kişinin profili kohort baseline'ına göre",
        "contribution_note": "Negatif barlar bir faktörün genel olarak koruyucu olduğu anlamına gelmez. Sadece bu kişinin değerinin kohort baseline’ına göre model skorunu aşağı çektiğini gösterir.",
        "patient_value": "Kişi değeri",
        "cohort_baseline": "Kohort baseline",
        "relative_position": "Göreli konum",
        "score_effect": "Model skoru etkisi",
        "higher_than_cohort": "Kohorttan yüksek",
        "lower_than_cohort": "Kohorttan düşük",
        "near_cohort": "Kohorta yakın",
        "raises_score": "Skoru yükseltir",
        "lowers_score": "Skoru düşürür",
        "neutral_score": "Etkisi sınırlı",
        "risk_band_note": "Risk düzeyi demo modele ait bir çıktıdır; tek başına klinik karar olarak yorumlanmamalıdır.",
        "similarity_note": "Benzerlik, sonuç nedenselliğini değil profil benzerliğini anlatır.",
        "feature_age_baseline": "Yaş vs kohort",
        "feature_smoking_exposure": "Sigara maruziyeti vs kohort",
        "feature_smoking_status": "Sigara durumu vs kohort",
        "feature_sys_bp_baseline": "Sistolik tansiyon vs kohort",
        "feature_tot_chol_baseline": "Toplam kolesterol vs kohort",
        "feature_bmi_baseline": "BMI vs kohort",
        "feature_glucose_baseline": "Glukoz vs kohort",
        "feature_heart_rate_baseline": "Nabız vs kohort",
        "feature_sex_baseline": "Cinsiyet profili vs kohort",
        "feature_hypertension_baseline": "Hipertansiyon öyküsü vs kohort",
        "feature_diabetes_baseline": "Diyabet durumu vs kohort",
    },
}


def t(texts: dict[str, str], key: str) -> str:
    return texts[key]


def inject_styles() -> None:
    return None


def section_intro(title: str, copy: str) -> None:
    st.markdown(f"### {title}")
    st.caption(copy)


def explanation_feature_label(texts: dict[str, str], label_key: str) -> str:
    return t(texts, label_key)


def build_risk_distribution_frame(reference_scores: pd.Series, selected_probability: float) -> pd.DataFrame:
    bins = [i / 20 for i in range(21)]
    categories = pd.cut(reference_scores, bins=bins, include_lowest=True)
    counts = categories.value_counts().sort_index()
    selected_bucket = pd.cut(
        pd.Series([selected_probability]),
        bins=bins,
        include_lowest=True,
    ).iloc[0]
    frame = pd.DataFrame(
        {
            "bucket": [str(interval) for interval in counts.index],
            "cohort_count": counts.values,
            "selected_bin": [counts.iloc[idx] if interval == selected_bucket else 0 for idx, interval in enumerate(counts.index)],
        }
    )
    return frame.set_index("bucket")


def build_similarity_bar_frame(neighbors: pd.DataFrame) -> pd.DataFrame:
    top10 = neighbors.head(10).copy()
    labels = top10["_source_row_id"].astype(int).astype(str)
    return pd.DataFrame({"similarity_pct": top10["similarity_pct"].values}, index=labels)


def build_contribution_frame(local_rows: list[dict[str, float | str]]) -> pd.DataFrame:
    frame = pd.DataFrame(local_rows[:10])[["feature", "contribution"]].copy()
    return frame.set_index("feature")


def build_display_explanation_chart(
    display_rows: list[dict[str, float | str]],
    texts: dict[str, str],
) -> pd.DataFrame:
    top_rows = display_rows[:8]
    labels = [explanation_feature_label(texts, str(row["feature_label_key"])) for row in top_rows]
    values = [float(row["contribution"]) for row in top_rows]
    return pd.DataFrame({"contribution": values}, index=labels)


def pairwise_similarity(left: pd.Series, right: pd.Series, medians: pd.Series, ranges: pd.Series) -> float:
    left_filled = left.fillna(medians)
    right_filled = right.fillna(medians)
    normalized = (left_filled - right_filled).abs().div(ranges).clip(0, 1)
    return float((1.0 - normalized.mean()) * 100)


def build_similarity_graph_data(
    full_data: pd.DataFrame,
    input_frame: pd.DataFrame,
    neighbors: pd.DataFrame,
    texts: dict[str, str],
) -> dict[str, object]:
    validated_reference, _ = validate_data(full_data)
    feature_frame = validated_reference[FEATURE_COLUMNS].copy()
    medians = feature_frame.median(numeric_only=True)
    ranges = (feature_frame.max(numeric_only=True) - feature_frame.min(numeric_only=True)).replace(0, 1.0)

    scaler = StandardScaler()
    standardized = scaler.fit_transform(feature_frame.fillna(medians))
    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(standardized)
    selected_coordinates = pca.transform(scaler.transform(input_frame[FEATURE_COLUMNS].fillna(medians)))[0]

    all_x = np.append(coordinates[:, 0], selected_coordinates[0])
    all_y = np.append(coordinates[:, 1], selected_coordinates[1])
    x_rank = pd.Series(all_x).rank(method="average", pct=True).to_numpy()
    y_rank = pd.Series(all_y).rank(method="average", pct=True).to_numpy()

    def spread_coordinate(rank_value: float, extent: float) -> float:
        centered = (rank_value - 0.5) * 2.0
        return float(centered * extent)

    spread_x = np.array([spread_coordinate(value, 1650.0) for value in x_rank])
    spread_y = np.array([spread_coordinate(value, 1080.0) for value in y_rank])
    coordinates = np.column_stack([spread_x[:-1], spread_y[:-1]])
    selected_coordinates = np.array([spread_x[-1], spread_y[-1]])

    highlighted_ids = {int(row["_source_row_id"]) for _, row in neighbors.head(50).iterrows()}
    top10_ids = [int(row["_source_row_id"]) for _, row in neighbors.head(10).iterrows()]
    highlight_lookup = {int(row["_source_row_id"]): row for _, row in neighbors.head(50).iterrows()}

    nodes: list[dict[str, object]] = [
        {
            "id": "selected",
            "label": "YOU",
            "originalLabel": "YOU",
            "alwaysLabel": True,
            "size": 34,
            "x": float(selected_coordinates[0]),
            "y": float(selected_coordinates[1]),
            "color": {"background": "#60a5fa", "border": "#bfdbfe", "highlight": {"background": "#93c5fd", "border": "#ffffff"}},
            "font": {"color": "#ffffff", "size": 16, "face": "Arial"},
            "title": f"<b>{t(texts, 'selected_patient')}</b>",
            "group": "selected",
            "isHighlighted": True,
        }
    ]
    edges: list[dict[str, object]] = []

    for idx, row in validated_reference.reset_index(drop=True).iterrows():
        row_id = int(row["_source_row_id"])
        node_id = f"patient-{row_id}"
        outcome_positive = int(row[TARGET_COLUMN]) == 1
        is_highlighted = row_id in highlighted_ids
        similarity = float(highlight_lookup[row_id]["similarity_pct"]) if is_highlighted else None
        smoker = t(texts, "yes") if int(row["currentSmoker"]) == 1 else t(texts, "no")
        sex = t(texts, "male") if int(row["male"]) == 1 else t(texts, "female")
        nodes.append(
            {
                "id": node_id,
                "label": str(row_id) if row_id in top10_ids else "",
                "originalLabel": str(row_id),
                "alwaysLabel": row_id in top10_ids,
                "size": round(11 + ((similarity - 60) / 40) * 10, 1) if is_highlighted else 4.2,
                "x": float(coordinates[idx][0] + np.sin(row_id) * 18),
                "y": float(coordinates[idx][1] + np.cos(row_id) * 18),
                "color": (
                    {
                        "background": "#0f766e" if not outcome_positive else "#ef4444",
                        "border": "#99f6e4" if not outcome_positive else "#fecaca",
                        "highlight": {"background": "#14b8a6" if not outcome_positive else "#f87171", "border": "#ffffff"},
                    }
                    if is_highlighted
                    else {
                        "background": "#64748b",
                        "border": "#94a3b8",
                        "highlight": {"background": "#94a3b8", "border": "#e2e8f0"},
                    }
                ),
                "font": {"color": "#f8fafc", "size": 12, "face": "Arial"},
                "title": (
                    f"<b>{t(texts, 'graph_node_id')}:</b> {row_id}<br>"
                    + (f"<b>{t(texts, 'similarity_pct')}:</b> {similarity:.1f}%<br>" if is_highlighted else "")
                    + f"<b>{t(texts, 'actual_outcome')}:</b> {outcome_label(texts, int(row[TARGET_COLUMN]))}<br>"
                    + f"<b>{t(texts, 'age')}:</b> {float(row['age']):.0f}<br>"
                    + f"<b>{t(texts, 'sex')}:</b> {sex}<br>"
                    + f"<b>{t(texts, 'current_smoker')}:</b> {smoker}<br>"
                    + f"<b>{t(texts, 'sys_bp')}:</b> {float(row['sysBP']):.0f}<br>"
                    + f"<b>{t(texts, 'tot_chol')}:</b> {float(row['totChol']):.0f}<br>"
                    + f"<b>{t(texts, 'bmi')}:</b> {float(row['BMI']):.1f}<br>"
                    + f"<b>{t(texts, 'glucose')}:</b> {float(row['glucose']):.0f}"
                ),
                "group": "highlight-positive" if (is_highlighted and outcome_positive) else "highlight-negative" if is_highlighted else "background",
                "isHighlighted": is_highlighted,
            }
        )

    backbone_model = NearestNeighbors(n_neighbors=3, metric="euclidean")
    backbone_model.fit(standardized)
    _, backbone_indices = backbone_model.kneighbors(standardized)
    background_edge_pairs: set[tuple[str, str]] = set()
    for idx, neighbor_indices in enumerate(backbone_indices):
        source_row_id = int(validated_reference.iloc[idx]["_source_row_id"])
        for neighbor_idx in neighbor_indices[1:]:
            target_row_id = int(validated_reference.iloc[int(neighbor_idx)]["_source_row_id"])
            background_edge_pairs.add(tuple(sorted((f"patient-{source_row_id}", f"patient-{target_row_id}"))))

    highlight_top = neighbors.head(50).copy().reset_index(drop=True)
    highlight_feature_frame = highlight_top[FEATURE_COLUMNS].copy()
    highlight_medians = highlight_feature_frame.median(numeric_only=True)
    highlight_ranges = (highlight_feature_frame.max(numeric_only=True) - highlight_feature_frame.min(numeric_only=True)).replace(0, 1.0)
    score_lookup: dict[frozenset[str], float] = {}
    internal_edge_pairs: set[tuple[str, str]] = set()
    for left_idx in range(len(highlight_top)):
        for right_idx in range(left_idx + 1, len(highlight_top)):
            left = highlight_top.iloc[left_idx]
            right = highlight_top.iloc[right_idx]
            score = pairwise_similarity(left[FEATURE_COLUMNS], right[FEATURE_COLUMNS], highlight_medians, highlight_ranges)
            left_id = f"patient-{int(left['_source_row_id'])}"
            right_id = f"patient-{int(right['_source_row_id'])}"
            score_lookup[frozenset((left_id, right_id))] = score
            if score >= 88.0:
                internal_edge_pairs.add(tuple(sorted((left_id, right_id))))

    if len(internal_edge_pairs) < 18:
        top_links_by_node: dict[str, list[tuple[float, str]]] = {}
        for pair, score in score_lookup.items():
            left_id, right_id = tuple(pair)
            top_links_by_node.setdefault(left_id, []).append((score, right_id))
            top_links_by_node.setdefault(right_id, []).append((score, left_id))
        for node_id, candidates in top_links_by_node.items():
            for score, target_id in sorted(candidates, key=lambda item: item[0], reverse=True)[:2]:
                internal_edge_pairs.add(tuple(sorted((node_id, target_id))))

    for left_id, right_id in sorted(background_edge_pairs):
        edges.append(
            {
                "from": left_id,
                "to": right_id,
                "value": 0.22,
                "color": {"color": "rgba(100, 116, 139, 0.16)", "highlight": "rgba(148,163,184,0.4)"},
                "width": 0.6,
            }
        )

    for _, row in highlight_top.iterrows():
        row_id = int(row["_source_row_id"])
        similarity = float(row["similarity_pct"])
        edges.append(
            {
                "from": "selected",
                "to": f"patient-{row_id}",
                "value": max(similarity / 14, 1.0),
                "color": {"color": "rgba(125, 211, 252, 0.78)", "highlight": "#ffffff"},
                "width": max(1.6, round(similarity / 26, 2)),
            }
        )

    for left_id, right_id in sorted(internal_edge_pairs):
        pair_score = score_lookup[frozenset((left_id, right_id))]
        edges.append(
            {
                "from": left_id,
                "to": right_id,
                "value": max(pair_score / 28, 0.5),
                "color": {"color": "rgba(148, 163, 184, 0.28)", "highlight": "rgba(255,255,255,0.72)"},
                "width": max(0.8, round(pair_score / 48, 2)),
            }
        )

    legend = [
        {"label": t(texts, "graph_legend_selected"), "color": "#60a5fa"},
        {"label": t(texts, "graph_legend_highlighted"), "color": "#14b8a6"},
        {"label": t(texts, "graph_legend_positive"), "color": "#ef4444"},
        {"label": t(texts, "graph_legend_negative"), "color": "#0f766e"},
        {"label": t(texts, "graph_legend_background"), "color": "#64748b"},
        {"label": t(texts, "graph_legend_strong"), "color": "#7dd3fc"},
        {"label": t(texts, "graph_legend_weak"), "color": "#94a3b8"},
    ]
    return {"nodes": nodes, "edges": edges, "legend": legend}


def render_similarity_graph_component(graph_data: dict[str, object], texts: dict[str, str]) -> None:
    payload = json.dumps(graph_data)
    html = f"""
    <div style="background:#111827;border-radius:14px;padding:10px 10px 6px 10px;border:1px solid #1f2937;">
      <div id="legend" style="display:flex;flex-wrap:wrap;gap:10px;padding:4px 6px 10px 6px;color:#e5e7eb;font:12px Arial,sans-serif;"></div>
      <div id="network" style="width:100%;height:420px;border-radius:10px;background:#0b1220;"></div>
      <div id="fallback" style="display:none;padding:10px;color:#cbd5e1;font:13px Arial,sans-serif;">{t(texts, "graph_fallback_note")}</div>
    </div>
    <script src="https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js"></script>
    <script>
      const payload = {payload};
      const legend = document.getElementById("legend");
      payload.legend.forEach((item) => {{
        const el = document.createElement("div");
        el.style.display = "flex";
        el.style.alignItems = "center";
        el.style.gap = "6px";
        el.innerHTML = `<span style="display:inline-block;width:10px;height:10px;border-radius:999px;background:${{item.color}};"></span><span>${{item.label}}</span>`;
        legend.appendChild(el);
      }});

      function buildGraph() {{
        if (!window.vis || !window.vis.Network) {{
          document.getElementById("fallback").style.display = "block";
          return;
        }}
        const nodes = new vis.DataSet(payload.nodes);
        const edges = new vis.DataSet(payload.edges);
        const container = document.getElementById("network");
        const options = {{
          autoResize: true,
          physics: false,
          interaction: {{ hover: true, zoomView: true, dragView: true, dragNodes: true }},
          nodes: {{
            shape: "dot",
            borderWidth: 1.5,
            shadow: {{ enabled: true, color: "rgba(255,255,255,0.12)", size: 10, x: 0, y: 0 }}
          }},
          edges: {{
            smooth: {{ type: "dynamic" }},
            selectionWidth: 2,
            hoverWidth: 1.5
          }}
        }};
        const network = new vis.Network(container, {{ nodes, edges }}, options);
        const labelThreshold = 1.08;

        function refreshLabels(scale, hoveredId = null) {{
          const updates = payload.nodes.map((node) => {{
            if (node.id === "selected") {{
              return {{ id: node.id, label: node.originalLabel, font: node.font }};
            }}
            const shouldShow = node.alwaysLabel || scale >= labelThreshold || hoveredId === node.id;
            return {{ id: node.id, label: shouldShow ? node.originalLabel : "", font: shouldShow ? node.font : {{ ...node.font, size: 0 }} }};
          }});
          nodes.update(updates);
        }}

        refreshLabels(network.getScale());
        network.on("zoom", () => refreshLabels(network.getScale()));
        network.on("hoverNode", (params) => refreshLabels(network.getScale(), params.node));
        network.on("blurNode", () => refreshLabels(network.getScale()));
        network.on("click", (params) => {{
          if (params.nodes.length) {{
            const nodeId = params.nodes[0];
            network.focus(nodeId, {{ scale: Math.max(network.getScale(), 1.1), animation: true }});
            network.selectNodes([nodeId]);
            network.selectEdges(network.getConnectedEdges(nodeId));
            refreshLabels(network.getScale(), nodeId);
          }} else {{
            network.unselectAll();
            refreshLabels(network.getScale());
          }}
        }});
      }}

      setTimeout(buildGraph, 80);
      setTimeout(() => {{
        if (!window.vis || !window.vis.Network) {{
          document.getElementById("fallback").style.display = "block";
        }}
      }}, 1200);
    </script>
    """
    components.html(html, height=500)


@st.cache_resource(show_spinner=False)
def load_runtime_artifacts():
    return load_artifacts(ARTIFACTS_DIR)


@st.cache_data(show_spinner=False)
def load_runtime_data():
    return load_data(DATA_PATH)


def risk_label(probability: float, texts: dict[str, str]) -> str:
    return texts[risk_band_label_key(probability)]


def build_comparison_frame(metadata: dict[str, object], texts: dict[str, str]) -> pd.DataFrame:
    rows = []
    for row in metadata["cv_summary"]:
        rows.append(
            {
                t(texts, "model_name"): row["model_name"],
                t(texts, "threshold"): row["threshold"],
                t(texts, "pr_auc"): f'{row["pr_auc_mean"]:.3f} ± {row["pr_auc_std"]:.3f}',
                t(texts, "roc_auc"): f'{row["roc_auc_mean"]:.3f} ± {row["roc_auc_std"]:.3f}',
                t(texts, "balanced_accuracy"): f'{row["balanced_accuracy_mean"]:.3f} ± {row["balanced_accuracy_std"]:.3f}',
                t(texts, "recall"): f'{row["recall_mean"]:.3f} ± {row["recall_std"]:.3f}',
                t(texts, "specificity"): f'{row["specificity_mean"]:.3f} ± {row["specificity_std"]:.3f}',
                t(texts, "ppv"): f'{row["ppv_mean"]:.3f} ± {row["ppv_std"]:.3f}',
                t(texts, "npv"): f'{row["npv_mean"]:.3f} ± {row["npv_std"]:.3f}',
                t(texts, "brier_score"): f'{row["brier_score_mean"]:.3f} ± {row["brier_score_std"]:.3f}',
                t(texts, "accuracy"): f'{row["accuracy_mean"]:.3f} ± {row["accuracy_std"]:.3f}',
                t(texts, "params"): str(row["params"]),
            }
        )
    return pd.DataFrame(rows)


def similarity_group_label(texts: dict[str, str], key: str) -> str:
    return {
        "very_high": t(texts, "group_very_high"),
        "high": t(texts, "group_high"),
        "moderate": t(texts, "group_moderate"),
        "broad": t(texts, "group_broad"),
    }[key]


def outcome_label(texts: dict[str, str], value: int) -> str:
    return t(texts, "outcome_positive") if int(value) == 1 else t(texts, "outcome_negative")


def build_patient_input(texts: dict[str, str]) -> dict[str, float | int]:
    return {
        "male": st.sidebar.selectbox(
            t(texts, "sex"),
            options=[0, 1],
            format_func=lambda value: t(texts, "female") if value == 0 else t(texts, "male"),
        ),
        "age": st.sidebar.slider(t(texts, "age"), min_value=30, max_value=80, value=52),
        "education": st.sidebar.selectbox(
            t(texts, "education"),
            options=[1, 2, 3, 4],
            help=t(texts, "education_help"),
        ),
        "currentSmoker": st.sidebar.selectbox(
            t(texts, "current_smoker"),
            options=[0, 1],
            format_func=lambda value: t(texts, "no") if value == 0 else t(texts, "yes"),
        ),
        "cigsPerDay": st.sidebar.slider(t(texts, "cigs_per_day"), min_value=0, max_value=50, value=0),
        "BPMeds": st.sidebar.selectbox(
            t(texts, "bp_meds"),
            options=[0, 1],
            format_func=lambda value: t(texts, "no") if value == 0 else t(texts, "yes"),
        ),
        "prevalentStroke": st.sidebar.selectbox(
            t(texts, "stroke"),
            options=[0, 1],
            format_func=lambda value: t(texts, "no") if value == 0 else t(texts, "yes"),
        ),
        "prevalentHyp": st.sidebar.selectbox(
            t(texts, "prevalent_hyp"),
            options=[0, 1],
            format_func=lambda value: t(texts, "no") if value == 0 else t(texts, "yes"),
        ),
        "diabetes": st.sidebar.selectbox(
            t(texts, "diabetes"),
            options=[0, 1],
            format_func=lambda value: t(texts, "no") if value == 0 else t(texts, "yes"),
        ),
        "totChol": st.sidebar.slider(t(texts, "tot_chol"), min_value=120, max_value=400, value=210),
        "sysBP": st.sidebar.slider(t(texts, "sys_bp"), min_value=90, max_value=220, value=128),
        "diaBP": st.sidebar.slider(t(texts, "dia_bp"), min_value=60, max_value=140, value=82),
        "BMI": st.sidebar.slider(t(texts, "bmi"), min_value=16.0, max_value=50.0, value=27.0, step=0.1),
        "heartRate": st.sidebar.slider(t(texts, "heart_rate"), min_value=40, max_value=130, value=72),
        "glucose": st.sidebar.slider(t(texts, "glucose"), min_value=50, max_value=250, value=92),
    }


def main() -> None:
    st.set_page_config(page_title="Clinical Risk Explorer", page_icon=":stethoscope:", layout="wide")
    inject_styles()
    language = st.sidebar.selectbox(
        "Language / Dil",
        options=["en", "tr"],
        index=0,
        format_func=lambda code: "English" if code == "en" else "Türkçe",
    )
    texts = TRANSLATIONS[language]

    try:
        artifacts = load_runtime_artifacts()
        data = load_runtime_data()
    except Exception as exc:
        st.error(f'{t(texts, "model_not_ready")}: {exc}')
        st.stop()

    metadata = artifacts.metadata
    test_metrics = metadata["test_metrics"]
    comparison_frame = build_comparison_frame(metadata, texts)

    st.sidebar.header(t(texts, "sidebar_header"))
    user_input = build_patient_input(texts)
    input_frame = pd.DataFrame([user_input], columns=FEATURE_COLUMNS)

    probability = float(artifacts.model.predict_proba(input_frame)[0][1])
    label_key = risk_band_label_key(probability)
    label = risk_label(probability, texts)
    threshold = float(metadata["threshold"])
    nearest = find_nearest_patients(input_frame, data, top_k=50)
    top50_positive_rate = float((nearest["neighbors"][TARGET_COLUMN] == 1).mean() * 100)
    validated_reference, _ = validate_data(data)
    reference_scores = pd.Series(
        artifacts.model.predict_proba(validated_reference[FEATURE_COLUMNS])[:, 1]
    )
    risk_distribution_frame = build_risk_distribution_frame(reference_scores, probability)
    risk_percentile = float((reference_scores <= probability).mean() * 100)

    st.title(t(texts, "page_title"))
    st.caption(t(texts, "caption"))
    st.info(t(texts, "compact_note"))
    st.caption(t(texts, "runtime_note"))
    if label_key == "risk_high":
        st.error(label)
    elif label_key == "risk_moderate":
        st.warning(label)
    else:
        st.success(label)
    st.caption(t(texts, "risk_band_note"))

    top_left, top_mid, top_right = st.columns([1.2, 1.2, 1.6])
    with top_left:
        st.metric(t(texts, "predicted_risk"), f"{probability:.1%}")
        attention_text = (
            t(texts, "high_attention") if label_key == "risk_high"
            else t(texts, "medium_attention") if label_key == "risk_moderate"
            else t(texts, "low_attention")
        )
        st.caption(attention_text)
    with top_mid:
        st.metric(t(texts, "decision_threshold"), f"{threshold:.3f}")
        st.metric(t(texts, "closest_match"), f'{float(nearest["closest"]["similarity_pct"]):.1f}%')
    with top_right:
        st.metric(t(texts, "selected_model"), metadata["selected_model_display_name"])
        st.caption(
            f'{t(texts, "model_card_note")} '
            f'{t(texts, "trained_at")}: {metadata["train_timestamp"]} | '
            f'{t(texts, "data_version")}: {metadata["data_version"]}'
        )

    summary_tab, neighbors_tab, insights_tab = st.tabs(
        [t(texts, "summary_tab"), t(texts, "neighbors_tab"), t(texts, "insights_tab")]
    )

    with summary_tab:
        section_intro(t(texts, "final_quality"), t(texts, "test_metric_note"))
        key_cols = st.columns(4)
        key_cols[0].metric(t(texts, "pr_auc"), f'{test_metrics["pr_auc"]:.3f}')
        key_cols[1].metric(t(texts, "roc_auc"), f'{test_metrics["roc_auc"]:.3f}')
        key_cols[2].metric(t(texts, "recall"), f'{test_metrics["recall"]:.3f}')
        key_cols[3].metric(t(texts, "specificity"), f'{test_metrics["specificity"]:.3f}')

        st.markdown(f'**{t(texts, "secondary_metrics")}**')
        secondary_cols = st.columns(4)
        secondary_cols[0].metric(t(texts, "ppv"), f'{test_metrics["ppv"]:.3f}')
        secondary_cols[1].metric(t(texts, "npv"), f'{test_metrics["npv"]:.3f}')
        secondary_cols[2].metric(t(texts, "balanced_accuracy"), f'{test_metrics["balanced_accuracy"]:.3f}')
        secondary_cols[3].metric(t(texts, "brier_score"), f'{test_metrics["brier_score"]:.3f}')

        compact_meta = st.columns(4)
        compact_meta[0].metric(t(texts, "train_rows"), str(metadata["training_rows"]))
        compact_meta[1].metric(t(texts, "test_rows"), str(metadata["test_rows"]))
        compact_meta[2].metric(t(texts, "validation_removed"), str(metadata["validation_report"]["dropped_rows"]))
        compact_meta[3].metric(t(texts, "outcome_rate_top50"), f"{top50_positive_rate:.1f}%")

        st.markdown(f'**{t(texts, "risk_distribution_chart")}**')
        st.caption(t(texts, "risk_distribution_note"))
        distribution_cols = st.columns([1, 3])
        distribution_cols[0].metric(t(texts, "risk_percentile"), f"{risk_percentile:.1f}%")
        distribution_cols[1].bar_chart(
            risk_distribution_frame.rename(
                columns={
                    "cohort_count": t(texts, "cohort_count"),
                    "selected_bin": t(texts, "selected_bin"),
                }
            ),
            height=280,
        )

    with neighbors_tab:
        section_intro(t(texts, "nearest_title"), t(texts, "top50_note"))
        closest = nearest["closest"]
        closest_cols = st.columns(4)
        closest_cols[0].metric(t(texts, "patient_id"), str(int(closest["_source_row_id"])))
        closest_cols[1].metric(t(texts, "similarity_pct"), f'{float(closest["similarity_pct"]):.1f}%')
        closest_cols[2].metric(t(texts, "group_label"), similarity_group_label(texts, str(closest["similarity_group"])))
        closest_cols[3].metric(t(texts, "actual_outcome"), outcome_label(texts, int(closest[TARGET_COLUMN])))
        st.caption(t(texts, "similarity_note"))

        group_frame = pd.DataFrame(nearest["group_summary"])
        group_frame[t(texts, "group_label")] = group_frame["group_key"].apply(lambda key: similarity_group_label(texts, str(key)))
        group_frame = group_frame[[t(texts, "group_label"), "count", "share_pct", "avg_similarity_pct"]]
        group_frame.columns = [
            t(texts, "group_label"),
            "Count",
            t(texts, "group_share"),
            t(texts, "avg_similarity"),
        ]
        st.dataframe(group_frame, width="stretch", hide_index=True)

        neighbor_viz_left, neighbor_viz_right = st.columns([1.15, 1.85])
        with neighbor_viz_left:
            st.markdown(f'**{t(texts, "similar_patients_chart")}**')
            st.caption(t(texts, "top10_similarity_note"))
            st.bar_chart(build_similarity_bar_frame(nearest["neighbors"]), height=280)
        with neighbor_viz_right:
            st.markdown(f'**{t(texts, "relationship_graph")}**')
            st.caption(t(texts, "graph_note"))
            render_similarity_graph_component(build_similarity_graph_data(data, input_frame, nearest["neighbors"], texts), texts)

        neighbors_frame = nearest["neighbors"][
            ["_source_row_id", "similarity_pct", "similarity_group", TARGET_COLUMN, "age", "male", "currentSmoker", "sysBP", "totChol", "BMI", "glucose"]
        ].copy()
        neighbors_frame[t(texts, "patient_id")] = neighbors_frame["_source_row_id"].astype(int)
        neighbors_frame[t(texts, "similarity_pct")] = neighbors_frame["similarity_pct"].map(lambda value: f"{float(value):.1f}%")
        neighbors_frame[t(texts, "group_label")] = neighbors_frame["similarity_group"].map(lambda key: similarity_group_label(texts, str(key)))
        neighbors_frame[t(texts, "actual_outcome")] = neighbors_frame[TARGET_COLUMN].map(lambda value: outcome_label(texts, int(value)))
        neighbors_frame[t(texts, "sex")] = neighbors_frame["male"].map(lambda value: t(texts, "male") if int(value) == 1 else t(texts, "female"))
        neighbors_frame[t(texts, "current_smoker")] = neighbors_frame["currentSmoker"].map(lambda value: t(texts, "yes") if int(value) == 1 else t(texts, "no"))
        neighbors_frame[t(texts, "age")] = neighbors_frame["age"]
        neighbors_frame[t(texts, "sys_bp")] = neighbors_frame["sysBP"]
        neighbors_frame[t(texts, "tot_chol")] = neighbors_frame["totChol"]
        neighbors_frame[t(texts, "bmi")] = neighbors_frame["BMI"]
        neighbors_frame[t(texts, "glucose")] = neighbors_frame["glucose"]
        neighbors_frame = neighbors_frame[
            [
                t(texts, "patient_id"),
                t(texts, "similarity_pct"),
                t(texts, "group_label"),
                t(texts, "actual_outcome"),
                t(texts, "age"),
                t(texts, "sex"),
                t(texts, "current_smoker"),
                t(texts, "sys_bp"),
                t(texts, "tot_chol"),
                t(texts, "bmi"),
                t(texts, "glucose"),
            ]
        ]
        st.markdown(f'**{t(texts, "top_matches")}**')
        st.caption(t(texts, "top_matches_note"))
        st.dataframe(neighbors_frame, width="stretch", hide_index=True)

    with insights_tab:
        section_intro(t(texts, "explanation_title"), t(texts, "cv_metric_note"))
        explanation = metadata["explainability"]
        if metadata["selected_model"] == "logistic_regression":
            explanation_bundle = build_patient_vs_cohort_explanation(artifacts.model, input_frame, data)
            display_rows = explanation_bundle["display_rows"]
            technical_rows = explanation_bundle["technical_rows"]
            contribution_chart_frame = build_display_explanation_chart(display_rows, texts)
            display_frame = pd.DataFrame(
                [
                    {
                        t(texts, "feature"): explanation_feature_label(texts, str(row["feature_label_key"])),
                        t(texts, "patient_value"): round(float(row["patient_value"]), 2),
                        t(texts, "cohort_baseline"): round(float(row["cohort_value"]), 2),
                        t(texts, "relative_position"): t(texts, str(row["relation_key"])),
                        t(texts, "score_effect"): t(texts, str(row["score_direction_key"])),
                    }
                    for row in display_rows[:8]
                ]
            )
            technical_local_frame = pd.DataFrame(technical_rows[:12])[["feature", "contribution"]]
            technical_local_frame.columns = [t(texts, "feature"), t(texts, "contribution")]
            global_rows = pd.DataFrame(explanation["rows"][:12])[["feature", "coefficient"]]
            global_rows.columns = [t(texts, "feature"), t(texts, "coefficient")]
            left, right = st.columns(2)
            with left:
                st.markdown(f'**{t(texts, "contribution_chart")}**')
                st.caption(t(texts, "contribution_note"))
                st.bar_chart(contribution_chart_frame, height=280)
                st.dataframe(display_frame, width="stretch", hide_index=True)
            with right:
                st.markdown(f'**{t(texts, "global_explanation")}**')
                st.dataframe(global_rows, width="stretch", hide_index=True)
        else:
            st.markdown(f'**{t(texts, "limited_explanation")}**')
            st.caption(f'{t(texts, "limited_explanation_text")} {t(texts, "contribution_note")}')
            importances = pd.DataFrame(explanation["rows"][:12])[["feature", "importance"]]
            importances.columns = [t(texts, "feature"), t(texts, "importance")]
            st.dataframe(importances, width="stretch", hide_index=True)

        with st.expander(t(texts, "advanced_details"), expanded=False):
            if metadata["selected_model"] == "logistic_regression":
                st.markdown(f'**{t(texts, "local_explanation")}**')
                st.dataframe(technical_local_frame, width="stretch", hide_index=True)
                st.markdown(f'**{t(texts, "global_explanation")}**')
                st.dataframe(global_rows, width="stretch", hide_index=True)
            st.markdown(f'**{t(texts, "cv_title")}**')
            st.dataframe(comparison_frame, width="stretch", hide_index=True)
            calibration_frame = pd.DataFrame(test_metrics["calibration_curve"])
            calibration_frame.columns = [t(texts, "predicted_probability"), t(texts, "observed_rate")]
            st.markdown(f'**{t(texts, "calibration_curve")}**')
            st.line_chart(calibration_frame.set_index(t(texts, "predicted_probability")))
            st.dataframe(calibration_frame, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
