# Clinical Risk Explorer

<div align="center">

**Interactive clinical risk exploration dashboard with explainability and cohort comparison**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/tests-13%20passing-brightgreen)](#quality)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

[English](#english) · [Türkçe](#türkçe) · [Quick Start](#quick-start) · [Safety Note](#safety-note)

</div>

---

## English

Clinical Risk Explorer is a Streamlit portfolio demo for exploring calibrated 10-year coronary heart disease risk on the Framingham dataset.

The project focuses on **transparent model behavior**: it shows model metadata, performance metrics, risk bands, similar-patient comparisons, and explanation views in a compact dashboard. It is designed as an educational and product-style example, not as a clinical decision system.

### Highlights

- Calibrated offline scikit-learn model loaded from local artifacts
- Streamlit dashboard with patient inputs and risk-band output
- Model snapshot panel with evaluation metrics and selected model details
- Similar-patient exploration against a validated reference cohort
- Local/global explanation views for interpretable model behavior
- English and Turkish interface copy in the same app

### Project Structure

```text
app.py                 Streamlit dashboard
modeling.py            Data validation, model training, metrics, explanations
train_model.py         Offline model training entrypoint
framingham.csv         Demo dataset
artifacts/             Pretrained model, metadata, and CV metrics
tests/                 Smoke tests, graph tests, and modeling tests
requirements.txt       Runtime and test dependencies
```

---

## Türkçe

Clinical Risk Explorer, Framingham veri seti üzerinde 10 yıllık koroner kalp hastalığı riskini keşfetmek için hazırlanmış Streamlit tabanlı bir portföy demosudur.

Proje, yalnızca bir tahmin sonucu göstermek yerine **model davranışını anlaşılır kılmaya** odaklanır. Dashboard içinde model metrikleri, risk bantları, benzer hasta karşılaştırmaları ve açıklanabilirlik panelleri birlikte sunulur. Klinik karar aracı değil, eğitim ve portföy amaçlı bir veri ürünü örneğidir.

### Öne Çıkanlar

- Lokal artifact olarak yüklenen kalibre edilmiş scikit-learn modeli
- Hasta girdileri ve risk bandı çıktısı içeren Streamlit dashboard
- Model seçimi, metrikler ve eğitim özeti için model snapshot paneli
- Validated referans kohort üzerinden benzer hasta keşfi
- Model davranışını açıklayan local/global explainability görünümleri
- Aynı uygulama içinde Türkçe ve İngilizce arayüz metinleri

### Proje Yapısı

```text
app.py                 Streamlit dashboard
modeling.py            Veri doğrulama, model eğitimi, metrikler, açıklamalar
train_model.py         Offline model eğitim komutu
framingham.csv         Demo veri seti
artifacts/             Eğitilmiş model, metadata ve CV metrikleri
tests/                 Smoke, grafik ve modelleme testleri
requirements.txt       Çalıştırma ve test bağımlılıkları
```

---

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Quality

The project was verified with:

```bash
python -m pytest
```

Latest local verification result before publishing: `13 passed`.

## Safety Note

This repository is an educational portfolio/demo project. It is **not** a medical device, diagnostic system, clinical protocol, or substitute for professional medical judgment. Risk thresholds and outputs are intended for model exploration only.

## License

Released under the [MIT License](LICENSE).
