# Clinical Risk Explorer

Clinical Risk Explorer is a Streamlit demo for exploring calibrated 10-year coronary heart disease risk on the Framingham dataset. It focuses on transparent model behavior, cohort comparison, and clear risk-band communication rather than clinical decision support.

## Features

- Offline-trained scikit-learn model artifacts loaded at runtime
- Streamlit dashboard with patient inputs, risk bands, and model snapshot metrics
- Similar-patient exploration against the validated reference cohort
- Local and global explanation views for the selected model
- English/Turkish interface copy in a single app

## Project Structure

```text
app.py                 Streamlit interface
modeling.py            Data validation, training, metrics, and explanation helpers
train_model.py         Offline model training entrypoint
framingham.csv         Demo dataset
artifacts/             Pretrained model metadata and metrics
tests/                 Smoke and modeling tests
requirements.txt       Runtime dependencies
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
python -m pytest
```

## Important Note

This repository is an educational portfolio/demo project. It is not a medical device, clinical protocol, or diagnostic tool. Risk thresholds and outputs are intended for model exploration only.
