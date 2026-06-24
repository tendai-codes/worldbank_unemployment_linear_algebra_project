# Downturn Dashboard: Run Guide

The dashboard now uses the scalable Rust + Python workflow.

```text
Rust feature engineering
→ data/worldbank_panel_engineered_rust.csv
→ Python model training
→ Streamlit dashboard
```

## 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dashboard.txt
```

Rust must also be available:

```bash
rustc --version
cargo --version
```

## 2. Generate the Rust-engineered dataset

From the project root:

```bash
./scripts/run_rust_pipeline.sh
```

This creates:

```text
data/worldbank_panel_engineered_rust.csv
```

## 3. Train the dashboard model

```bash
python train_downturn_model.py
```

The training script now expects the Rust-engineered CSV. It creates or updates:

```text
models/downturn_random_forest.joblib
models/model_features.csv
models/cv_metrics.csv
models/optimal_threshold.csv
models/country_baselines_latest.csv
data/feature_matrix.csv
```

## 4. Run the dashboard

```bash
python -m streamlit run dashboard/app.py
```

## 5. Full rebuild command sequence

```bash
./scripts/run_rust_pipeline.sh
python train_downturn_model.py
python -m streamlit run dashboard/app.py
```

## Notes

- The dashboard reads `data/worldbank_panel_engineered_rust.csv` for time-series views.
- The model is trained from the same Rust-engineered dataset.
- Re-run the Rust pipeline and training script whenever `data/worldbank_panel_final.csv` changes.
