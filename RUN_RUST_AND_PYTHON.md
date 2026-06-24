# Running the Scalable Rust + Python Workflow

This repository now uses Rust as the deterministic feature-engineering layer and Python as the modelling, notebook, and Streamlit layer.

## Architecture

```text
Raw / cleaned World Bank panel
        ↓
Rust CLI feature engineering
        ↓
data/worldbank_panel_engineered_rust.csv
        ↓
Python model training
        ↓
models/*.csv and models/downturn_random_forest.joblib
        ↓
Streamlit dashboard
```

## 1. Install dependencies

### Rust

```bash
rustc --version
cargo --version
```

If Rust is missing, install it from <https://rustup.rs/>.

### Python

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dashboard.txt
```

## 2. Generate engineered features with Rust

From the project root:

```bash
./scripts/run_rust_pipeline.sh
```

Equivalent manual command:

```bash
cd rust_pipeline
cargo run -- ../data/worldbank_panel_final.csv ../data/worldbank_panel_engineered_rust.csv
```

This creates:

```text
data/worldbank_panel_engineered_rust.csv
```

The Rust output includes:

- current macroeconomic levels
- 1-year lags
- 2-year lags
- 1-year annual changes
- 3-year least-squares trend slopes
- next-year downturn target

## 3. Train the Python model from Rust output

From the project root:

```bash
python train_downturn_model.py
```

The training script now expects:

```text
data/worldbank_panel_engineered_rust.csv
```

It no longer recomputes lags, annual changes, trends, or the target in Python.

## 4. Run Streamlit

```bash
python -m streamlit run dashboard/app.py
```

The dashboard uses:

```text
data/worldbank_panel_engineered_rust.csv
```

for time-series displays and uses the model artefacts created by `train_downturn_model.py` for predictions.

## 5. Recommended full rebuild

```bash
./scripts/run_rust_pipeline.sh
python train_downturn_model.py
python -m streamlit run dashboard/app.py
```

## 6. Git hygiene

Commit these:

```text
rust_pipeline/Cargo.toml
rust_pipeline/Cargo.lock
rust_pipeline/src/main.rs
rust_pipeline/README.md
scripts/run_rust_pipeline.sh
RUN_RUST_AND_PYTHON.md
data/worldbank_panel_engineered_rust.csv
models/*.csv
models/downturn_random_forest.joblib
```

Do not commit:

```text
rust_pipeline/target/
.venv/
__pycache__/
.ipynb_checkpoints/
```
