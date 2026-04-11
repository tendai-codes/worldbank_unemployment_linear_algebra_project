# World Bank Linear Algebra Project

This project uses World Bank API data to study matrix structure in cross-country macroeconomic data and extend that analysis into a simple predictive downturn-risk model.

## Project goals

### Part 1: Unemployment-only analysis
Build a **country × year** unemployment matrix and study:

- missingness structure
- matrix rank
- REF and RREF on a subset
- correlation between years
- country similarity
- PCA on unemployment trajectories

### Part 2: Multi-indicator matrix analysis
Build a **country × feature** matrix using:

- unemployment
- inflation
- GDP growth
- life expectancy
- population growth

Then analyse:

- missingness
- rank
- feature correlation
- country similarity
- PCA

### Part 3: Predictive modelling
Use the cleaned **country-year panel** to classify **next-year downturn risk** with tree-based models.

## Project structure

```text
worldbank_unemployment_linear_algebra/
├── README.md
├── main.py
├── requirements.txt
├── requirements-dashboard.txt
├── RUN_DASHBOARD.md
├── train_downturn_model.py
├── src/
├── dashboard/
├── notebooks/
├── data/
├── models/
└── output/
```

## Why the code is split into modules

- `src/config.py` → constants and indicator codes
- `src/api_client.py` → World Bank API access
- `src/data_processing.py` → DataFrame and matrix construction
- `src/matrix_analysis.py` → rank, REF/RREF, similarity, PCA
- `src/pipeline.py` → step-by-step project workflow
- `main.py` → entry point
- `train_downturn_model.py` → Part 3 model training and artifact generation
- `dashboard/` → scenario simulation dashboard for the Part 3 model

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install core requirements:

```bash
pip install -r requirements.txt
```

Install dashboard requirements as well if you want to run the simulation app:

```bash
pip install -r requirements-dashboard.txt
```

## Run the core analysis

```bash
python main.py
```

## Export notebook outputs to `data/`

Run this cell near the end of your notebook after the relevant dataframes have been created:

```python
from pathlib import Path

data_dir = Path("../data") if Path.cwd().name == "notebooks" else Path("data")
data_dir.mkdir(parents=True, exist_ok=True)

if "df_unemployment" in globals():
    df_unemployment.to_csv(data_dir / "unemployment.csv", index=False)

if "unemployment_matrix" in globals():
    unemployment_matrix.to_csv(data_dir / "unemployment_matrix.csv", index=True)

if "indicator_dataframes" in globals():
    for name, df in indicator_dataframes.items():
        df.to_csv(data_dir / f"{name}.csv", index=False)

if "panel_df" in globals():
    panel_df.to_csv(data_dir / "worldbank_panel.csv", index=False)

if "panel_df_final" in globals():
    panel_df_final.to_csv(data_dir / "worldbank_panel_final.csv", index=False)

if "feature_matrix" in globals():
    feature_matrix.to_csv(data_dir / "feature_matrix.csv", index=True)

print("Export complete. Files saved to:", data_dir.resolve())
```

## Expected data artifacts

The analysis should save CSV snapshots in `data/` such as:

- `unemployment.csv`
- `unemployment_matrix.csv`
- `inflation.csv`
- `gdp_growth.csv`
- `life_expectancy.csv`
- `population_growth.csv`
- `worldbank_panel.csv`
- `worldbank_panel_final.csv`
- `feature_matrix.csv`

## Note on REF / RREF

REF and RREF are computed on a smaller rounded subset of the unemployment matrix.

For large real-valued economic matrices, exact symbolic row reduction becomes messy and less interpretable. Numeric rank is the serious tool. REF/RREF is included mainly to demonstrate linear algebra reasoning.
