# Downturn dashboard: run guide

## Place these files in your repo

Recommended project structure:

```text
worldbank_unemployment_linear_algebra/
├── train_downturn_model.py
├── requirements-dashboard.txt
├── dashboard/
│   ├── app.py
│   ├── model_utils.py
│   └── scenario_utils.py
├── data/
│   ├── worldbank_panel_final.csv
│   └── ...
├── models/
│   └── (created after training)
```

## 1. Install dependencies

```bash
pip install -r requirements-dashboard.txt
```

If you already use a project `requirements.txt`, you can merge these lines into it instead:

```text
streamlit
plotly
joblib
pandas
scikit-learn
```

## 2. Save your cleaned panel CSV

The training script looks for one of these files:

```text
data/worldbank_panel_final.csv
data/worldbank_panel.csv
```

The CSV must contain at least these columns:

- `country_code`
- `year`
- `unemployment`
- `inflation`
- `gdp_growth`
- `life_expectancy`
- `population_growth`

Run this export cell near the end of your notebook:

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

## 3. Train the dashboard model

From the project root:

```bash
python train_downturn_model.py
```

This creates a `models/` folder with:

- `downturn_random_forest.joblib`
- `model_features.csv`
- `cv_metrics.csv`
- `country_baselines_latest.csv`

## 4. Run the dashboard

From the project root:

```bash
streamlit run dashboard/app.py
```

## 5. What the dashboard does

- loads the trained Random Forest model
- loads the latest baseline values for each country
- lets you pick a country baseline
- lets you adjust current and lagged indicators
- predicts next-year downturn probability
- shows baseline vs scenario changes
- shows top feature importances

## Notes

- Run `python train_downturn_model.py` again whenever you update the cleaned panel data.
- If Streamlit says the model file is missing, it means the training step has not been run yet.
- The dashboard uses the Part 3 binary target: next year's GDP growth below zero.
