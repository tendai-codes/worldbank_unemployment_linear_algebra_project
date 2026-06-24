# Rust Feature Engineering Pipeline

This Rust CLI is the production-style preprocessing layer for the World Bank downturn-risk project.

It reads:

```text
../data/worldbank_panel_final.csv
```

and writes:

```text
../data/worldbank_panel_engineered_rust.csv
```

## Generated columns

The output keeps the raw country-year fields and adds:

- `gdp_growth_next_year`
- `downturn_risk_next_year`
- `downturn_next_year` as a compatibility alias
- `{feature}_lag1`
- `{feature}_lag2`
- `{feature}_change_1y`
- `{feature}_trend_3y`

for these base features:

- `unemployment`
- `inflation`
- `gdp_growth`
- `life_expectancy`
- `population_growth`

## Run

From the project root:

```bash
./scripts/run_rust_pipeline.sh
```

or from this directory:

```bash
cargo run -- ../data/worldbank_panel_final.csv ../data/worldbank_panel_engineered_rust.csv
```

## Role in the project

Rust is useful for the repeatable preprocessing part of the project. It gives strict typing, explicit error handling, and fast CSV processing. In this case Rust controls feature engineering processing. Python remains better for exploration, notebooks, modelling, and dashboard work and is used to process the engineered CSV for model training, notebooks, and the Streamlit dashboard.

