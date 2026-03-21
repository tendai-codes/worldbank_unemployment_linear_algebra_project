# World Bank Linear Algebra Project (Modular Version)

This project uses World Bank API data to study matrix structure in global economic data.

## Project goals

### Part 1: Unemployment-only analysis
Build a **country × year** unemployment matrix and study:

- missingness structure
- matrix rank
- REF and RREF on a subset
- correlation between years
- country similarity
- PCA on unemployment trajectories

### Part 2: Stronger multi-indicator matrix
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

---

## Why the code is split into modules

The project is organised so each file has one job:

- `src/config.py` → constants and indicator codes
- `src/api_client.py` → World Bank API access
- `src/data_processing.py` → DataFrame and matrix construction
- `src/matrix_analysis.py` → rank, REF/RREF, similarity, PCA
- `src/pipeline.py` → step-by-step project workflow
- `main.py` → entry point

This makes the project easier to debug, run, and explain.

---

## Folder structure

```text
worldbank_linear_algebra_modular_project
│
├── main.py
├── README.md
├── requirements.txt
│
├── src
│   ├── __init__.py
│   ├── config.py
│   ├── api_client.py
│   ├── data_processing.py
│   ├── matrix_analysis.py
│   └── pipeline.py
│
├── notebooks
│   └── notebook_worldbank_linear_algebra.ipynb
│
├── data
└── output
```

---

## Setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

Mac/Linux

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

---

## Run the project

```bash
python main.py
```

---

## CSV snapshots vs live API only

This project uses the sensible hybrid approach:

- fetch **live** from the World Bank API
- save **CSV snapshots** into `data/`

Why this is useful:

- live data keeps the project current
- CSV snapshots make results reproducible
- reruns are easier to debug
- you can show the extraction output clearly on GitHub

So yes, keeping CSV exports is useful here.

---

## Expected outputs

The script saves these snapshots in `data/`:

- `unemployment.csv`
- `unemployment_matrix.csv`
- `inflation.csv`
- `gdp_growth.csv`
- `life_expectancy.csv`
- `population_growth.csv`
- `worldbank_panel.csv`
- `feature_matrix.csv`

---

## Note on REF / RREF

REF and RREF are computed on a smaller rounded subset of the unemployment matrix.

That is intentional.

For large real-valued economic matrices, exact symbolic row reduction becomes messy and less interpretable. Numeric rank is the serious tool. REF/RREF is included mainly to demonstrate linear algebra reasoning.
