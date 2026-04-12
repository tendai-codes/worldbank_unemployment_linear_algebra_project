# Macroeconomic Downturn Prediction Using Linear Algebra and Machine Learning

## Live app

Deployed dashboard: https://macroeconomic-risk.streamlit.app/

## Problem

This project constructs a country-year macroeconomic panel dataset from World Bank indicators and applies linear algebra and supervised learning techniques to estimate the probability that a country will experience an economic downturn in the following year.

The main goal is to represent each country-year observation as a vector in macroeconomic feature space, analyse how those vectors evolve over time, and use those representations for classification and scenario simulation.

## Linear algebra perspective

Each country-year observation is represented as a feature vector:

x ∈ ℝⁿ

using macroeconomic indicators such as:

- GDP growth
- inflation
- unemployment
- life expectancy
- population growth

The project expands these vectors with temporal structure using:

- lag features
- annual change features
- 3-year trend features

This supports three linear algebra ideas:

1. **Feature-space representation**  
   Countries are embedded as vectors in a common macroeconomic space.

2. **Temporal transformation**  
   Annual changes and rolling least-squares slopes capture direction and momentum.

3. **Similarity geometry**  
   Similar countries are identified using Euclidean distance in feature space.

## Project workflow

The workflow consists of four stages:

1. Build a cleaned country-year panel dataset from World Bank indicators
2. Create matrix-style representations with lag, delta, and trend features
3. Train classification models to estimate next-year downturn risk
4. Deploy an interactive Streamlit dashboard for scenario testing and multi-country comparison

## Key outputs

### 1. Predictive model
The project trains classification models on country-year macroeconomic feature vectors to estimate the probability of a downturn in the following year.

### 2. Streamlit dashboard
The dashboard allows users to:

- select a country
- adjust macroeconomic conditions
- simulate alternative economic scenarios
- estimate predicted downturn probability
- compare similar countries
- view multi-country time-series charts across levels, annual changes, and trends

Live app: https://macroeconomic-risk.streamlit.app/

### 3. Time-series diagnostics notebook
The diagnostics notebook visualises:

- raw macroeconomic levels
- annual changes
- 3-year least-squares trends
- downturn-next-year overlays

This notebook helps justify the inclusion of temporal features in the final model.

## Time-series diagnostics

The repository also includes a time-series diagnostics notebook that visualises:

- raw macroeconomic levels
- annual changes
- 3-year least-squares trends
- downturn-next-year overlays

This notebook is used to interpret whether downturn periods are better explained by structural levels, short-term deterioration, or sustained medium-term trends.

These diagnostics support the feature engineering choices used in the predictive model, particularly the inclusion of delta and trend features.

## Run locally

Install dependencies:

```bash
pip install -r requirements-dashboard.txt

python train_downturn_model.py

python -m streamlit run dashboard/app.py