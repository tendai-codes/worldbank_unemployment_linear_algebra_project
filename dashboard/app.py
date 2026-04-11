from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

# Ensure project root is on sys.path when running via Streamlit
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.model_utils import (
    load_model,
    load_country_baselines,
    load_model_features,
    load_cv_metrics,
)
from dashboard.scenario_utils import (
    apply_scenario_preset,
    build_model_input_row,
)

st.set_page_config(
    page_title="Macroeconomic Downturn Risk Simulator",
    page_icon="📉",
    layout="wide",
)

MODEL_PATH = PROJECT_ROOT / "models" / "downturn_random_forest.joblib"
BASELINES_PATH = PROJECT_ROOT / "models" / "country_baselines_latest.csv"
FEATURES_PATH = PROJECT_ROOT / "models" / "model_features.csv"
CV_METRICS_PATH = PROJECT_ROOT / "models" / "cv_metrics.csv"

FEATURE_LABELS = {
    "gdp_growth": "GDP growth",
    "gdp_growth_change_1y": "GDP growth (1-year change)",
    "gdp_growth_lag1": "GDP growth (lag 1 year)",
    "gdp_growth_lag2": "GDP growth (lag 2 years)",
    "inflation": "Inflation",
    "inflation_change_1y": "Inflation (1-year change)",
    "inflation_lag1": "Inflation (lag 1 year)",
    "inflation_lag2": "Inflation (lag 2 years)",
    "unemployment": "Unemployment",
    "unemployment_change_1y": "Unemployment (1-year change)",
    "unemployment_lag1": "Unemployment (lag 1 year)",
    "unemployment_lag2": "Unemployment (lag 2 years)",
    "life_expectancy": "Life expectancy",
    "life_expectancy_lag1": "Life expectancy (lag 1 year)",
    "life_expectancy_lag2": "Life expectancy (lag 2 years)",
    "population_growth": "Population growth",
    "population_growth_lag1": "Population growth (lag 1 year)",
    "population_growth_lag2": "Population growth (lag 2 years)",
}

INPUT_LABELS = {
    "unemployment": "Unemployment (%)",
    "inflation": "Inflation (%)",
    "gdp_growth": "GDP growth (%)",
    "life_expectancy": "Life expectancy",
    "population_growth": "Population growth (%)",
    "unemployment_lag1": "Unemployment lag 1 (%)",
    "inflation_lag1": "Inflation lag 1 (%)",
    "gdp_growth_lag1": "GDP growth lag 1 (%)",
    "life_expectancy_lag1": "Life expectancy lag 1",
    "population_growth_lag1": "Population growth lag 1 (%)",
    "unemployment_lag2": "Unemployment lag 2 (%)",
    "inflation_lag2": "Inflation lag 2 (%)",
    "gdp_growth_lag2": "GDP growth lag 2 (%)",
    "life_expectancy_lag2": "Life expectancy lag 2",
    "population_growth_lag2": "Population growth lag 2 (%)",
    "unemployment_change_1y": "Unemployment change (1 year)",
    "inflation_change_1y": "Inflation change (1 year)",
    "gdp_growth_change_1y": "GDP growth change (1 year)",
}


def prettify_feature_name(name: str) -> str:
    return FEATURE_LABELS.get(name, name.replace("_", " ").title())


def show_missing_artifact_error(path: Path, label: str) -> None:
    st.error(f"{label} not found at\n\n`{path}`")
    st.stop()


st.title("Macroeconomic Downturn Risk Simulator")
st.write(
    "Interactive scenario tool built from the Part 3 country-year predictive model. "
    "Adjust present and lagged indicators to estimate next-year downturn probability."
)

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found at\n\n`{MODEL_PATH}`.\n\n"
        "Run `python train_downturn_model.py` first."
    )
    st.stop()

if not BASELINES_PATH.exists():
    st.error(
        f"Country baseline file not found at\n\n`{BASELINES_PATH}`.\n\n"
        "Run `python train_downturn_model.py` first."
    )
    st.stop()

if not FEATURES_PATH.exists():
    st.error(
        f"Model feature file not found at\n\n`{FEATURES_PATH}`.\n\n"
        "Run `python train_downturn_model.py` first."
    )
    st.stop()

model = load_model(MODEL_PATH)
country_baselines = load_country_baselines(BASELINES_PATH)
model_features = load_model_features(FEATURES_PATH)
cv_metrics = load_cv_metrics(CV_METRICS_PATH) if CV_METRICS_PATH.exists() else None

if country_baselines.empty:
    st.error("Country baselines file is empty.")
    st.stop()

country_options = sorted(country_baselines["country_code"].dropna().unique().tolist())

with st.sidebar:
    st.header("Scenario setup")

    selected_country = st.selectbox(
        "Select country baseline",
        options=country_options,
        index=0,
    )

    preset = st.selectbox(
        "Scenario preset",
        options=[
            "Custom",
            "Inflation shock",
            "Growth slowdown",
            "Labour market deterioration",
            "Broad downturn stress",
        ],
        index=0,
    )

country_row = country_baselines.loc[
    country_baselines["country_code"] == selected_country
].iloc[0].to_dict()

scenario_values = apply_scenario_preset(country_row.copy(), preset)

st.subheader(f"Country baseline: {selected_country}")

left_col, right_col = st.columns([1.1, 1.1])

editable_features = [
    "unemployment",
    "inflation",
    "gdp_growth",
    "life_expectancy",
    "population_growth",
    "unemployment_lag1",
    "inflation_lag1",
    "gdp_growth_lag1",
    "life_expectancy_lag1",
    "population_growth_lag1",
    "unemployment_lag2",
    "inflation_lag2",
    "gdp_growth_lag2",
    "life_expectancy_lag2",
    "population_growth_lag2",
    "unemployment_change_1y",
    "inflation_change_1y",
    "gdp_growth_change_1y",
]

with left_col:
    st.markdown("### Current and lagged inputs")
    for feature in editable_features:
        if feature not in scenario_values:
            continue

        default_value = float(scenario_values[feature])

        if "life_expectancy" in feature:
            min_value, max_value, step = 30.0, 90.0, 0.1
        elif "population_growth" in feature:
            min_value, max_value, step = -5.0, 10.0, 0.1
        elif "gdp_growth" in feature:
            min_value, max_value, step = -20.0, 20.0, 0.1
        elif "inflation" in feature:
            min_value, max_value, step = -20.0, 50.0, 0.1
        elif "unemployment" in feature:
            min_value, max_value, step = 0.0, 50.0, 0.1
        else:
            min_value, max_value, step = -20.0, 20.0, 0.1

        scenario_values[feature] = st.slider(
            INPUT_LABELS.get(feature, prettify_feature_name(feature)),
            min_value=min_value,
            max_value=max_value,
            value=float(default_value),
            step=step,
        )

model_input = build_model_input_row(
    scenario_values=scenario_values,
    model_features=model_features,
)

probability = float(model.predict_proba(model_input)[0, 1])
prediction = int(probability >= 0.5)

with right_col:
    st.markdown("### Predicted downturn risk")
    st.metric(
        label="Probability of downturn next year",
        value=f"{probability:.1%}",
    )

    if prediction == 1:
        st.error("Model classification: downturn risk")
    else:
        st.success("Model classification: no downturn risk")

    st.markdown("### Baseline vs scenario")

    comparison_rows = []
    for feature in editable_features:
        if feature not in country_row or feature not in scenario_values:
            continue

        baseline_val = float(country_row[feature])
        scenario_val = float(scenario_values[feature])
        delta_val = scenario_val - baseline_val

        comparison_rows.append(
            {
                "Feature": INPUT_LABELS.get(feature, prettify_feature_name(feature)),
                "Baseline": baseline_val,
                "Scenario": scenario_val,
                "Change": delta_val,
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

if cv_metrics is not None and not cv_metrics.empty:
    st.markdown("---")
    st.subheader("Cross-validation summary")
    pretty_metrics = cv_metrics.copy()
    if "model" in pretty_metrics.columns:
        st.dataframe(pretty_metrics, use_container_width=True, hide_index=True)
    else:
        st.dataframe(pretty_metrics, use_container_width=True)

if hasattr(model, "feature_importances_"):
    st.markdown("---")
    st.subheader("Top model feature importances")

    importance_df = pd.DataFrame(
        {
            "feature": model_features,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance_df["feature_display"] = importance_df["feature"].map(prettify_feature_name)

    top_n = min(10, len(importance_df))
    importance_plot_df = importance_df.head(top_n).sort_values("importance", ascending=True)

    fig = px.bar(
        importance_plot_df,
        x="importance",
        y="feature_display",
        orientation="h",
        template="plotly_dark",
        labels={
            "importance": "Importance",
            "feature_display": "Feature",
        },
        title="Top model feature importances",
    )

    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        title_x=0.0,
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption(
    "This dashboard uses the trained Part 3 classification model to simulate next-year "
    "downturn risk from current and lagged macroeconomic indicators."
)