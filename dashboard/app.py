from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.model_utils import (
    load_model,
    load_country_baselines,
    load_model_features,
    load_feature_matrix,
    load_optimal_threshold,
    load_panel_data,
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
FEATURE_MATRIX_PATH = PROJECT_ROOT / "data" / "feature_matrix.csv"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "optimal_threshold.csv"
PANEL_PATH = PROJECT_ROOT / "data" / "worldbank_panel_final.csv"

FEATURE_LABELS = {
    "gdp_growth": "GDP growth",
    "gdp_growth_change_1y": "GDP growth (1-year change)",
    "gdp_growth_lag1": "GDP growth (lag 1 year)",
    "gdp_growth_lag2": "GDP growth (lag 2 years)",
    "gdp_growth_trend_3y": "GDP growth trend (3 years)",
    "inflation": "Inflation",
    "inflation_change_1y": "Inflation (1-year change)",
    "inflation_lag1": "Inflation (lag 1 year)",
    "inflation_lag2": "Inflation (lag 2 years)",
    "inflation_trend_3y": "Inflation trend (3 years)",
    "unemployment": "Unemployment",
    "unemployment_change_1y": "Unemployment (1-year change)",
    "unemployment_lag1": "Unemployment (lag 1 year)",
    "unemployment_lag2": "Unemployment (lag 2 years)",
    "unemployment_trend_3y": "Unemployment trend (3 years)",
    "life_expectancy": "Life expectancy",
    "life_expectancy_change_1y": "Life expectancy (1-year change)",
    "life_expectancy_lag1": "Life expectancy (lag 1 year)",
    "life_expectancy_lag2": "Life expectancy (lag 2 years)",
    "life_expectancy_trend_3y": "Life expectancy trend (3 years)",
    "population_growth": "Population growth",
    "population_growth_change_1y": "Population growth (1-year change)",
    "population_growth_lag1": "Population growth (lag 1 year)",
    "population_growth_lag2": "Population growth (lag 2 years)",
    "population_growth_trend_3y": "Population growth trend (3 years)",
}

INPUT_LABELS = {
    "unemployment": "Unemployment (%)",
    "inflation": "Inflation (%)",
    "gdp_growth": "GDP growth (%)",
    "life_expectancy": "Life expectancy",
    "population_growth": "Population growth (%)",
    "unemployment_change_1y": "Unemployment change (1 year)",
    "inflation_change_1y": "Inflation change (1 year)",
    "gdp_growth_change_1y": "GDP growth change (1 year)",
    "life_expectancy_change_1y": "Life expectancy change (1 year)",
    "population_growth_change_1y": "Population growth change (1 year)",
    "unemployment_trend_3y": "Unemployment trend (3 years)",
    "inflation_trend_3y": "Inflation trend (3 years)",
    "gdp_growth_trend_3y": "GDP growth trend (3 years)",
    "life_expectancy_trend_3y": "Life expectancy trend (3 years)",
    "population_growth_trend_3y": "Population growth trend (3 years)",
}


def prettify_feature_name(name: str) -> str:
    return FEATURE_LABELS.get(name, name.replace("_", " ").title())


def feature_slider_bounds(feature: str) -> tuple[float, float, float]:
    if "life_expectancy" in feature:
        if "change" in feature or "trend" in feature:
            return -10.0, 10.0, 0.1
        return 30.0, 90.0, 0.1

    if "population_growth" in feature:
        return -5.0, 10.0, 0.1

    if "gdp_growth" in feature:
        return -20.0, 20.0, 0.1

    if "inflation" in feature:
        return -20.0, 50.0, 0.1

    if "unemployment" in feature:
        if "change" in feature or "trend" in feature:
            return -10.0, 10.0, 0.1
        return 0.0, 50.0, 0.1

    return -20.0, 20.0, 0.1


def find_similar_countries(
    selected_country_code: str,
    feature_matrix: pd.DataFrame,
    baselines_df: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    if selected_country_code not in feature_matrix.index:
        return pd.DataFrame(columns=["Country", "Country code", "Distance"])

    target_vector = feature_matrix.loc[selected_country_code].values

    distances = (
        feature_matrix.drop(index=selected_country_code)
        .apply(lambda row: np.linalg.norm(row.values - target_vector), axis=1)
        .sort_values()
        .head(top_n)
    )

    code_to_name = (
        baselines_df[["country_code", "country"]]
        .drop_duplicates()
        .set_index("country_code")["country"]
        .to_dict()
    )

    return pd.DataFrame(
        {
            "Country": [code_to_name.get(code, code) for code in distances.index],
            "Country code": distances.index,
            "Distance": distances.values.round(4),
        }
    )


def plot_multi_country_timeseries(
    panel_df: pd.DataFrame,
    country_names: list[str],
    feature: str,
    title: str | None = None,
):
    plot_df = panel_df.loc[panel_df["country"].isin(country_names)].copy()

    fig = px.line(
        plot_df,
        x="year",
        y=feature,
        color="country",
        markers=True,
        title=title or f"{prettify_feature_name(feature)} comparison across countries",
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Year",
        yaxis_title=prettify_feature_name(feature),
        height=500,
        legend_title_text="Country",
        title_x=0.0,
    )

    return fig


st.title("Macroeconomic Downturn Risk Simulator")
st.write(
    "Choose a country, adjust macroeconomic conditions, and estimate the model's "
    "predicted probability of a downturn next year."
)

for path, label in [
    (MODEL_PATH, "Model file"),
    (BASELINES_PATH, "Country baseline file"),
    (FEATURES_PATH, "Model feature file"),
    (FEATURE_MATRIX_PATH, "Feature matrix file"),
    (PANEL_PATH, "Panel data file"),
]:
    if not path.exists():
        st.error(f"{label} not found at `{path}`.")
        st.stop()

model = load_model(MODEL_PATH)
country_baselines = load_country_baselines(BASELINES_PATH)
model_features = load_model_features(FEATURES_PATH)
feature_matrix = load_feature_matrix(FEATURE_MATRIX_PATH)
classification_threshold = load_optimal_threshold(THRESHOLD_PATH, default=0.35)
panel_df = load_panel_data(PANEL_PATH)

if "country" not in panel_df.columns:
    panel_df["country"] = panel_df["country_code"]

if country_baselines.empty:
    st.error("Country baselines file is empty.")
    st.stop()

country_lookup = (
    country_baselines[["country", "country_code"]]
    .drop_duplicates()
    .sort_values("country")
    .reset_index(drop=True)
)

country_name_options = country_lookup["country"].tolist()

selector_col, preset_col = st.columns([1.6, 1.0])

with selector_col:
    selected_country_name = st.selectbox(
        "Select country",
        options=country_name_options,
        index=0,
    )

with preset_col:
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

selected_country_code = country_lookup.loc[
    country_lookup["country"] == selected_country_name,
    "country_code"
].iloc[0]

country_row = country_baselines.loc[
    country_baselines["country_code"] == selected_country_code
].iloc[0].to_dict()

scenario_values = apply_scenario_preset(country_row.copy(), preset)

st.subheader(f"Country baseline: {selected_country_name}")

left_col, right_col = st.columns([1.1, 1.1])

priority_features = [
    "unemployment",
    "inflation",
    "gdp_growth",
    "life_expectancy",
    "population_growth",
    "unemployment_change_1y",
    "inflation_change_1y",
    "gdp_growth_change_1y",
    "unemployment_trend_3y",
    "inflation_trend_3y",
    "gdp_growth_trend_3y",
]

editable_features = [
    feature for feature in priority_features
    if feature in model_features and feature in scenario_values
]

with left_col:
    st.markdown("### Adjust macroeconomic inputs")

    for feature in editable_features:
        default_value = float(scenario_values[feature])
        min_value, max_value, step = feature_slider_bounds(feature)

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
prediction = int(probability >= classification_threshold)

with right_col:
    st.markdown("### Predicted downturn risk")
    st.metric(
        label="Probability of downturn next year",
        value=f"{probability:.1%}",
    )

    st.caption(f"Classification threshold: {classification_threshold:.2f}")

    if prediction == 1:
        st.error("Model classification: downturn risk")
    else:
        st.success("Model classification: no downturn risk")

    st.markdown("### Baseline vs scenario")

    comparison_rows = []
    for feature in editable_features:
        baseline_val = float(country_row.get(feature, 0.0))
        scenario_val = float(scenario_values.get(feature, 0.0))
        delta_val = scenario_val - baseline_val

        comparison_rows.append(
            {
                "Feature": INPUT_LABELS.get(feature, prettify_feature_name(feature)),
                "Baseline": round(baseline_val, 3),
                "Scenario": round(scenario_val, 3),
                "Change": round(delta_val, 3),
            }
        )

    st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)

st.markdown("---")

similar_df = find_similar_countries(
    selected_country_code=selected_country_code,
    feature_matrix=feature_matrix,
    baselines_df=country_baselines,
    top_n=5,
)

st.subheader("Countries with similar macroeconomic profiles")

if similar_df.empty:
    st.info("No similar-country comparison is available for this selection.")
else:
    st.dataframe(similar_df, width="stretch", hide_index=True)

st.markdown("---")
st.subheader("Multi-country time series comparison")

timeseries_feature_groups = {
    "Levels": {
        "GDP growth": "gdp_growth",
        "Inflation": "inflation",
        "Unemployment": "unemployment",
        "Life expectancy": "life_expectancy",
        "Population growth": "population_growth",
    },
    "Annual changes": {
        "GDP growth (1-year change)": "gdp_growth_change_1y",
        "Inflation (1-year change)": "inflation_change_1y",
        "Unemployment (1-year change)": "unemployment_change_1y",
        "Life expectancy (1-year change)": "life_expectancy_change_1y",
        "Population growth (1-year change)": "population_growth_change_1y",
    },
    "3-year trends": {
        "GDP growth trend (3 years)": "gdp_growth_trend_3y",
        "Inflation trend (3 years)": "inflation_trend_3y",
        "Unemployment trend (3 years)": "unemployment_trend_3y",
        "Life expectancy trend (3 years)": "life_expectancy_trend_3y",
        "Population growth trend (3 years)": "population_growth_trend_3y",
    },
}

group_col, feature_col = st.columns([1, 1])

with group_col:
    selected_feature_group = st.selectbox(
        "Select feature group",
        options=list(timeseries_feature_groups.keys()),
        index=0,
    )

feature_options = timeseries_feature_groups[selected_feature_group]

with feature_col:
    selected_timeseries_label = st.selectbox(
        "Select feature",
        options=list(feature_options.keys()),
        index=0,
    )

selected_timeseries_feature = feature_options[selected_timeseries_label]

default_country_set = [selected_country_name]
if not similar_df.empty:
    for similar_country in similar_df["Country"].tolist():
        if similar_country != selected_country_name and len(default_country_set) < 4:
            default_country_set.append(similar_country)

country_selection_options = sorted(panel_df["country"].dropna().unique().tolist())

selected_timeseries_countries = st.multiselect(
    "Select up to 4 countries",
    options=country_selection_options,
    default=default_country_set,
    max_selections=4,
)

if selected_timeseries_countries:
    fig = plot_multi_country_timeseries(
        panel_df=panel_df,
        country_names=selected_timeseries_countries,
        feature=selected_timeseries_feature,
        title=f"{selected_timeseries_label} comparison across selected countries",
    )
    st.plotly_chart(fig, width="stretch")
else:
    st.info("Select at least one country to display the time series chart.")

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
        labels={"importance": "Importance", "feature_display": "Feature"},
        title="Top model feature importances",
    )

    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        title_x=0.0,
    )

    st.plotly_chart(fig, width="stretch")

st.markdown("---")
st.caption(
    "This dashboard uses the trained Part 3 classification model to estimate "
    "next-year downturn risk from current levels, annual changes, and 3-year macroeconomic trends."
)