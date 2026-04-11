from __future__ import annotations

import pandas as pd


def apply_scenario_preset(country_row: dict, preset: str) -> dict:
    """
    Apply a simple scenario preset to the selected country baseline.
    Returns a modified copy of the input dictionary.
    """
    values = country_row.copy()

    if preset == "Custom":
        return values

    if preset == "Inflation shock":
        values["inflation"] = float(values.get("inflation", 0.0)) + 3.0
        values["inflation_change_1y"] = float(values.get("inflation_change_1y", 0.0)) + 2.0

    elif preset == "Growth slowdown":
        values["gdp_growth"] = float(values.get("gdp_growth", 0.0)) - 2.5
        values["gdp_growth_change_1y"] = float(values.get("gdp_growth_change_1y", 0.0)) - 2.0

    elif preset == "Labour market deterioration":
        values["unemployment"] = float(values.get("unemployment", 0.0)) + 2.0
        values["unemployment_change_1y"] = float(values.get("unemployment_change_1y", 0.0)) + 1.5

    elif preset == "Broad downturn stress":
        values["gdp_growth"] = float(values.get("gdp_growth", 0.0)) - 3.0
        values["gdp_growth_change_1y"] = float(values.get("gdp_growth_change_1y", 0.0)) - 2.5
        values["inflation"] = float(values.get("inflation", 0.0)) + 2.5
        values["inflation_change_1y"] = float(values.get("inflation_change_1y", 0.0)) + 1.5
        values["unemployment"] = float(values.get("unemployment", 0.0)) + 2.5
        values["unemployment_change_1y"] = float(values.get("unemployment_change_1y", 0.0)) + 1.5

    return values


def build_model_input_row(scenario_values: dict, model_features: list[str]) -> pd.DataFrame:
    """
    Build a one-row dataframe in the exact feature order expected by the trained model.
    Missing features default to 0.0.
    """
    row = {}
    for feature in model_features:
        row[feature] = float(scenario_values.get(feature, 0.0))
    return pd.DataFrame([row], columns=model_features)