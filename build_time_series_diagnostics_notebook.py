from pathlib import Path
import nbformat as nbf


def make_markdown_cell(text: str):
    return nbf.v4.new_markdown_cell(text)


def make_code_cell(code: str):
    return nbf.v4.new_code_cell(code)


def build_cells():
    cells = []

    cells.append(make_markdown_cell(
        """# 03 Time Series Feature Diagnostics

This notebook inspects the time-series behaviour of the macroeconomic indicators
used in the downturn-risk project.

It visualises:

- raw indicator levels
- year-to-year changes
- 3-year least-squares slopes
- downturn-next-year overlays
"""
    ))

    cells.append(make_markdown_cell(
        """## Expected file location

This notebook expects one of these files:

- `data/worldbank_panel_final.csv`
- `data/worldbank_panel.csv`
"""
    ))

    cells.append(make_code_cell(
        """from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

current = Path.cwd().resolve()
project_root = current.parent if current.name == "notebooks" else current

panel_final_path = project_root / "data" / "worldbank_panel_final.csv"
panel_fallback_path = project_root / "data" / "worldbank_panel.csv"

if panel_final_path.exists():
    panel_path = panel_final_path
elif panel_fallback_path.exists():
    panel_path = panel_fallback_path
else:
    raise FileNotFoundError(
        "Could not find input panel CSV. Expected one of:\\n"
        f"- {panel_final_path}\\n"
        f"- {panel_fallback_path}"
    )

print("Using panel file:", panel_path)
"""
    ))

    cells.append(make_code_cell(
        """panel_df = pd.read_csv(panel_path)
panel_df = panel_df.sort_values(["country_code", "year"]).reset_index(drop=True)

required_cols = [
    "country_code",
    "year",
    "unemployment",
    "inflation",
    "gdp_growth",
    "life_expectancy",
    "population_growth",
]

missing_cols = [c for c in required_cols if c not in panel_df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

if "country" not in panel_df.columns:
    panel_df["country"] = panel_df["country_code"]

panel_df.head()
"""
    ))

    cells.append(make_markdown_cell(
        """## Construct delta and trend features

Annual change:

\\[
\\Delta x_t = x_t - x_{t-1}
\\]

3-year trend using least-squares slope:

\\[
\\hat{\\beta} = \\arg\\min_{\\beta} ||X\\beta - y||
\\]
"""
    ))

    cells.append(make_code_cell(
        """def slope_3(values):
    x = np.array([0.0, 1.0, 2.0], dtype=float)
    y = np.asarray(values, dtype=float)
    return float(np.polyfit(x, y, 1)[0])

features = [
    "unemployment",
    "inflation",
    "gdp_growth",
    "life_expectancy",
    "population_growth",
]

ts_df = panel_df.copy()

for col in features:
    ts_df[f"{col}_change_1y"] = ts_df.groupby("country_code")[col].diff()
    ts_df[f"{col}_trend_3y"] = (
        ts_df.groupby("country_code")[col]
        .transform(lambda s: s.rolling(3).apply(lambda x: slope_3(np.array(x)), raw=False))
    )

ts_df["gdp_growth_next_year"] = ts_df.groupby("country_code")["gdp_growth"].shift(-1)
ts_df["downturn_next_year"] = (ts_df["gdp_growth_next_year"] < 0).astype(int)

ts_df.head(10)
"""
    ))

    cells.append(make_code_cell(
        """country_lookup = (
    ts_df[["country", "country_code"]]
    .drop_duplicates()
    .sort_values("country")
    .reset_index(drop=True)
)

country_lookup.head()
"""
    ))

    cells.append(make_markdown_cell(
        """## Select a country for diagnostics"""
    ))

    cells.append(make_code_cell(
        """selected_country = "South Africa"

country_row = country_lookup.loc[country_lookup["country"] == selected_country]
if country_row.empty:
    raise ValueError(
        f"Country '{selected_country}' not found. "
        "Choose a value from country_lookup['country']."
    )

selected_country_code = country_row["country_code"].iloc[0]
country_ts = ts_df.loc[ts_df["country_code"] == selected_country_code].copy()

print("Selected country:", selected_country)
print("Country code:", selected_country_code)
country_ts.head()
"""
    ))

    cells.append(make_code_cell(
        """def downturn_marker_years(df):
    return df.loc[df["downturn_next_year"] == 1, "year"].tolist()


def add_downturn_markers(fig, df, y_col, name="Downturn next year"):
    downturn_years = downturn_marker_years(df)
    if downturn_years:
        marker_df = df.loc[df["year"].isin(downturn_years), ["year", y_col]].dropna()
        if not marker_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=marker_df["year"],
                    y=marker_df[y_col],
                    mode="markers",
                    name=name,
                    marker=dict(size=10, symbol="diamond"),
                )
            )
    return fig


def plot_series(df, y_col, title, y_label):
    fig = px.line(df, x="year", y=y_col, markers=True, title=title)
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=y_label,
        template="plotly_white",
        height=450,
    )
    fig = add_downturn_markers(fig, df, y_col)
    return fig


def plot_feature_triptych(df, feature, country_name):
    fig_level = plot_series(
        df,
        feature,
        f"{country_name}: {feature.replace('_', ' ').title()} level",
        feature.replace("_", " ").title(),
    )
    fig_delta = plot_series(
        df,
        f"{feature}_change_1y",
        f"{country_name}: {feature.replace('_', ' ').title()} annual change",
        f"Δ {feature.replace('_', ' ').title()}",
    )
    fig_trend = plot_series(
        df,
        f"{feature}_trend_3y",
        f"{country_name}: {feature.replace('_', ' ').title()} 3-year trend",
        f"{feature.replace('_', ' ').title()} trend",
    )
    return fig_level, fig_delta, fig_trend
"""
    ))

    cells.append(make_code_cell(
        """feature_to_plot = "gdp_growth"

fig_level, fig_delta, fig_trend = plot_feature_triptych(
    country_ts,
    feature_to_plot,
    selected_country,
)

fig_level.show()
fig_delta.show()
fig_trend.show()
"""
    ))

    cells.append(make_code_cell(
        """for feature in features:
    fig_level, fig_delta, fig_trend = plot_feature_triptych(
        country_ts,
        feature,
        selected_country,
    )
    fig_level.show()
    fig_delta.show()
    fig_trend.show()
"""
    ))

    cells.append(make_code_cell(
        """latest_summary = country_ts.sort_values("year").tail(1).copy()

summary_rows = []
for feature in features:
    summary_rows.append(
        {
            "Feature": feature.replace("_", " ").title(),
            "Current value": latest_summary[feature].iloc[0],
            "Annual change": latest_summary[f"{feature}_change_1y"].iloc[0],
            "3-year trend": latest_summary[f"{feature}_trend_3y"].iloc[0],
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_df
"""
    ))

    cells.append(make_code_cell(
        """comparison_country = "Brazil"
comparison_row = country_lookup.loc[country_lookup["country"] == comparison_country]

if comparison_row.empty:
    raise ValueError(
        f"Country '{comparison_country}' not found. "
        "Choose a value from country_lookup['country']."
    )

comparison_code = comparison_row["country_code"].iloc[0]
comparison_ts = ts_df.loc[ts_df["country_code"] == comparison_code].copy()

comparison_feature = "gdp_growth"

plot_compare_df = pd.concat(
    [
        country_ts[["year", comparison_feature]].assign(country=selected_country),
        comparison_ts[["year", comparison_feature]].assign(country=comparison_country),
    ],
    ignore_index=True,
)

fig = px.line(
    plot_compare_df,
    x="year",
    y=comparison_feature,
    color="country",
    markers=True,
    title=f"{comparison_feature.replace('_', ' ').title()} comparison: {selected_country} vs {comparison_country}",
)
fig.update_layout(template="plotly_white", height=450)
fig.show()
"""
    ))

    cells.append(make_code_cell(
        """aggregate_feature = "gdp_growth_change_1y"

aggregate_plot_df = (
    ts_df.groupby(["year", "downturn_next_year"], as_index=False)[aggregate_feature]
    .mean()
)

aggregate_plot_df["downturn_label"] = aggregate_plot_df["downturn_next_year"].map(
    {0: "No downturn next year", 1: "Downturn next year"}
)

fig = px.line(
    aggregate_plot_df,
    x="year",
    y=aggregate_feature,
    color="downturn_label",
    markers=True,
    title=f"Average {aggregate_feature.replace('_', ' ')} by year and downturn regime",
)
fig.update_layout(template="plotly_white", height=450)
fig.show()
"""
    ))

    cells.append(make_markdown_cell(
        """## Interpretation prompts

### Levels
- Does the country enter downturn periods at unusually low GDP growth levels?
- Are unemployment and inflation levels structurally high before downturn years?

### Deltas
- Are downturns preceded by sudden negative changes in GDP growth?
- Do unemployment and inflation changes become unstable before downturns?

### Trends
- Are downturns associated with sustained negative 3-year GDP slopes?
- Does unemployment trend upward before downturn years?

### Comparison
- Do similar countries share the same level, delta, or trend patterns?
- Are trend features more informative than raw levels for early warning?
"""
    ))

    cells.append(make_markdown_cell(
        """## Suggested conclusion template

You can adapt the following:

> The time-series diagnostics suggest that downturn periods are better characterised by **deteriorating direction and sustained trends** than by levels alone. In particular, negative GDP-growth slopes and worsening annual changes appear before several downturn episodes, supporting the inclusion of delta and 3-year trend features in the predictive model.

This links the visual evidence back to your modelling choices.
"""
    ))

    return cells


def build_notebook():
    nb = nbf.v4.new_notebook()
    nb["cells"] = build_cells()
    return nb


def save_notebook(nb, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)


def main():
    nb = build_notebook()

    output_path = Path("03_time_series_feature_diagnostics.ipynb")
    save_notebook(nb, output_path)

    print(f"Created notebook at: {output_path.resolve()}")


if __name__ == "__main__":
    main()