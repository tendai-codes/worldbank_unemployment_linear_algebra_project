import pandas as pd
from src.config import AGGREGATE_REGION_LABEL


def raw_to_dataframe(rows, value_name):
    """
    Convert raw indicator rows into a tidy DataFrame.
    """
    cleaned_rows = []

    for row in rows:
        country_code = row.get("countryiso3code")

        if not country_code or len(country_code) != 3:
            continue

        cleaned_rows.append({
            "country": row.get("country", {}).get("value"),
            "country_code": country_code,
            "year": int(row["date"]),
            value_name: row.get("value"),
            "indicator_code": row.get("indicator", {}).get("id"),
            "indicator_name": row.get("indicator", {}).get("value"),
        })

    return pd.DataFrame(cleaned_rows)


def country_metadata_to_dataframe(rows):
    """
    Convert raw country metadata rows into a DataFrame.
    """
    cleaned_rows = []

    for row in rows:
        country_code = row.get("id")

        if not country_code or len(country_code) != 3:
            continue

        cleaned_rows.append({
            "country_code": country_code,
            "wb_country_name": row.get("name"),
            "region": row.get("region", {}).get("value"),
            "income_level": row.get("incomeLevel", {}).get("value"),
            "lending_type": row.get("lendingType", {}).get("value"),
        })

    return pd.DataFrame(cleaned_rows)


def merge_country_metadata(indicator_df, metadata_df):
    """
    Merge country metadata into the indicator DataFrame.
    """
    return indicator_df.merge(metadata_df, on="country_code", how="left")


def remove_aggregate_entities(df):
    """
    Remove aggregate rows using World Bank region metadata.
    """
    aggregate_mask = df["region"] == AGGREGATE_REGION_LABEL
    removed_entities = sorted(df.loc[aggregate_mask, "country"].dropna().unique())
    filtered_df = df.loc[~aggregate_mask].reset_index(drop=True)

    if removed_entities:
        print(
            f"Removed {aggregate_mask.sum()} aggregate rows: "
            f"{', '.join(removed_entities[:20])}"
        )
        if len(removed_entities) > 20:
            print(f"... and {len(removed_entities) - 20} more.")

    return filtered_df


def build_country_year_matrix(df, value_column):
    matrix = df.pivot(
        index="country_code",
        columns="year",
        values=value_column
    ).sort_index()

    return matrix


def build_panel_dataset(indicator_dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all indicator tables into one country-year panel.

    Merge on country_code and year to avoid duplicate rows caused by slight naming
    differences. Country names are reattached afterwards from the first available
    non-null name.
    """
    if not indicator_dataframes:
        raise ValueError("indicator_dataframes is empty")

    panel = None
    country_lookup_frames = []

    for feature_name, df in indicator_dataframes.items():
        temp = df[["country_code", "year", feature_name]].copy()
        panel = temp if panel is None else panel.merge(temp, on=["country_code", "year"], how="outer")

        if "country" in df.columns:
            country_lookup_frames.append(
                df[["country_code", "country"]].dropna().drop_duplicates(subset=["country_code"])
            )

    if country_lookup_frames:
        country_lookup = (
            pd.concat(country_lookup_frames, ignore_index=True)
            .drop_duplicates(subset=["country_code"], keep="first")
        )
        panel = panel.merge(country_lookup, on="country_code", how="left")
        panel = panel[["country", "country_code", "year"] + [c for c in panel.columns if c not in {"country", "country_code", "year"}]]

    return panel.sort_values(["country_code", "year"]).reset_index(drop=True)


def summarise_country_feature_missingness(panel_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Count missing years per country and feature."""
    missing_counts = (
        panel_df.groupby(["country", "country_code"], dropna=False)[feature_columns]
        .apply(lambda g: g.isna().sum())
        .reset_index()
    )
    return missing_counts


def filter_countries_with_complete_indicator_gaps(panel_df: pd.DataFrame, feature_columns: list[str]):
    """
    Remove countries that are completely missing at least one whole indicator series.

    This is the correct rule for the multi-indicator matrix. We remove a country only
    when an indicator is missing for *all* years, not when it merely has a few gaps.
    """
    n_years = panel_df["year"].nunique()
    missing_counts = summarise_country_feature_missingness(panel_df, feature_columns)

    detail = missing_counts.copy()
    detail["fully_missing_indicators"] = detail[feature_columns].apply(
        lambda row: [feature for feature in feature_columns if row[feature] == n_years],
        axis=1,
    )
    detail["remove_country"] = detail["fully_missing_indicators"].apply(bool)

    countries_to_remove = detail.loc[detail["remove_country"], ["country", "country_code", "fully_missing_indicators"]].reset_index(drop=True)

    if countries_to_remove.empty:
        filtered = panel_df.copy()
    else:
        filtered = panel_df.merge(
            countries_to_remove[["country_code"]].assign(remove_flag=1),
            on="country_code",
            how="left",
        )
        filtered = (
            filtered.loc[filtered["remove_flag"].isna()]
            .drop(columns=["remove_flag"])
            .reset_index(drop=True)
        )

    return filtered, countries_to_remove, detail


def interpolate_panel_by_country(panel_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """
    Interpolate within each country over time for each feature.

    This only fills internal gaps. Leading or trailing missing values remain missing,
    which avoids inventing data too aggressively.
    """
    result = panel_df.sort_values(["country_code", "year"]).copy()

    for feature in feature_columns:
        result[feature] = (
            result.groupby("country_code")[feature]
            .transform(lambda s: s.interpolate(method="linear", limit_area="inside"))
        )

    return result


def build_average_feature_matrix(panel_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    feature_matrix = panel_df.groupby("country_code")[feature_columns].mean()
    feature_matrix = feature_matrix.dropna().sort_index()
    return feature_matrix
