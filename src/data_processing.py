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


def build_panel_dataset(indicator_dataframes):
    panel = None

    for feature_name, df in indicator_dataframes.items():
        temp = df[["country", "country_code", "year", feature_name]].copy()

        if panel is None:
            panel = temp
        else:
            panel = panel.merge(
                temp,
                on=["country", "country_code", "year"],
                how="outer"
            )

    return panel


def build_average_feature_matrix(panel_df, feature_columns):
    feature_matrix = (
        panel_df.groupby("country_code")[feature_columns]
        .mean()
        .dropna()
    )

    return feature_matrix