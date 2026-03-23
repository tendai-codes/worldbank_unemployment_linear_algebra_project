from src.config import (
    START_YEAR,
    END_YEAR,
    INDICATORS,
    SAVE_CSV_SNAPSHOTS,
)
from src.api_client import fetch_indicator_data, fetch_country_metadata
from src.data_processing import (
    raw_to_dataframe,
    country_metadata_to_dataframe,
    merge_country_metadata,
    remove_aggregate_entities,
    build_country_year_matrix,
    build_panel_dataset,
    build_average_feature_matrix,
)
from src.matrix_analysis import (
    inspect_missingness,
    compute_rank,
    compute_ref_rref,
    compute_column_correlation,
    compute_country_similarity,
    run_pca,
)

def save_if_needed(df, path):
    if SAVE_CSV_SNAPSHOTS:
        df.to_csv(path, index=True if df.index.name is not None else False)


def run_unemployment_analysis():
    print("\n=== PART 1: UNEMPLOYMENT ANALYSIS ===")

    print("\nStep 1: Fetch unemployment data")
    raw_unemployment = fetch_indicator_data(
        INDICATORS["unemployment"],
        START_YEAR,
        END_YEAR
    )
    print("Verification: raw rows fetched =", len(raw_unemployment))

    raw_country_metadata = fetch_country_metadata()
    country_metadata_df = country_metadata_to_dataframe(raw_country_metadata)

    print("\nStep 2: Convert raw rows to DataFrame")
    df_unemployment = raw_to_dataframe(raw_unemployment, "unemployment_rate")
    df_unemployment = merge_country_metadata(df_unemployment, country_metadata_df)
    df_unemployment = remove_aggregate_entities(df_unemployment)
    df_unemployment = df_unemployment.sort_values(["country_code", "year"]).reset_index(drop=True)

    print(df_unemployment.head())
    print("Verification: dataframe shape =", df_unemployment.shape)

    save_if_needed(df_unemployment, "data/unemployment.csv")

    print("\nStep 3: Build country-year unemployment matrix")
    unemployment_matrix = build_country_year_matrix(df_unemployment, "unemployment_rate")
    print(unemployment_matrix.head())
    print("Verification: matrix shape =", unemployment_matrix.shape)

    save_if_needed(unemployment_matrix, "data/unemployment_matrix.csv")

    print("\nStep 4: Inspect missingness")
    missing_by_country, missing_by_year = inspect_missingness(unemployment_matrix)
    print("Top countries with most missing values:")
    print(missing_by_country.head(10))
    print("\nMissing values by year:")
    print(missing_by_year)

    print("\nStep 5: Compute rank")
    rank_value = compute_rank(unemployment_matrix)
    print("Verification: matrix rank =", rank_value)

    print("\nStep 6: Compute REF and RREF on a subset")
    ref_rref = compute_ref_rref(unemployment_matrix)
    print("Subset used:")
    print(ref_rref["subset"])
    print("\nREF pivots:", ref_rref["ref_pivots"])
    print("RREF pivots:", ref_rref["rref_pivots"])

    print("\nStep 7: Correlation between years")
    year_corr = compute_column_correlation(unemployment_matrix)
    print(year_corr.round(3).iloc[:5, :5])

    print("\nStep 8: Country similarity")
    country_similarity = compute_country_similarity(unemployment_matrix)
    print(country_similarity.iloc[:5, :5].round(3))

    print("\nStep 9: PCA on unemployment trajectories")
    unemployment_pca = run_pca(unemployment_matrix, n_components=2)
    print("Verification: explained variance ratio =",
          unemployment_pca["explained_variance_ratio"])
    print(unemployment_pca["pca_scores"].head())

    return {
        "df_unemployment": df_unemployment,
        "unemployment_matrix": unemployment_matrix,
        "missing_by_country": missing_by_country,
        "missing_by_year": missing_by_year,
        "rank": rank_value,
        "ref_rref": ref_rref,
        "year_correlation": year_corr,
        "country_similarity": country_similarity,
        "pca_results": unemployment_pca,
    }

def run_multi_indicator_analysis():
    print("\n=== PART 2: MULTI-INDICATOR ANALYSIS ===")

    raw_country_metadata = fetch_country_metadata()
    country_metadata_df = country_metadata_to_dataframe(raw_country_metadata)

    indicator_dataframes = {}

    for feature_name, indicator_code in INDICATORS.items():
        try:
            print(f"\nStep 1: Fetch {feature_name}")
            raw_rows = fetch_indicator_data(indicator_code, START_YEAR, END_YEAR)
            print("Verification: raw rows fetched =", len(raw_rows))

            df_feature = raw_to_dataframe(raw_rows, feature_name)
            df_feature = merge_country_metadata(df_feature, country_metadata_df)
            df_feature = remove_aggregate_entities(df_feature)
            df_feature = df_feature.sort_values(["country_code", "year"]).reset_index(drop=True)

            print(df_feature.head())
            print("Verification: dataframe shape =", df_feature.shape)

            indicator_dataframes[feature_name] = df_feature
            save_if_needed(df_feature, f"data/{feature_name}.csv")

        except Exception as e:
            print(f"Failed for {feature_name}: {e}")

    print("\nIndicators successfully loaded:")
    print(indicator_dataframes.keys())

    print("\nStep 2: Build merged panel dataset")
    panel_df = build_panel_dataset(indicator_dataframes)
    print(panel_df.head())
    print("Verification: panel shape =", panel_df.shape)
    print("\nMissing values by column:")
    print(panel_df.isna().sum())

    save_if_needed(panel_df, "data/worldbank_panel.csv")

    print("\nStep 3: Build average country-feature matrix")
    feature_columns = [col for col in INDICATORS.keys() if col in panel_df.columns]
    feature_matrix = build_average_feature_matrix(panel_df, feature_columns)

    print(feature_matrix.head())
    print("Verification: feature matrix shape =", feature_matrix.shape)
    print("Verification: total missing values =", feature_matrix.isna().sum().sum())

    save_if_needed(feature_matrix, "data/feature_matrix.csv")

    print("\nStep 4: Compute rank")
    feature_rank = compute_rank(feature_matrix)
    print("Verification: feature matrix rank =", feature_rank)

    print("\nStep 5: Correlation between features")
    feature_corr = compute_column_correlation(feature_matrix)
    print(feature_corr.round(3))

    print("\nStep 6: Country similarity in feature space")
    feature_similarity = compute_country_similarity(feature_matrix)
    print(feature_similarity.iloc[:5, :5].round(3))

    print("\nStep 7: PCA on country-feature matrix")
    feature_pca = run_pca(feature_matrix, n_components=2)
    print("Verification: explained variance ratio =",
          feature_pca["explained_variance_ratio"])
    print(feature_pca["pca_scores"].head())

    return {
        "panel_df": panel_df,
        "feature_matrix": feature_matrix,
        "rank": feature_rank,
        "feature_correlation": feature_corr,
        "country_similarity": feature_similarity,
        "pca_results": feature_pca,
    }

def run_pipeline():
    unemployment_results = run_unemployment_analysis()
    multi_indicator_results = run_multi_indicator_analysis()

    print("\n=== PIPELINE COMPLETE ===")
    print("Unemployment matrix shape:", unemployment_results["unemployment_matrix"].shape)
    print("Feature matrix shape:", multi_indicator_results["feature_matrix"].shape)
