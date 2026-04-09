import pandas as pd
import numpy as np
from sympy import Matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

def inspect_missingness(matrix_df):
    """
    Count missing values by row and by column.

    Why this is done:
    Missing values affect rank, PCA, correlation, and similarity.

    Hoped-for outcome:
    A clear view of where the matrix is incomplete.
    """
    missing_by_row = matrix_df.isna().sum(axis=1).sort_values(ascending=False)
    missing_by_col = matrix_df.isna().sum(axis=0).sort_values(ascending=False)
    return missing_by_row, missing_by_col


def compute_rank(matrix_df):
    """
    Compute the numeric rank of a matrix.

    Why this is done:
    Rank tells us how many independent directions exist in the data.

    Hoped-for outcome:
    A single integer describing the matrix dimensionality.
    """
    clean_df = matrix_df.dropna(axis=0).dropna(axis=1)
    if clean_df.empty:
        raise ValueError("Matrix is empty after dropping missing values; rank cannot be computed.")
    return np.linalg.matrix_rank(clean_df.values)


def compute_ref_rref(matrix_df, row_limit=10, col_limit=8, rounding=3):
    """
    Compute REF and RREF on a smaller rounded subset.

    Why this is done:
    REF and RREF are mainly explanatory tools here. On large floating-point
    economic matrices, exact symbolic row reduction becomes messy.

    Hoped-for outcome:
    A manageable demonstration of pivots and row reduction.
    """
    subset = matrix_df.dropna(axis=0).iloc[:row_limit, :col_limit].copy()
    if subset.empty:
        raise ValueError("Subset for REF/RREF is empty after dropping missing rows.")
    subset = subset.round(rounding)

    sympy_matrix = Matrix(subset.values)
    ref_matrix, ref_pivots = sympy_matrix.echelon_form(with_pivots=True)
    rref_matrix, rref_pivots = sympy_matrix.rref()

    return {
        "subset": subset,
        "ref_matrix": ref_matrix,
        "ref_pivots": ref_pivots,
        "rref_matrix": rref_matrix,
        "rref_pivots": rref_pivots,
    }


def compute_column_correlation(matrix_df):
    """
    Compute correlation between columns.

    Why this is done:
    In the unemployment matrix, columns are years. Correlation between years
    shows whether years behave similarly across countries.

    Hoped-for outcome:
    A correlation matrix that reveals similar or unusual years.
    """
    clean_df = matrix_df.dropna(axis=0)
    if clean_df.empty:
        raise ValueError("No complete rows available for correlation.")
    return clean_df.corr()

def compute_country_similarity(matrix_df):
    """
    Compute cosine similarity between country vectors.

    Why this is done:
    We want to compare countries based on the shape of their trajectories or
    feature profiles.

    Hoped-for outcome:
    A country-by-country similarity matrix.
    """
    clean_df = matrix_df.dropna(axis=0)
    if clean_df.empty:
        raise ValueError("No complete rows available for similarity.")

    similarity = cosine_similarity(clean_df.values)
    similarity_df = pd.DataFrame(similarity, index=clean_df.index, columns=clean_df.index)
    return similarity_df


def run_pca(matrix_df, n_components=2):
    """
    Standardise the matrix and run PCA.

    Why this is done:
    PCA finds dominant orthogonal directions of variation.

    Hoped-for outcome:
    Lower-dimensional scores plus explained variance ratios.
    """
    clean_df = matrix_df.dropna(axis=0).copy()
    if clean_df.empty:
        raise ValueError("No complete rows available for PCA.")
    if n_components > min(clean_df.shape):
        raise ValueError("n_components is larger than the allowable PCA dimensionality.")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clean_df.values)

    pca = PCA(n_components=n_components)
    pca_scores = pca.fit_transform(scaled_data)

    pca_df = pd.DataFrame(
        pca_scores,
        index=clean_df.index,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )

    return {
        "clean_df": clean_df,
        "scaled_data": scaled_data,
        "pca_model": pca,
        "pca_scores": pca_df,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "components": pca.components_,
    }

