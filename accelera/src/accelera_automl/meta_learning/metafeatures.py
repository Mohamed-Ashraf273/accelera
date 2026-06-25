import numpy as np
import pandas as pd


def as_dataframe(X):
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X)


def compute_classification_metafeatures(X, y):
    X = as_dataframe(X)
    y = np.asarray(y)

    num_of_rows, num_of_features = X.shape
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    class_entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
    num_cols, cat_cols = get_num_and_cat_cols(X)
    num_arr = (
        num_cols.to_numpy(dtype=float)
        if not num_cols.empty
        else np.empty((num_of_rows, 0), dtype=float)
    )

    if num_arr.shape[1] > 0:
        means = np.nanmean(num_arr, axis=0)
        stds = np.nanstd(num_arr, axis=0)
        centered = num_arr - means
        skew = np.nanmean((centered / (stds + 1e-12)) ** 3, axis=0)
        kurt = np.nanmean((centered / (stds + 1e-12)) ** 4, axis=0) - 3.0
    else:
        skew = np.asarray([0.0])
        kurt = np.asarray([0.0])

    missing_mask = X.isna().to_numpy()
    num_of_missings = float(missing_mask.sum())

    n_instances_with_missing = float(np.any(missing_mask, axis=1).sum())
    n_features_with_missing = float(np.any(missing_mask, axis=0).sum())

    dataset_ratio = float(num_of_rows / max(num_of_features, 1))
    inverse_dataset_ratio = float(num_of_rows / max(num_of_features, 1))
    n_categorical = float(cat_cols.shape[1])
    n_numeric = float(num_cols.shape[1])
    symbol_counts = categorical_symbol_counts(cat_cols)

    return {
        "ClassEntropy": class_entropy,
        "ClassProbabilityMax": float(probs.max()),
        "ClassProbabilityMean": float(probs.mean()),
        "ClassProbabilityMin": float(probs.min()),
        "ClassProbabilitySTD": float(probs.std()),
        "DatasetRatio": dataset_ratio,
        "InverseDatasetRatio": inverse_dataset_ratio,
        "KurtosisMax": float(np.nanmax(kurt)),
        "KurtosisMean": float(np.nanmean(kurt)),
        "KurtosisMin": float(np.nanmin(kurt)),
        "KurtosisSTD": float(np.nanstd(kurt)),
        "LogDatasetRatio": float(np.log(dataset_ratio)),
        "LogInverseDatasetRatio": float(np.log(inverse_dataset_ratio)),
        "LogNumberOfFeatures": float(np.log(num_of_features)),
        "LogNumberOfInstances": float(np.log(num_of_rows)),
        "NumberOfCategoricalFeatures": n_categorical,
        "NumberOfClasses": float(len(classes)),
        "NumberOfFeatures": float(num_of_features),
        "NumberOfFeaturesWithMissingValues": n_features_with_missing,
        "NumberOfInstances": float(num_of_rows),
        "NumberOfInstancesWithMissingValues": n_instances_with_missing,
        "NumberOfMissingValues": num_of_missings,
        "NumberOfNumericFeatures": n_numeric,
        "PercentageOfFeaturesWithMissingValues": float(
            n_features_with_missing / num_of_features
        ),
        "PercentageOfInstancesWithMissingValues": float(
            n_instances_with_missing / num_of_rows
        ),
        "PercentageOfMissingValues": float(
            num_of_missings / num_of_rows * num_of_features
        ),
        "RatioNominalToNumerical": float(n_categorical / max(n_numeric, 1.0)),
        "RatioNumericalToNominal": float(n_numeric / max(n_categorical, 1.0)),
        "SkewnessMax": float(np.nanmax(skew)),
        "SkewnessMean": float(np.nanmean(skew)),
        "SkewnessMin": float(np.nanmin(skew)),
        "SkewnessSTD": float(np.nanstd(skew)),
        "SymbolsMax": float(symbol_counts.max(initial=0.0)),
        "SymbolsMean": float(symbol_counts.mean() if symbol_counts.size else 0.0),
        "SymbolsMin": float(symbol_counts.min(initial=0.0)),
        "SymbolsSTD": float(symbol_counts.std() if symbol_counts.size else 0.0),
        "SymbolsSum": float(symbol_counts.sum()),
    }


def compute_basic_regression_metafeatures(X, y):
    X = as_dataframe(X)
    y = np.asarray(y, dtype=float).reshape(-1)

    n_instances, n_features = X.shape
    numeric_df, categorical_df = get_num_and_cat_cols(X)
    numeric_arr = (
        numeric_df.to_numpy(dtype=float)
        if not numeric_df.empty
        else np.empty((n_instances, 0), dtype=float)
    )

    if numeric_arr.shape[1] > 0:
        means = np.nanmean(numeric_arr, axis=0)
        stds = np.nanstd(numeric_arr, axis=0)
        centered = numeric_arr - means
        skew = np.nanmean((centered / (stds + 1e-12)) ** 3, axis=0)
        kurt = np.nanmean((centered / (stds + 1e-12)) ** 4, axis=0) - 3.0
    else:
        skew = np.asarray([0.0])
        kurt = np.asarray([0.0])

    target_mean = float(np.nanmean(y))
    target_std = float(np.nanstd(y))
    centered_y = y - target_mean
    target_skew = float(np.nanmean((centered_y / (target_std + 1e-12)) ** 3))
    target_kurtosis = float(
        np.nanmean((centered_y / (target_std + 1e-12)) ** 4) - 3.0
    )

    missing_mask = X.isna().to_numpy()
    n_missing_values = float(missing_mask.sum())
    n_instances_with_missing = float(np.any(missing_mask, axis=1).sum())
    n_features_with_missing = float(np.any(missing_mask, axis=0).sum())

    dataset_ratio = float(n_features / max(n_instances, 1))
    inverse_dataset_ratio = float(n_instances / max(n_features, 1))
    n_categorical = float(categorical_df.shape[1])
    n_numeric = float(numeric_df.shape[1])
    symbol_counts = categorical_symbol_counts(categorical_df)

    return {
        "DatasetRatio": dataset_ratio,
        "InverseDatasetRatio": inverse_dataset_ratio,
        "KurtosisMax": float(np.nanmax(kurt)),
        "KurtosisMean": float(np.nanmean(kurt)),
        "KurtosisMin": float(np.nanmin(kurt)),
        "KurtosisSTD": float(np.nanstd(kurt)),
        "LogDatasetRatio": float(np.log(dataset_ratio)),
        "LogInverseDatasetRatio": float(np.log(inverse_dataset_ratio)),
        "LogNumberOfFeatures": float(np.log(n_features)),
        "LogNumberOfInstances": float(np.log(n_instances)),
        "NumberOfCategoricalFeatures": n_categorical,
        "NumberOfFeatures": float(n_features),
        "NumberOfFeaturesWithMissingValues": n_features_with_missing,
        "NumberOfInstances": float(n_instances),
        "NumberOfInstancesWithMissingValues": n_instances_with_missing,
        "NumberOfMissingValues": n_missing_values,
        "NumberOfNumericFeatures": n_numeric,
        "PercentageOfFeaturesWithMissingValues": float(
            n_features_with_missing / n_features
        ),
        "PercentageOfInstancesWithMissingValues": float(
            n_instances_with_missing / n_instances
        ),
        "PercentageOfMissingValues": float(
            n_missing_values / n_instances * n_features
        ),
        "RatioNominalToNumerical": float(n_categorical / max(n_numeric, 1.0)),
        "RatioNumericalToNominal": float(n_numeric / max(n_categorical, 1.0)),
        "SkewnessMax": float(np.nanmax(skew)),
        "SkewnessMean": float(np.nanmean(skew)),
        "SkewnessMin": float(np.nanmin(skew)),
        "SkewnessSTD": float(np.nanstd(skew)),
        "SymbolsMax": float(symbol_counts.max(initial=0.0)),
        "SymbolsMean": float(symbol_counts.mean() if symbol_counts.size else 0.0),
        "SymbolsMin": float(symbol_counts.min(initial=0.0)),
        "SymbolsSTD": float(symbol_counts.std() if symbol_counts.size else 0.0),
        "SymbolsSum": float(symbol_counts.sum()),
        "TargetMean": target_mean,
        "TargetSTD": target_std,
        "TargetSkewness": target_skew,
        "TargetKurtosis": target_kurtosis,
        "LogTargetSTD": float(np.log(max(abs(target_std), 1e-12))),
    }


def get_num_and_cat_cols(X):
    num_cols = []
    cat_cols = []

    for column in X.columns:
        col = X[column]
        if pd.api.types.is_numeric_dtype(col):
            num_cols.append(column)
            continue

        coerced = pd.to_numeric(col, errors="coerce")
        non_missing_original = col.notna().sum()
        non_missing_coerced = coerced.notna().sum()
        if non_missing_original > 0 and non_missing_original == non_missing_coerced:
            num_cols.append(column)
        else:
            cat_cols.append(column)

    numeric_df = (
        X[num_cols].apply(pd.to_numeric, errors="coerce")
        if num_cols
        else pd.DataFrame(index=X.index)
    )
    categorical_df = X[cat_cols] if cat_cols else pd.DataFrame(index=X.index)
    return numeric_df, categorical_df


def categorical_symbol_counts(categorical_df):
    if categorical_df.empty:
        return np.asarray([], dtype=float)

    counts = []
    for column in categorical_df.columns:
        col = categorical_df[column]
        counts.append(float(col.dropna().nunique()))
    return np.asarray(counts, dtype=float)


compute_basic_classification_metafeatures = compute_classification_metafeatures
