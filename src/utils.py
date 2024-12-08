import pandas as pd
import numpy as np
from scipy import stats


def load_data(file_path):
    return pd.read_csv(file_path)


def inspect_basic_structure(data):
    print("\033[32mInspect Basic Structure\033[0m\n")

    print("\033[33mData Shape\033[0m")
    print(data.shape, "\n")

    print("\033[33mData Info\033[0m")
    print(data.info(), "\n")

    print("\033[33mData Head\033[0m")
    print(data.head(), "\n")

    print("\033[33mData Tail\033[0m")
    print(data.tail(), "\n")


def examine_summary_statistics(data):
    print("\033[32mExamine Summary Statistics\033[0m\n")

    print("\033[33mData Describe\033[0m")
    print(data.describe(), "\n")


def identify_missing_values(data):
    print("\033[32mIdentify Missing Values\033[0m\n")

    print("\033[33mTotal Missing Values\033[0m")
    total_missing = data.isnull().sum().sum()
    print(total_missing, "\n")

    print("\033[33mMissing Values Per Column\033[0m")
    missing_per_column = data.isnull().sum()
    print(missing_per_column[missing_per_column > 0], "\n")

    print("\033[33mPercentage of Missing Values Per Column\033[0m")
    missing_percentage = (data.isnull().sum() / len(data)) * 100
    print(missing_percentage[missing_percentage > 0], "\n")


def check_for_duplicates(data):
    print("\033[32mCheck for Duplicates\033[0m\n")

    print("\033[33mData Duplicates\033[0m")
    print(data.duplicated().sum(), "\n")


def check_for_constant_columns(data):
    print("\033[32mCheck for Constant Columns\033[0m\n")

    print("\033[33mConstant Columns\033[0m")
    print(data.columns[data.nunique() == 1], "\n")


def check_data_types(data):
    print("\033[32mCheck Data Types\033[0m\n")

    print("\033[33mData Types\033[0m")
    print(data.dtypes, "\n")


def detect_outliers(data):
    print("\033[32mDetect Outliers\033[0m\n")

    numeric_columns = data.select_dtypes(include=['float64', 'int64'])
    q1 = numeric_columns.quantile(0.25)
    q3 = numeric_columns.quantile(0.75)

    iqr = q3 - q1
    z_scores = np.abs(stats.zscore(numeric_columns, nan_policy='omit'))

    outliers_iqr = ((numeric_columns < (q1 - 1.5 * iqr)) |
                    (numeric_columns > (q3 + 1.5 * iqr))).any(axis=0)

    outliers_z = (z_scores > 3).any(axis=0)

    print("\033[33mOutliers Detected (IQR Method)\033[0m")
    print(outliers_iqr.sum())
    print(outliers_iqr, "\n")

    print("\033[33mOutliers Detected (Z-Score Method)\033[0m")
    print(outliers_z.sum())
    print(outliers_z, "\n")


def save_data(data, file_path):
    data.to_csv(file_path, index=False)
