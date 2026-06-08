from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # 2. Encode string categorical features (dtype `object`)
    object_cols = working_train_df.select_dtypes(include="object").columns.tolist()
    binary_cols = [col for col in object_cols if working_train_df[col].nunique() <= 2]
    multi_cols = [col for col in object_cols if working_train_df[col].nunique() > 2]

    ord_enc = OrdinalEncoder()
    working_train_df[binary_cols] = ord_enc.fit_transform(working_train_df[binary_cols])
    working_val_df[binary_cols] = ord_enc.transform(working_val_df[binary_cols])
    working_test_df[binary_cols] = ord_enc.transform(working_test_df[binary_cols])

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    train_ohe = ohe.fit_transform(working_train_df[multi_cols])
    val_ohe = ohe.transform(working_val_df[multi_cols])
    test_ohe = ohe.transform(working_test_df[multi_cols])

    working_train_df = working_train_df.drop(columns=multi_cols)
    working_val_df = working_val_df.drop(columns=multi_cols)
    working_test_df = working_test_df.drop(columns=multi_cols)

    train = np.concatenate([working_train_df.values, train_ohe], axis=1)
    val = np.concatenate([working_val_df.values, val_ohe], axis=1)
    test = np.concatenate([working_test_df.values, test_ohe], axis=1)

    # 3. Impute values for all columns with missing data
    imputer = SimpleImputer(strategy="median")
    train = imputer.fit_transform(train)
    val = imputer.transform(val)
    test = imputer.transform(test)

    # 4. Feature scaling with Min-Max scaler
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)

    return train, val, test
