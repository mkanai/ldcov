"""
Utility functions for handling categorical variables.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def one_hot_encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically detect and one-hot encode categorical columns in a DataFrame.

    Note: For each categorical variable, the first category in alphabetical order is dropped
    to avoid collinearity issues in regression models.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing potential categorical columns

    Returns:
    --------
    pandas.DataFrame
        DataFrame with categorical columns replaced by one-hot encoded columns
    """
    # Make a copy to avoid modifying the original
    encoded_df = df.copy()

    # Find categorical columns
    categorical_columns = []
    for col in encoded_df.columns:
        if encoded_df[col].dtype == "object" or isinstance(
            encoded_df[col].dtype, pd.CategoricalDtype
        ):
            categorical_columns.append(col)

    # Apply one-hot encoding to all categorical columns
    if categorical_columns:
        logger.info(
            f"Applying one-hot encoding to {len(categorical_columns)} categorical columns: {', '.join(categorical_columns)}"
        )

        for cat_col in categorical_columns:
            # One-hot encode without dropping any column (we'll handle this manually)
            dummies = pd.get_dummies(encoded_df[cat_col], prefix=cat_col, drop_first=False)

            # If there are only two categories, drop one column to avoid collinearity
            if dummies.shape[1] == 2:
                # For binary variables, drop the first column (alphabetically)
                drop_col = dummies.columns[0]
                logger.info(
                    f"Dropping {drop_col} from binary column {cat_col} to avoid collinearity"
                )
                dummies = dummies.drop(columns=[drop_col])
            # For multi-category variables, also drop the first column (alphabetically)
            elif dummies.shape[1] > 2:
                drop_col = dummies.columns[0]
                logger.info(
                    f"Dropping {drop_col} from multi-category column {cat_col} to avoid collinearity"
                )
                dummies = dummies.drop(columns=[drop_col])

            # Remove the original column and add the dummy variables
            encoded_df = encoded_df.drop(columns=[cat_col])
            encoded_df = pd.concat([encoded_df, dummies], axis=1)

    return encoded_df
