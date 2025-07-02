"""
Tests for categorical utilities in the ldcov package.

This module tests categorical variable handling for:
- Simple one-hot encoding
- Binary variable encoding
- Multiple categorical columns
- Missing value handling
- Edge cases
"""

import pytest
import pandas as pd

from ldcov.utils.categorical_utils import one_hot_encode_categorical


def test_one_hot_encode_simple():
    """Test simple one-hot encoding."""
    df = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0],
            "category": ["A", "B", "A"],
        }
    )

    encoded = one_hot_encode_categorical(df)

    # Check structure
    assert "numeric" in encoded.columns
    # For binary variables, only one column is kept (the second alphabetically)
    assert "category_B" in encoded.columns
    assert "category_A" not in encoded.columns  # Dropped to avoid collinearity
    assert "category" not in encoded.columns

    # Check values
    assert encoded["category_B"].tolist() == [0, 1, 0]


def test_one_hot_encode_binary():
    """Test one-hot encoding of binary variables."""
    df = pd.DataFrame(
        {
            "binary": ["yes", "no", "yes", "no"],
            "value": [1, 2, 3, 4],
        }
    )

    encoded = one_hot_encode_categorical(df)

    # Should only create one column for binary (keeps the second alphabetically)
    assert "binary_yes" in encoded.columns
    assert "binary_no" not in encoded.columns  # Dropped


def test_one_hot_encode_multiple_categories():
    """Test one-hot encoding with multiple categorical columns."""
    df = pd.DataFrame(
        {
            "cat1": ["A", "B", "C", "A"],
            "cat2": ["X", "Y", "X", "Y"],
            "numeric": [1, 2, 3, 4],
        }
    )

    encoded = one_hot_encode_categorical(df)

    # Check categories are encoded (first column dropped for each)
    assert "numeric" in encoded.columns
    # cat1: A dropped (first alphabetically), B and C kept
    assert "cat1_A" not in encoded.columns
    assert "cat1_B" in encoded.columns
    assert "cat1_C" in encoded.columns
    # cat2: X dropped (first alphabetically), Y kept
    assert "cat2_X" not in encoded.columns
    assert "cat2_Y" in encoded.columns


def test_one_hot_encode_with_nan():
    """Test one-hot encoding with missing values."""
    df = pd.DataFrame(
        {
            "category": ["A", "B", None, "A"],
            "numeric": [1, 2, 3, 4],
        }
    )

    encoded = one_hot_encode_categorical(df)

    # NaN should be encoded as 0 in remaining one-hot columns
    # A is dropped, B is kept
    assert "category_A" not in encoded.columns
    assert "category_B" in encoded.columns
    assert encoded["category_B"].iloc[2] == 0


def test_one_hot_encode_numeric_strings():
    """Test that numeric strings are treated as categorical."""
    df = pd.DataFrame(
        {
            "str_numeric": ["1", "2", "3"],
            "actual_string": ["A", "B", "C"],
        }
    )

    encoded = one_hot_encode_categorical(df)

    # Numeric strings are treated as categorical (object dtype)
    # str_numeric: 1 dropped, 2 and 3 kept
    assert "str_numeric_1" not in encoded.columns
    assert "str_numeric_2" in encoded.columns
    assert "str_numeric_3" in encoded.columns

    # Actual strings: A dropped, B and C kept
    assert "actual_string_A" not in encoded.columns
    assert "actual_string_B" in encoded.columns
    assert "actual_string_C" in encoded.columns


def test_one_hot_encode_edge_cases():
    """Test edge cases in one-hot encoding."""
    # Single unique value
    df1 = pd.DataFrame({"single": ["A", "A", "A"]})
    encoded1 = one_hot_encode_categorical(df1)
    assert "single_A" in encoded1.columns

    # Empty dataframe
    df2 = pd.DataFrame()
    encoded2 = one_hot_encode_categorical(df2)
    assert len(encoded2.columns) == 0

    # All numeric
    df3 = pd.DataFrame({"num1": [1, 2, 3], "num2": [4, 5, 6]})
    encoded3 = one_hot_encode_categorical(df3)
    assert list(encoded3.columns) == ["num1", "num2"]
