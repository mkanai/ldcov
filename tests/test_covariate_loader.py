"""
Tests for covariate loading functionality.

This module contains all tests related to loading and processing covariate files,
including handling of categorical variables, missing data, and sample filtering.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

from ldcov.io.covariate_loader import load_covariates
from ldcov.utils.categorical_utils import one_hot_encode_categorical


class TestCovariateLoading:
    """Test loading covariates from various file formats."""
    
    def test_load_covariates_csv(self, tmp_path):
        """Test loading covariates from CSV file."""
        # Create test CSV
        cov_data = pd.DataFrame(
            {
                "IID": ["sample1", "sample2", "sample3"],
                "PC1": [0.1, 0.2, 0.3],
                "PC2": [-0.1, -0.2, -0.3],
                "sex": ["male", "female", "male"],
            }
        )
        cov_file = tmp_path / "test_cov.csv"
        cov_data.to_csv(cov_file, index=False)
        
        # Load covariates
        loaded_cov = load_covariates(str(cov_file))
        
        # Check one-hot encoding was applied
        assert "sex_male" in loaded_cov.columns
        assert "sex" not in loaded_cov.columns
        assert list(loaded_cov.index) == ["sample1", "sample2", "sample3"]
    
    def test_load_covariates_whitespace(self, tmp_path):
        """Test loading whitespace-delimited covariate file."""
        # Create whitespace-delimited file
        cov_file = tmp_path / "test_cov.txt"
        with open(cov_file, "w") as f:
            f.write("IID PC1 PC2 batch\n")
            f.write("s1   0.1  0.2 A\n")
            f.write("s2   0.3  0.4 B\n")
        
        loaded_cov = load_covariates(str(cov_file))
        
        # Check loading and one-hot encoding
        assert len(loaded_cov) == 2
        # batch_A is dropped (first alphabetically), batch_B is kept
        assert "batch_A" not in loaded_cov.columns
        assert "batch_B" in loaded_cov.columns
    
    def test_load_covariates_custom_id_column(self, tmp_path):
        """Test loading covariates with custom ID column."""
        cov_data = pd.DataFrame(
            {
                "FID": ["fam1", "fam2"],
                "IID": ["ind1", "ind2"],
                "PC1": [0.1, 0.2],
            }
        )
        cov_file = tmp_path / "test_cov_fid.csv"
        cov_data.to_csv(cov_file, index=False)
        
        # Load with FID as ID column
        loaded_cov = load_covariates(str(cov_file), id_col="FID")
        assert list(loaded_cov.index) == ["fam1", "fam2"]
        assert "IID_ind2" in loaded_cov.columns  # IID becomes a feature


class TestCovariateErrorHandling:
    """Test error handling in covariate loading."""
    
    def test_load_covariates_error_handling(self, tmp_path):
        """Test error handling in covariate loading."""
        # Non-existent file
        with pytest.raises(ValueError):
            load_covariates("/nonexistent/file.txt")
        
        # Missing ID column
        cov_data = pd.DataFrame({"PC1": [0.1, 0.2], "PC2": [0.3, 0.4]})
        cov_file = tmp_path / "test_no_id.csv"
        cov_data.to_csv(cov_file, index=False)
        
        with pytest.raises(ValueError, match="ID column 'IID' not found"):
            load_covariates(str(cov_file), id_col="IID")


class TestCovariateSampleFiltering:
    """Test sample filtering functionality in covariate loading."""
    
    def test_load_covariates_sample_filtering(self, tmp_path):
        """Test covariate loading with sample filtering."""
        cov_data = pd.DataFrame(
            {
                "IID": ["1", "2", "3", "4", "5"],
                "PC1": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )
        cov_file = tmp_path / "test_filter.csv"
        cov_data.to_csv(cov_file, index=False)
        
        # Load with subset of samples
        loaded_cov = load_covariates(str(cov_file), sample_ids=["1", "3", "5", "99"])
        
        # Should only have samples that exist in both
        assert len(loaded_cov) == 3
        assert list(loaded_cov.index) == ["1", "3", "5"]


class TestCovariateColumnSelection:
    """Test column selection functionality in covariate loading."""
    
    def test_load_covariates_with_specific_columns(self, tmp_path):
        """Test loading covariates with specific columns selected."""
        # Create test CSV with multiple columns
        cov_data = pd.DataFrame(
            {
                "IID": ["sample1", "sample2", "sample3"],
                "PC1": [0.1, 0.2, 0.3],
                "PC2": [-0.1, -0.2, -0.3],
                "PC3": [0.5, 0.6, 0.7],
                "batch": ["A", "B", "A"],
                "age": [25, 45, 65],
            }
        )
        cov_file = tmp_path / "test_multi_cov.csv"
        cov_data.to_csv(cov_file, index=False)
        
        # Load with specific columns
        loaded_cov = load_covariates(str(cov_file), cols_to_use=["PC1", "PC3", "age"])
        
        # Check that only requested columns are present (plus any one-hot encoded)
        assert "PC1" in loaded_cov.columns
        assert "PC3" in loaded_cov.columns
        assert "age" in loaded_cov.columns
        assert "PC2" not in loaded_cov.columns
        assert "batch" not in loaded_cov.columns
        
        # Check all samples are present
        assert list(loaded_cov.index) == ["sample1", "sample2", "sample3"]
    
    def test_load_covariates_invalid_columns(self, tmp_path):
        """Test error handling when requesting non-existent columns."""
        cov_data = pd.DataFrame(
            {
                "IID": ["sample1", "sample2"],
                "PC1": [0.1, 0.2],
                "PC2": [-0.1, -0.2],
            }
        )
        cov_file = tmp_path / "test_invalid_cols.csv"
        cov_data.to_csv(cov_file, index=False)
        
        # Request non-existent columns
        with pytest.raises(ValueError, match="Requested covariate columns not found"):
            load_covariates(str(cov_file), cols_to_use=["PC1", "NonExistent"])


class TestCategoricalEncoding:
    """Test one-hot encoding of categorical variables."""
    
    def test_one_hot_encoding(self):
        """Test one-hot encoding of categorical variables."""
        df = pd.DataFrame(
            {
                "numeric": [1.0, 2.0, 3.0],
                "category": ["A", "B", "A"],
                "binary": ["yes", "no", "yes"],
            }
        )
        
        encoded = one_hot_encode_categorical(df)
        
        # Check encoding (first categories dropped)
        assert "numeric" in encoded.columns
        # category_A dropped (first alphabetically), category_B kept
        assert "category_A" not in encoded.columns
        assert "category_B" in encoded.columns
        # binary: "no" dropped (first alphabetically), "yes" kept
        assert "binary_no" not in encoded.columns
        assert "binary_yes" in encoded.columns
        # Original categorical columns removed
        assert "category" not in encoded.columns
        assert "binary" not in encoded.columns