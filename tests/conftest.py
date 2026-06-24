"""
Shared test fixtures and utilities for the ldcov test suite.

This module provides common fixtures and helper functions used across multiple test files
to reduce code duplication and improve test maintainability.
"""

import pytest
import tempfile
import shutil
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional

# Import for type annotations
from ldcov.io import load_bgen

# ==================== Path Fixtures ====================


@pytest.fixture(scope="session")
def examples_dir() -> Path:
    """Return path to the examples directory."""
    return Path(__file__).parents[1] / "examples"


@pytest.fixture(scope="session")
def data_dir(examples_dir) -> Path:
    """Return path to the test data directory."""
    return examples_dir / "data"


@pytest.fixture(scope="session")
def bgen_file(data_dir) -> Path:
    """Path to the main test BGEN file."""
    return data_dir / "data.bgen"


@pytest.fixture(scope="session")
def bgi_file(data_dir) -> Path:
    """Path to the main test BGI index file."""
    return data_dir / "data.bgen.bgi"


@pytest.fixture(scope="session")
def sample_file(data_dir) -> Path:
    """Path to the main test sample file."""
    return data_dir / "data.sample"


@pytest.fixture(scope="session")
def ref_bcor_file(data_dir) -> Path:
    """Path to the reference BCOR file."""
    return data_dir / "data.bcor"


@pytest.fixture(scope="session")
def ref_ld_file(data_dir) -> Path:
    """Path to the reference LD file."""
    return data_dir / "data.ld"


# ==================== Directory Fixtures ====================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix="ldcov_test_")
    yield temp_dir
    shutil.rmtree(temp_dir)


# ==================== Data Loading Fixtures ====================


@pytest.fixture(scope="session")
def loaded_test_data(
    bgen_file, bgi_file, sample_file
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """Load the main test BGEN data once for the session."""
    genotypes, variant_info, sample_ids = load_bgen(
        file_path=str(bgen_file),
        index_path=str(bgi_file),
        sample_path=str(sample_file),
    )
    return genotypes, variant_info, sample_ids


@pytest.fixture
def genotypes(loaded_test_data) -> np.ndarray:
    """Return loaded genotypes."""
    return loaded_test_data[0].copy()  # Return a copy to avoid test interference


@pytest.fixture
def variant_info(loaded_test_data) -> pd.DataFrame:
    """Return loaded variant information."""
    return loaded_test_data[1].copy()


@pytest.fixture
def sample_ids(loaded_test_data) -> List[str]:
    """Return loaded sample IDs."""
    return loaded_test_data[2].copy()


# ==================== Covariate Fixtures ====================


@pytest.fixture
def create_covariate_file(temp_dir, sample_ids):
    """Factory fixture to create covariate files with various configurations."""

    def _create_cov_file(
        n_samples: Optional[int] = None,
        columns: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        id_col: str = "IID",
        file_format: str = "csv",
        custom_sample_ids: Optional[List[str]] = None,
    ) -> str:
        """
        Create a covariate file with specified configuration.

        Args:
            n_samples: Number of samples (defaults to all sample_ids)
            columns: List of column names (defaults to ["PC1", "PC2", "sex"])
            categorical_cols: List of categorical columns (defaults to ["sex"])
            id_col: Name of the ID column
            file_format: Output format ("csv" or "txt" for whitespace-delimited)
            custom_sample_ids: Custom sample IDs to use

        Returns:
            Path to created covariate file
        """
        if n_samples is None:
            n_samples = len(sample_ids)

        if columns is None:
            columns = ["PC1", "PC2", "sex"]

        if categorical_cols is None:
            categorical_cols = ["sex"]

        # Use custom sample IDs or subset of actual sample IDs
        if custom_sample_ids is not None:
            ids_to_use = custom_sample_ids
        else:
            ids_to_use = sample_ids[:n_samples]

        # Update n_samples to match actual IDs used
        n_samples = len(ids_to_use)

        # Create dataframe
        np.random.seed(42)
        data = {id_col: ids_to_use}

        for col in columns:
            if col in categorical_cols:
                if col == "sex":
                    data[col] = np.random.choice(["male", "female"], n_samples)
                else:
                    data[col] = np.random.choice(["A", "B", "C"], n_samples)
            else:
                data[col] = np.random.normal(0, 1, n_samples)

        df = pd.DataFrame(data)

        # Save file with unique filename
        import uuid

        unique_id = str(uuid.uuid4())[:8]
        if file_format == "csv":
            filename = os.path.join(temp_dir, f"covariates_{n_samples}_{unique_id}.csv")
            df.to_csv(filename, index=False)
        else:  # whitespace-delimited
            filename = os.path.join(temp_dir, f"covariates_{n_samples}_{unique_id}.txt")
            df.to_csv(filename, sep="\t", index=False)

        return filename

    return _create_cov_file


# ==================== CLI Testing Fixtures ====================


@pytest.fixture
def run_cli():
    """Factory fixture to run CLI commands with common defaults."""

    def _run_cli(args: List[str], catch_exceptions: bool = False):
        """
        Run ldcov CLI with given arguments.

        Args:
            args: List of command line arguments
            catch_exceptions: Whether to catch exceptions

        Returns:
            Result object with exit_code, output, and exception attributes
        """
        import subprocess
        import sys
        from typing import NamedTuple

        class Result(NamedTuple):
            exit_code: int
            output: str
            exception: Optional[Exception] = None

        # Ensure args start with 'ldcov'
        if not args or args[0] != "ldcov":
            args = ["ldcov"] + args

        try:
            # Run the command as a subprocess
            result = subprocess.run(
                [sys.executable, "-m", "ldcov.cli.main"] + args[1:],
                capture_output=True,
                text=True,
                check=False,
            )

            # Combine stdout and stderr like Click does
            output = result.stdout
            if result.stderr:
                output += result.stderr

            return Result(exit_code=result.returncode, output=output, exception=None)
        except Exception as e:
            if catch_exceptions:
                return Result(exit_code=1, output=str(e), exception=e)
            else:
                raise

    return _run_cli


# ==================== BGEN Creation Fixtures ====================


@pytest.fixture
def create_temp_bgen(temp_dir, bgen_file, bgi_file):
    """Factory fixture to create temporary BGEN files."""

    def _create_temp_bgen(name: str = "test", with_bgi: bool = True) -> Tuple[str, Optional[str]]:
        """
        Create a temporary BGEN file by copying the test file.

        Args:
            name: Base name for the BGEN file
            with_bgi: Whether to also copy the BGI index

        Returns:
            Tuple of (bgen_path, bgi_path or None)
        """
        temp_bgen = os.path.join(temp_dir, f"{name}.bgen")
        shutil.copy(str(bgen_file), temp_bgen)

        temp_bgi = None
        if with_bgi:
            temp_bgi = os.path.join(temp_dir, f"{name}.bgen.bgi")
            shutil.copy(str(bgi_file), temp_bgi)

        return temp_bgen, temp_bgi

    return _create_temp_bgen


# ==================== Z-file Creation Fixtures ====================


@pytest.fixture
def create_z_file(temp_dir, variant_info):
    """Factory fixture to create Z-files for variant filtering."""

    def _create_z_file(
        n_variants: Optional[int] = None,
        variant_indices: Optional[List[int]] = None,
        filename: str = "test.z",
    ) -> str:
        """
        Create a Z-file with variant information.

        Args:
            n_variants: Number of variants to include (from start)
            variant_indices: Specific variant indices to include
            filename: Output filename

        Returns:
            Path to created Z-file
        """
        if variant_indices is not None:
            subset_info = variant_info.iloc[variant_indices]
        elif n_variants is not None:
            subset_info = variant_info.iloc[:n_variants]
        else:
            subset_info = variant_info

        z_data = pd.DataFrame(
            {
                "rsid": subset_info["rsid"].tolist(),
                "chromosome": subset_info["chrom"].tolist(),
                "position": subset_info["pos"].astype(str).tolist(),
                "allele1": subset_info["ref"].tolist(),
                "allele2": subset_info["alt"].tolist(),
            }
        )

        z_file = os.path.join(temp_dir, filename)
        z_data.to_csv(z_file, sep="\t", index=False)

        return z_file

    return _create_z_file
