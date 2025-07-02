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
from click.testing import CliRunner
from typing import Tuple, List, Optional, Union

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


@pytest.fixture(scope="session")
def test_bgen_files(data_dir) -> dict:
    """Dictionary of test BGEN files by type."""
    return {
        "basic": data_dir / "data.bgen",
        "8bit": data_dir / "example.8bits.bgen",
        "16bit": data_dir / "example.16bits.bgen",
        "32bit": data_dir / "example.32bits.bgen",
        "zstd": data_dir / "example.16bits.zstd.bgen",
        "v11": data_dir / "example.v11.bgen",
    }


# ==================== Directory Fixtures ====================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix="ldcov_test_")
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="class")
def class_temp_dir(request):
    """Create a class-scoped temporary directory."""
    temp_dir = tempfile.mkdtemp(prefix=f"ldcov_test_{request.cls.__name__}_")
    request.cls.temp_dir = temp_dir
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


@pytest.fixture
def standard_cov_file(create_covariate_file):
    """Create a standard covariate file with PC1, PC2, and sex."""
    return create_covariate_file()


# ==================== CLI Testing Fixtures ====================


@pytest.fixture
def cli_runner():
    """Create a Click CLI test runner."""
    return CliRunner()


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


# ==================== Matrix Creation Fixtures ====================


@pytest.fixture
def create_test_matrix():
    """Factory fixture to create test correlation matrices."""

    def _create_matrix(
        n_vars: int = 10,
        symmetric: bool = True,
        unit_diagonal: bool = True,
        seed: int = 42,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Create a test correlation matrix with variant info.

        Args:
            n_vars: Number of variants
            symmetric: Whether to make the matrix symmetric
            unit_diagonal: Whether to set diagonal to 1.0
            seed: Random seed

        Returns:
            Tuple of (correlation_matrix, variant_info)
        """
        np.random.seed(seed)

        # Create matrix
        matrix = np.random.rand(n_vars, n_vars)
        if symmetric:
            matrix = (matrix + matrix.T) / 2

        # Ensure values are in [-1, 1] range for correlation
        matrix = matrix * 2 - 1

        if unit_diagonal:
            np.fill_diagonal(matrix, 1.0)

        # Create variant info
        variant_info = pd.DataFrame(
            {
                "rsid": [f"rs{i}" for i in range(n_vars)],
                "chrom": ["01"] * n_vars,
                "pos": list(range(1000, 1000 + n_vars * 100, 100)),
                "ref": ["A"] * n_vars,
                "alt": ["G"] * n_vars,
            }
        )

        return matrix, variant_info

    return _create_matrix


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


# ==================== Helper Functions ====================


def create_mock_genotypes(
    n_samples: int = 100,
    n_variants: int = 10,
    maf: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    """
    Create mock genotype data.

    Args:
        n_samples: Number of samples
        n_variants: Number of variants
        maf: Minor allele frequency
        seed: Random seed

    Returns:
        Genotype matrix (n_samples x n_variants)
    """
    np.random.seed(seed)
    return np.random.binomial(2, maf, size=(n_samples, n_variants)).astype(np.float64)


def assert_correlation_matrix_properties(matrix: np.ndarray, tol: float = 1e-10):
    """
    Assert that a matrix has properties of a correlation matrix.

    Args:
        matrix: Matrix to check
        tol: Tolerance for numerical comparisons
    """
    # Should be square
    assert matrix.shape[0] == matrix.shape[1], "Matrix should be square"

    # Should be symmetric
    assert np.allclose(matrix, matrix.T, atol=tol), "Matrix should be symmetric"

    # Diagonal should be close to 1 (for standard correlation matrices)
    # Note: For adjusted LD, diagonal might not be exactly 1
    diag_vals = np.diag(matrix)
    assert np.all(diag_vals >= 0), "Diagonal values should be non-negative"
    assert np.all(diag_vals <= 1.1), "Diagonal values should not exceed 1.1"

    # Values should be in [-1, 1] range
    assert np.all(matrix >= -1.0), "Correlation values should be >= -1"
    assert np.all(matrix <= 1.0), "Correlation values should be <= 1"


def compare_matrices(
    matrix1: np.ndarray,
    matrix2: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Tuple[float, float]:
    """
    Compare two matrices and return max and mean differences.

    Args:
        matrix1: First matrix
        matrix2: Second matrix
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Tuple of (max_diff, mean_diff)
    """
    diff = np.abs(matrix1 - matrix2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Also check with numpy's allclose for a more comprehensive comparison
    is_close = np.allclose(matrix1, matrix2, rtol=rtol, atol=atol)

    return max_diff, mean_diff


# ==================== Skip Decorators ====================


def skip_if_no_ldstore(func):
    """Skip test if LDstore is not available."""
    import shutil

    ldstore_available = shutil.which("ldstore") is not None
    return pytest.mark.skipif(not ldstore_available, reason="LDstore not available")(func)


def skip_if_slow(func):
    """Skip test if running in fast mode."""
    return pytest.mark.skipif(
        os.environ.get("LDCOV_FAST_TESTS", "0") == "1", reason="Skipping slow test"
    )(func)
