"""
Comprehensive CLI tests for the ldcov package.

This module tests:
- All CLI modes and options
- Auto BGI detection
- Custom covariate ID column
- Error handling and validation
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestCLI:
    """Test cases for CLI functionality."""

    @pytest.fixture(autouse=True)
    def setup_test_data(self, temp_dir, sample_ids, create_covariate_file):
        """Set up test data for CLI tests."""
        # Store references for use in tests
        self.temp_dir = temp_dir
        self.sample_ids = sample_ids

        # Create standard covariate file
        self.cov_file = create_covariate_file()

        # Create covariate file with custom ID column
        n_samples = len(sample_ids)
        self.custom_cov_file = create_covariate_file(
            columns=["PC1", "PC2"],
            id_col="FID",
            categorical_cols=[],
            custom_sample_ids=sample_ids,
        )

        # Add IID as a categorical column to the custom covariate file
        df = pd.read_csv(self.custom_cov_file)
        df["IID"] = ["patient" if i % 2 == 0 else "control" for i in range(n_samples)]
        df.to_csv(self.custom_cov_file, index=False)

    # ==================== Basic Mode Tests ====================

    def test_compute_ld_only(self, bgen_file, bgi_file, run_cli):
        """Test LD computation only mode."""
        output_prefix = os.path.join(self.temp_dir, "ld_only")

        result = run_cli(
            [
                "--bgen",
                str(bgen_file),
                "--out",
                output_prefix,
                "--compute-ld",
                "--bgi",
                str(bgi_file),
            ]
        )

        assert result.exit_code == 0
        # Check outputs
        ld_file = f"{output_prefix}.ld"
        assert os.path.exists(ld_file)
        assert not os.path.exists(f"{output_prefix}.adj.bgen")

    def test_compute_ld_with_covariates(self, bgen_file, bgi_file, sample_file, run_cli):
        """Test computing LD with covariate adjustment."""
        output_prefix = os.path.join(self.temp_dir, "ld_with_cov")

        result = run_cli(
            [
                "--bgen",
                str(bgen_file),
                "--out",
                output_prefix,
                "--compute-ld",
                "-c",
                self.cov_file,
                "--bgi",
                str(bgi_file),
                "--sample",
                str(sample_file),
            ]
        )

        assert result.exit_code == 0
        # Check LD output exists
        assert os.path.exists(f"{output_prefix}.ld")

    # ==================== Auto BGI Detection Tests ====================

    def test_auto_bgi_detection(self, create_temp_bgen, run_cli):
        """Test automatic BGI file detection."""
        # Create a temporary BGEN file with accompanying BGI
        temp_bgen, temp_bgi = create_temp_bgen("test_auto", with_bgi=True)

        output_prefix = os.path.join(self.temp_dir, "auto_bgi")

        # Run without specifying --bgi
        result = run_cli(
            [
                "--bgen",
                temp_bgen,
                "--out",
                output_prefix,
                "--compute-ld",
            ]
        )

        # Should succeed using auto-detected BGI
        assert result.exit_code == 0
        assert os.path.exists(f"{output_prefix}.ld")

    def test_no_auto_bgi_available(self, create_temp_bgen, run_cli):
        """Test behavior when no BGI file exists."""
        # Create a temporary BGEN without BGI
        temp_bgen, _ = create_temp_bgen("test_no_bgi", with_bgi=False)

        output_prefix = os.path.join(self.temp_dir, "no_bgi")

        result = run_cli(
            [
                "--bgen",
                temp_bgen,
                "--out",
                output_prefix,
                "--compute-ld",
            ]
        )

        # Should fail without BGI
        assert result.exit_code == 1

    # ==================== Custom Covariate ID Column Tests ====================

    def test_custom_covariate_id_column(self, bgen_file, bgi_file, sample_file, run_cli):
        """Test using custom ID column in covariate file."""
        output_prefix = os.path.join(self.temp_dir, "custom_id")

        result = run_cli(
            [
                "--bgen",
                str(bgen_file),
                "--out",
                output_prefix,
                "--compute-ld",
                "-c",
                self.custom_cov_file,
                "--covariate-id-col",
                "FID",
                "--bgi",
                str(bgi_file),
                "--sample",
                str(sample_file),
            ]
        )

        assert result.exit_code == 0
        # Should complete successfully
        assert os.path.exists(f"{output_prefix}.ld")

    def test_missing_covariate_id_column(self, bgen_file, bgi_file, run_cli):
        """Test error when specified ID column doesn't exist."""
        output_prefix = os.path.join(self.temp_dir, "missing_id_col")

        result = run_cli(
            [
                "--bgen",
                str(bgen_file),
                "--out",
                output_prefix,
                "--compute-ld",
                "-c",
                self.cov_file,
                "--covariate-id-col",
                "NONEXISTENT",
                "--bgi",
                str(bgi_file),
            ]
        )

        # Should raise an error
        assert result.exit_code != 0

    # ==================== Output Format Tests ====================

    def test_output_formats(self, bgen_file, bgi_file, run_cli):
        """Test different LD output formats."""
        for fmt in ["matrix", "long", "bcor"]:
            output_prefix = os.path.join(self.temp_dir, f"format_{fmt}")

            result = run_cli(
                [
                    "--bgen",
                    str(bgen_file),
                    "--out",
                    output_prefix,
                    "--compute-ld",
                    "--output-format",
                    fmt,
                    "--bgi",
                    str(bgi_file),
                ]
            )

            assert result.exit_code == 0

            # Check appropriate file was created
            if fmt == "matrix":
                assert os.path.exists(f"{output_prefix}.ld")
            elif fmt == "long":
                assert os.path.exists(f"{output_prefix}.ld.gz")
            elif fmt == "bcor":
                assert os.path.exists(f"{output_prefix}.bcor")
                # Sidecar index is written by default — regression guard for Task 5.
                assert os.path.exists(f"{output_prefix}.bcor.idx")

    # ==================== Region and Z-file Tests ====================

    def test_region_filtering(self, bgen_file, bgi_file, variant_info, run_cli):
        """Test region-based filtering."""
        # Use a region that contains at least some variants
        first_chrom = variant_info["chrom"].iloc[0]
        min_pos = variant_info["pos"].min()
        max_pos = variant_info["pos"].max()
        mid_pos = (min_pos + max_pos) // 2

        # Create a region that should contain some variants
        region = f"{first_chrom}:{min_pos}-{mid_pos}"

        output_prefix = os.path.join(self.temp_dir, "region_test")

        result = run_cli(
            [
                "--bgen",
                str(bgen_file),
                "--out",
                output_prefix,
                "--compute-ld",
                "--region",
                region,
                "--bgi",
                str(bgi_file),
            ]
        )

        assert result.exit_code == 0
        assert os.path.exists(f"{output_prefix}.ld")

    def test_z_file_filtering(self, bgen_file, bgi_file, create_z_file, run_cli):
        """Test Z-file based variant filtering."""
        # Create Z-file with first 3 variants
        z_file = create_z_file(n_variants=3)

        output_prefix = os.path.join(self.temp_dir, "z_file_test")

        result = run_cli(
            [
                "--bgen",
                str(bgen_file),
                "--out",
                output_prefix,
                "--compute-ld",
                "--z",
                z_file,
                "--bgi",
                str(bgi_file),
            ]
        )

        assert result.exit_code == 0
        assert os.path.exists(f"{output_prefix}.ld")

    # ==================== Error Handling Tests ====================

    def test_no_mode_specified(self, bgen_file, run_cli):
        """Test error when no mode is specified."""
        output_prefix = os.path.join(self.temp_dir, "no_mode")

        result = run_cli(
            [
                "--bgen",
                str(bgen_file),
                "--out",
                output_prefix,
            ]
        )

        assert result.exit_code != 0

    def test_invalid_bgen_file(self, run_cli):
        """Test error with invalid BGEN file."""
        output_prefix = os.path.join(self.temp_dir, "invalid_bgen")

        result = run_cli(
            [
                "--bgen",
                "/nonexistent/file.bgen",
                "--out",
                output_prefix,
                "--compute-ld",
            ]
        )

        assert result.exit_code != 0

    def test_whitespace_delimited_covariates(self, bgen_file, bgi_file, sample_file, run_cli):
        """Test loading whitespace-delimited covariate file."""
        # Create tab-delimited file
        ws_cov_file = os.path.join(self.temp_dir, "whitespace_cov.txt")
        with open(ws_cov_file, "w") as f:
            f.write("IID\tPC1\tPC2\n")
            # Use tab-delimited format - write data for more samples
            for i, sid in enumerate(self.sample_ids[:20]):  # Use 20 samples
                pc1 = 0.1 * (i + 1)
                pc2 = 0.2 * (i + 1)
                f.write(f"{sid}\t{pc1}\t{pc2}\n")

        output_prefix = os.path.join(self.temp_dir, "ws_cov")

        result = run_cli(
            [
                "--bgen",
                str(bgen_file),
                "--out",
                output_prefix,
                "--compute-ld",
                "-c",
                ws_cov_file,
                "--bgi",
                str(bgi_file),
                "--sample",
                str(sample_file),
            ]
        )

        # Should work with whitespace-delimited file
        assert result.exit_code == 0
        assert os.path.exists(f"{output_prefix}.ld")

    def test_verbose_mode(self, bgen_file, bgi_file, run_cli):
        """Test verbose logging mode."""
        output_prefix = os.path.join(self.temp_dir, "verbose_test")

        result = run_cli(
            [
                "--bgen",
                str(bgen_file),
                "--out",
                output_prefix,
                "--compute-ld",
                "--verbose",
                "--bgi",
                str(bgi_file),
            ]
        )

        # Should complete without error
        assert result.exit_code == 0
        assert os.path.exists(f"{output_prefix}.ld")

    def test_specific_covariate_columns(
        self, bgen_file, bgi_file, sample_file, create_covariate_file, run_cli
    ):
        """Test using specific columns as covariates."""
        # Create covariate file with multiple columns
        multi_cov_file = create_covariate_file(
            columns=["PC1", "PC2", "PC3", "PC4", "batch", "age"],
            categorical_cols=["batch"],
        )

        # Add age column to the file
        df = pd.read_csv(multi_cov_file)
        df["age"] = np.random.randint(20, 80, len(df))
        df.to_csv(multi_cov_file, index=False)

        output_prefix = os.path.join(self.temp_dir, "specific_cols")

        # Use only PC1 and PC2 as covariates
        result = run_cli(
            [
                "--bgen",
                str(bgen_file),
                "--out",
                output_prefix,
                "--compute-ld",
                "-c",
                multi_cov_file,
                "--covariate-cols",
                "PC1",
                "PC2",
                "--bgi",
                str(bgi_file),
                "--sample",
                str(sample_file),
            ]
        )

        assert result.exit_code == 0
        # Should complete successfully
        assert os.path.exists(f"{output_prefix}.ld")

    def test_invalid_covariate_columns(self, bgen_file, bgi_file, run_cli):
        """Test error when specifying non-existent covariate columns."""
        output_prefix = os.path.join(self.temp_dir, "invalid_cols")

        result = run_cli(
            [
                "--bgen",
                str(bgen_file),
                "--out",
                output_prefix,
                "--compute-ld",
                "-c",
                self.cov_file,
                "--covariate-cols",
                "PC1",
                "NonExistentColumn",
                "--bgi",
                str(bgi_file),
            ]
        )

        # Should raise an error
        assert result.exit_code != 0


def test_cli_emits_bcor_idx_by_default(tmp_path):
    examples = Path(__file__).parents[1] / "examples" / "data"
    out_prefix = tmp_path / "ld_out"
    result = subprocess.run(
        [
            sys.executable, "-m", "ldcov.cli.main",
            "--bgen", str(examples / "data.bgen"),
            "--sample", str(examples / "data.sample"),
            "--out", str(out_prefix),
            "--compute-ld",
            "--output-format", "bcor",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "ld_out.bcor").exists()
    assert (tmp_path / "ld_out.bcor.idx").exists()


def test_cli_no_bcor_idx_flag_skips_sidecar(tmp_path):
    examples = Path(__file__).parents[1] / "examples" / "data"
    out_prefix = tmp_path / "ld_out"
    result = subprocess.run(
        [
            sys.executable, "-m", "ldcov.cli.main",
            "--bgen", str(examples / "data.bgen"),
            "--sample", str(examples / "data.sample"),
            "--out", str(out_prefix),
            "--compute-ld",
            "--output-format", "bcor",
            "--no-bcor-idx",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "ld_out.bcor").exists()
    assert not (tmp_path / "ld_out.bcor.idx").exists()
