"""
Comprehensive CLI tests for the ldcov package.

This module tests:
- All CLI modes and options
- Auto BGI detection
- Custom covariate ID column
- Error handling and validation
"""

import unittest
import sys
import os
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

from ldcov.cli.main import main
from ldcov.io.bgen_reader import load_bgen


class TestCLI(unittest.TestCase):
    """Test cases for CLI functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        cls.examples_dir = Path(__file__).parents[1] / "examples"
        cls.bgen_file = cls.examples_dir / "data" / "data.bgen"
        cls.bgi_file = cls.examples_dir / "data" / "data.bgen.bgi"
        cls.sample_file = cls.examples_dir / "data" / "data.sample"

        # Create temporary directory
        cls.temp_dir = tempfile.mkdtemp(prefix="ldcov_test_cli_")

        # Load sample IDs for creating covariate files
        _, _, cls.sample_ids = load_bgen(
            file_path=str(cls.bgen_file),
            index_path=str(cls.bgi_file),
            sample_path=str(cls.sample_file),
        )

        # Create standard covariate file
        n_samples = len(cls.sample_ids)
        np.random.seed(42)
        covariates = pd.DataFrame(
            {
                "IID": cls.sample_ids,
                "PC1": np.random.normal(0, 1, n_samples),
                "PC2": np.random.normal(0, 1, n_samples),
                "sex": np.random.choice(["male", "female"], n_samples),
            }
        )
        cls.cov_file = os.path.join(cls.temp_dir, "test_covariates.csv")
        covariates.to_csv(cls.cov_file, index=False)

        # Create covariate file with custom ID column
        # Use FID as the matching column, and create a reasonable categorical IID column
        covariates_custom = pd.DataFrame(
            {
                "FID": cls.sample_ids,
                "IID": ["patient" if i % 2 == 0 else "control" for i in range(n_samples)],
                "PC1": np.random.normal(0, 1, n_samples),
                "PC2": np.random.normal(0, 1, n_samples),
            }
        )
        cls.custom_cov_file = os.path.join(cls.temp_dir, "custom_id_covariates.csv")
        covariates_custom.to_csv(cls.custom_cov_file, index=False)

    @classmethod
    def tearDownClass(cls):
        """Clean up test data."""
        shutil.rmtree(cls.temp_dir)

    # ==================== Basic Mode Tests ====================

    def test_compute_ld_only(self):
        """Test LD computation only mode."""
        output_prefix = os.path.join(self.temp_dir, "ld_only")

        sys.argv = [
            "ldcov",
            "--bgen",
            str(self.bgen_file),
            "--out",
            output_prefix,
            "--compute-ld",
            "--bgi",
            str(self.bgi_file),
        ]

        main()

        # Check outputs
        ld_file = f"{output_prefix}.ld"
        self.assertTrue(os.path.exists(ld_file))
        self.assertFalse(os.path.exists(f"{output_prefix}.adj.bgen"))

    def test_export_adjusted_only(self):
        """Test adjusted genotype export only mode."""
        output_prefix = os.path.join(self.temp_dir, "adj_only")

        sys.argv = [
            "ldcov",
            "--bgen",
            str(self.bgen_file),
            "--out",
            output_prefix,
            "--export-adjusted-bgen",
            "-c",
            self.cov_file,
            "--bgi",
            str(self.bgi_file),
        ]

        main()

        # Check outputs
        adj_file = f"{output_prefix}.adj.bgen"
        metadata_file = f"{output_prefix}.adj.metadata.tsv.gz"
        self.assertTrue(os.path.exists(adj_file))
        self.assertTrue(os.path.exists(metadata_file))
        self.assertFalse(os.path.exists(f"{output_prefix}.ld"))

    def test_both_modes(self):
        """Test computing LD and exporting adjusted genotypes."""
        output_prefix = os.path.join(self.temp_dir, "both_modes")

        sys.argv = [
            "ldcov",
            "--bgen",
            str(self.bgen_file),
            "--out",
            output_prefix,
            "--compute-ld",
            "--export-adjusted-bgen",
            "-c",
            self.cov_file,
            "--bgi",
            str(self.bgi_file),
        ]

        main()

        # Check both outputs exist
        self.assertTrue(os.path.exists(f"{output_prefix}.ld"))
        self.assertTrue(os.path.exists(f"{output_prefix}.adj.bgen"))
        self.assertTrue(os.path.exists(f"{output_prefix}.adj.metadata.tsv.gz"))

    # ==================== Auto BGI Detection Tests ====================

    def test_auto_bgi_detection(self):
        """Test automatic BGI file detection."""
        # Create a temporary BGEN file with accompanying BGI
        temp_bgen = os.path.join(self.temp_dir, "test_auto.bgen")
        temp_bgi = os.path.join(self.temp_dir, "test_auto.bgen.bgi")

        # Copy files
        shutil.copy(str(self.bgen_file), temp_bgen)
        shutil.copy(str(self.bgi_file), temp_bgi)

        output_prefix = os.path.join(self.temp_dir, "auto_bgi")

        # Run without specifying --bgi
        sys.argv = [
            "ldcov",
            "--bgen",
            temp_bgen,
            "--out",
            output_prefix,
            "--compute-ld",
        ]

        # Should succeed using auto-detected BGI
        main()

        self.assertTrue(os.path.exists(f"{output_prefix}.ld"))

    def test_no_auto_bgi_available(self):
        """Test behavior when no BGI file exists."""
        # Create a temporary BGEN without BGI
        temp_bgen = os.path.join(self.temp_dir, "test_no_bgi.bgen")
        shutil.copy(str(self.bgen_file), temp_bgen)

        output_prefix = os.path.join(self.temp_dir, "no_bgi")

        sys.argv = [
            "ldcov",
            "--bgen",
            temp_bgen,
            "--out",
            output_prefix,
            "--compute-ld",
        ]

        # Should raise an error without BGI
        with self.assertRaises(SystemExit) as context:
            main()
        
        # Verify it exited with error code
        self.assertEqual(context.exception.code, 1)

    # ==================== Custom Covariate ID Column Tests ====================

    def test_custom_covariate_id_column(self):
        """Test using custom ID column in covariate file."""
        output_prefix = os.path.join(self.temp_dir, "custom_id")

        sys.argv = [
            "ldcov",
            "--bgen",
            str(self.bgen_file),
            "--out",
            output_prefix,
            "--compute-ld",
            "--export-adjusted-bgen",
            "-c",
            self.custom_cov_file,
            "--covariate-id-col",
            "FID",
            "--bgi",
            str(self.bgi_file),
        ]

        main()

        # Should complete successfully
        self.assertTrue(os.path.exists(f"{output_prefix}.ld"))
        self.assertTrue(os.path.exists(f"{output_prefix}.adj.bgen"))

    def test_missing_covariate_id_column(self):
        """Test error when specified ID column doesn't exist."""
        output_prefix = os.path.join(self.temp_dir, "missing_id_col")

        sys.argv = [
            "ldcov",
            "--bgen",
            str(self.bgen_file),
            "--out",
            output_prefix,
            "--compute-ld",
            "-c",
            self.cov_file,
            "--covariate-id-col",
            "NONEXISTENT",
            "--bgi",
            str(self.bgi_file),
        ]

        # Should raise an error
        with self.assertRaises(SystemExit):
            main()

    # ==================== Output Format Tests ====================

    def test_output_formats(self):
        """Test different LD output formats."""
        for fmt in ["matrix", "long", "bcor"]:
            output_prefix = os.path.join(self.temp_dir, f"format_{fmt}")

            sys.argv = [
                "ldcov",
                "--bgen",
                str(self.bgen_file),
                "--out",
                output_prefix,
                "--compute-ld",
                "--output-format",
                fmt,
                "--bgi",
                str(self.bgi_file),
            ]

            main()

            # Check appropriate file was created
            if fmt == "matrix":
                self.assertTrue(os.path.exists(f"{output_prefix}.ld"))
            elif fmt == "long":
                self.assertTrue(os.path.exists(f"{output_prefix}.ld.gz"))
            elif fmt == "bcor":
                self.assertTrue(os.path.exists(f"{output_prefix}.bcor"))

    # ==================== Region and Z-file Tests ====================

    def test_region_filtering(self):
        """Test region-based filtering."""
        # First load some data to find a valid region
        _, variant_info, _ = load_bgen(
            file_path=str(self.bgen_file),
            index_path=str(self.bgi_file),
        )

        # Use a region that contains at least some variants
        first_chrom = variant_info["chrom"].iloc[0]
        min_pos = variant_info["pos"].min()
        max_pos = variant_info["pos"].max()
        mid_pos = (min_pos + max_pos) // 2

        # Create a region that should contain some variants
        region = f"{first_chrom}:{min_pos}-{mid_pos}"

        output_prefix = os.path.join(self.temp_dir, "region_test")

        sys.argv = [
            "ldcov",
            "--bgen",
            str(self.bgen_file),
            "--out",
            output_prefix,
            "--compute-ld",
            "--region",
            region,
            "--bgi",
            str(self.bgi_file),
        ]

        main()

        self.assertTrue(os.path.exists(f"{output_prefix}.ld"))

    def test_z_file_filtering(self):
        """Test Z-file based variant filtering."""
        # Load some actual variants from the BGEN file to create a valid Z-file
        _, variant_info, _ = load_bgen(
            file_path=str(self.bgen_file),
            index_path=str(self.bgi_file),
        )

        # Use first 3 variants from BGEN file
        subset_variants = variant_info.iloc[:3]
        z_file = os.path.join(self.temp_dir, "test.z")
        z_data = pd.DataFrame(
            {
                "rsid": subset_variants["id"].tolist(),
                "chromosome": subset_variants["chrom"].tolist(),
                "position": subset_variants["pos"].astype(str).tolist(),
                "allele1": subset_variants["ref"].tolist(),
                "allele2": subset_variants["alt"].tolist(),
            }
        )
        z_data.to_csv(z_file, sep="\t", index=False)  # No compression

        output_prefix = os.path.join(self.temp_dir, "z_file_test")

        sys.argv = [
            "ldcov",
            "--bgen",
            str(self.bgen_file),
            "--out",
            output_prefix,
            "--compute-ld",
            "--z",
            z_file,
            "--bgi",
            str(self.bgi_file),
        ]

        main()

        self.assertTrue(os.path.exists(f"{output_prefix}.ld"))

    # ==================== Error Handling Tests ====================

    def test_no_mode_specified(self):
        """Test error when no mode is specified."""
        output_prefix = os.path.join(self.temp_dir, "no_mode")

        sys.argv = [
            "ldcov",
            "--bgen",
            str(self.bgen_file),
            "--out",
            output_prefix,
        ]

        with self.assertRaises(SystemExit):
            main()

    def test_export_adjusted_without_covariates(self):
        """Test error when exporting adjusted genotypes without covariates."""
        output_prefix = os.path.join(self.temp_dir, "no_cov_error")

        sys.argv = [
            "ldcov",
            "--bgen",
            str(self.bgen_file),
            "--out",
            output_prefix,
            "--export-adjusted-bgen",
        ]

        with self.assertRaises(SystemExit):
            main()

    def test_invalid_bgen_file(self):
        """Test error with invalid BGEN file."""
        output_prefix = os.path.join(self.temp_dir, "invalid_bgen")

        sys.argv = [
            "ldcov",
            "--bgen",
            "/nonexistent/file.bgen",
            "--out",
            output_prefix,
            "--compute-ld",
        ]

        with self.assertRaises(SystemExit):
            main()

    def test_whitespace_delimited_covariates(self):
        """Test loading whitespace-delimited covariate file."""
        # Create tab-delimited file (more reliable than space-delimited)
        # Use more samples to avoid rank deficiency issues
        ws_cov_file = os.path.join(self.temp_dir, "whitespace_cov.txt")
        with open(ws_cov_file, "w") as f:
            f.write("IID\tPC1\tPC2\n")
            # Use tab-delimited format - write data for more samples
            for i, sid in enumerate(self.sample_ids[:20]):  # Use 20 samples
                pc1 = 0.1 * (i + 1)
                pc2 = 0.2 * (i + 1)
                f.write(f"{sid}\t{pc1}\t{pc2}\n")

        output_prefix = os.path.join(self.temp_dir, "ws_cov")

        sys.argv = [
            "ldcov",
            "--bgen",
            str(self.bgen_file),
            "--out",
            output_prefix,
            "--compute-ld",
            "-c",
            ws_cov_file,
            "--bgi",
            str(self.bgi_file),
        ]

        # Should work with whitespace-delimited file
        main()

        self.assertTrue(os.path.exists(f"{output_prefix}.ld"))

    def test_verbose_mode(self):
        """Test verbose logging mode."""
        output_prefix = os.path.join(self.temp_dir, "verbose_test")

        sys.argv = [
            "ldcov",
            "--bgen",
            str(self.bgen_file),
            "--out",
            output_prefix,
            "--compute-ld",
            "--verbose",
            "--bgi",
            str(self.bgi_file),
        ]

        # Should complete without error
        main()

        self.assertTrue(os.path.exists(f"{output_prefix}.ld"))

    def test_specific_covariate_columns(self):
        """Test using specific columns as covariates."""
        # Create covariate file with multiple columns
        n_samples = len(self.sample_ids)
        np.random.seed(42)
        covariates_multi = pd.DataFrame(
            {
                "IID": self.sample_ids,
                "PC1": np.random.normal(0, 1, n_samples),
                "PC2": np.random.normal(0, 1, n_samples),
                "PC3": np.random.normal(0, 1, n_samples),
                "PC4": np.random.normal(0, 1, n_samples),
                "batch": np.random.choice(["A", "B", "C"], n_samples),
                "age": np.random.randint(20, 80, n_samples),
            }
        )
        multi_cov_file = os.path.join(self.temp_dir, "multi_covariates.csv")
        covariates_multi.to_csv(multi_cov_file, index=False)

        output_prefix = os.path.join(self.temp_dir, "specific_cols")

        # Use only PC1 and PC2 as covariates
        sys.argv = [
            "ldcov",
            "--bgen",
            str(self.bgen_file),
            "--out",
            output_prefix,
            "--compute-ld",
            "-c",
            multi_cov_file,
            "--covariate-cols",
            "PC1",
            "PC2",
            "--bgi",
            str(self.bgi_file),
        ]

        main()

        # Should complete successfully
        self.assertTrue(os.path.exists(f"{output_prefix}.ld"))

    def test_invalid_covariate_columns(self):
        """Test error when specifying non-existent covariate columns."""
        output_prefix = os.path.join(self.temp_dir, "invalid_cols")

        sys.argv = [
            "ldcov",
            "--bgen",
            str(self.bgen_file),
            "--out",
            output_prefix,
            "--compute-ld",
            "-c",
            self.cov_file,
            "--covariate-cols",
            "PC1",
            "NonExistentColumn",
            "--bgi",
            str(self.bgi_file),
        ]

        # Should raise an error
        with self.assertRaises(SystemExit):
            main()


if __name__ == "__main__":
    unittest.main()
