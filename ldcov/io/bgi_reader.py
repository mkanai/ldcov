"""
BGI (BGEN Index) file reader for efficient access to variant metadata.

This module provides a pandas-based reader for BGI files,
optimized for ldcov's specific needs with improved efficiency and readability.
"""

import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BGIReader:
    """
    Reader for BGEN index (BGI) files.

    BGI files are SQLite databases created by bgenix that contain
    variant metadata and file offsets for efficient BGEN access.
    """

    def __init__(self, bgi_path: str):
        """
        Initialize BGI reader.

        Parameters
        ----------
        bgi_path : str
            Path to BGI index file

        Raises
        ------
        FileNotFoundError
            If BGI file doesn't exist
        ValueError
            If BGI file is invalid
        """
        self.bgi_path = Path(bgi_path)
        if not self.bgi_path.exists():
            raise FileNotFoundError(f"BGI index file not found: {bgi_path}")

        logger.debug(f"Opening BGI index: {bgi_path}")

        # Open SQLite connection
        self.conn = sqlite3.connect(str(self.bgi_path))
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self.cursor = self.conn.cursor()

        # Verify BGI structure
        self._verify_bgi_structure()

        # Cache for frequently accessed data
        self._variant_count = None

    def _verify_bgi_structure(self):
        """Verify this is a valid BGI file with expected tables."""
        try:
            tables = self.cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = {t["name"] for t in tables}

            if "Variant" not in table_names or "Metadata" not in table_names:
                raise ValueError(f"Invalid BGI file: missing required tables")

            # Verify Variant table has expected columns
            columns = self.cursor.execute("PRAGMA table_info(Variant)").fetchall()
            column_names = {c["name"] for c in columns}

            required_columns = {
                "chromosome",
                "position",
                "rsid",
                "number_of_alleles",
                "allele1",
                "allele2",
                "file_start_position",
                "size_in_bytes",
            }

            missing = required_columns - column_names
            if missing:
                raise ValueError(f"Invalid BGI file: missing columns {missing}")

        except sqlite3.Error as e:
            raise ValueError(f"Error reading BGI file: {e}")

    def get_variant_count(self) -> int:
        """Get total number of variants in the index."""
        if self._variant_count is None:
            self._variant_count = self.cursor.execute("SELECT COUNT(*) FROM Variant").fetchone()[0]
        return self._variant_count

    def get_all_variants(self) -> pd.DataFrame:
        """
        Get metadata for all variants.

        Returns
        -------
        pd.DataFrame
            DataFrame with variant metadata
        """
        query = """
        SELECT chromosome as chrom, position as pos, rsid, number_of_alleles as n_alleles, 
               allele1 as ref, allele2 as alt, file_start_position as file_offset, 
               size_in_bytes as size_bytes
        FROM Variant 
        ORDER BY file_start_position
        """

        # Use pandas to read directly from SQL
        df = pd.read_sql_query(query, self.conn)
        # Handle NULL alt alleles
        df["alt"] = df["alt"].fillna("")
        return df

    def get_variants_in_region(self, chrom: str, start: int, end: int) -> pd.DataFrame:
        """
        Get metadata for variants in a genomic region.

        Parameters
        ----------
        chrom : str
            Chromosome
        start : int
            Start position (inclusive)
        end : int
            End position (inclusive)

        Returns
        -------
        pd.DataFrame
            DataFrame with variant metadata
        """
        query = """
        SELECT chromosome as chrom, position as pos, rsid, number_of_alleles as n_alleles,
               allele1 as ref, allele2 as alt, file_start_position as file_offset,
               size_in_bytes as size_bytes
        FROM Variant 
        WHERE chromosome = :chrom AND position >= :start AND position <= :end
        ORDER BY position
        """

        # Use pandas with named parameters
        df = pd.read_sql_query(
            query, self.conn, params={"chrom": chrom, "start": start, "end": end}
        )
        df["alt"] = df["alt"].fillna("")
        return df

    def find_variants_by_filter(
        self, chromosome: str, positions: np.ndarray, alleles1: List[str], alleles2: List[str]
    ) -> pd.DataFrame:
        """
        Find variants matching chromosome, position, and allele combinations.

        This method matches variants exactly based on chromosome, position,
        allele1, and allele2. No allele swapping is performed.

        Parameters
        ----------
        chromosome : str
            Chromosome to filter on (required to prevent cross-chromosome matches)
        positions : np.ndarray
            Positions to match
        alleles1 : List[str]
            First alleles (must match exactly)
        alleles2 : List[str]
            Second alleles (must match exactly)

        Returns
        -------
        pd.DataFrame
            DataFrame with matched variants in order found
        """
        # Convert inputs to DataFrame for efficient operations
        filter_df = pd.DataFrame(
            {
                "position": positions.astype(np.int32),
                "allele1": alleles1,
                "allele2": alleles2,
                "original_idx": np.arange(len(positions)),
            }
        )

        # Get unique positions for efficient querying
        unique_positions = filter_df["position"].unique()

        # Process in batches to avoid SQL query limits
        batch_size = 1000
        all_db_variants = []

        for i in range(0, len(unique_positions), batch_size):
            batch_positions = unique_positions[i : i + batch_size]

            # Create parameterized query with exact chromosome matching
            placeholders = ",".join(["?"] * len(batch_positions))
            query = f"""
            SELECT chromosome, position, rsid, number_of_alleles,
                   allele1, allele2, file_start_position, size_in_bytes
            FROM Variant
            WHERE chromosome = ? AND position IN ({placeholders})
            """
            params = [chromosome] + batch_positions.tolist()

            # Read batch from database
            batch_df = pd.read_sql_query(query, self.conn, params=params)
            all_db_variants.append(batch_df)

        # Combine all batches
        if all_db_variants:
            db_df = pd.concat(all_db_variants, ignore_index=True)

            # Handle NULL allele2 values
            db_df["allele2"] = db_df["allele2"].fillna("")

            # Exact match on position, allele1, and allele2
            # Chromosome is already filtered in the SQL query if provided
            merged = filter_df.merge(
                db_df,
                left_on=["position", "allele1", "allele2"],
                right_on=["position", "allele1", "allele2"],
                how="left",
                suffixes=("_filter", "_db"),
            )

            # Filter to only matched variants
            matched = merged[merged["chromosome"].notna()].copy()

            if not matched.empty:
                # Sort by original index to maintain order
                matched = matched.sort_values("original_idx")

                # Return DataFrame with renamed columns to match expected format
                matched_variants = matched[
                    [
                        "chromosome",
                        "position",
                        "rsid",
                        "number_of_alleles",
                        "allele1",
                        "allele2",
                        "file_start_position",
                        "size_in_bytes",
                    ]
                ].copy()
                matched_variants.columns = [
                    "chrom",
                    "pos",
                    "rsid",
                    "n_alleles",
                    "ref",
                    "alt",
                    "file_offset",
                    "size_bytes",
                ]

                return matched_variants

        # Return empty DataFrame with correct columns
        empty_df = pd.DataFrame(
            columns=["chrom", "pos", "rsid", "n_alleles", "ref", "alt", "file_offset", "size_bytes"]
        )
        return empty_df

    def get_file_offsets_by_indices(self, indices: List[int]) -> np.ndarray:
        """
        Get file offsets for specific variant indices.

        Parameters
        ----------
        indices : List[int]
            Variant indices (0-based)

        Returns
        -------
        np.ndarray
            File offsets
        """
        # Build query with placeholders
        placeholders = ",".join("?" * len(indices))
        query = f"""
        SELECT file_start_position
        FROM (SELECT file_start_position, ROW_NUMBER() OVER (ORDER BY file_start_position) - 1 as idx FROM Variant)
        WHERE idx IN ({placeholders})
        """

        rows = self.cursor.execute(query, indices).fetchall()
        return np.array([r[0] for r in rows], dtype=np.int64)

    def close(self):
        """Close the database connection."""
        if hasattr(self, "cursor") and self.cursor:
            self.cursor.close()
            self.cursor = None
        if hasattr(self, "conn") and self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        """Ensure connection is closed on deletion."""
        self.close()
