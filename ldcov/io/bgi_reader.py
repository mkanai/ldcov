"""
BGI (BGEN Index) file reader for efficient access to variant metadata.

This module provides a custom SQLite-based reader for BGI files,
optimized for ldcov's specific needs.
"""

import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np

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
            table_names = {t['name'] for t in tables}
            
            if 'Variant' not in table_names or 'Metadata' not in table_names:
                raise ValueError(f"Invalid BGI file: missing required tables")
                
            # Verify Variant table has expected columns
            columns = self.cursor.execute("PRAGMA table_info(Variant)").fetchall()
            column_names = {c['name'] for c in columns}
            
            required_columns = {
                'chromosome', 'position', 'rsid', 'number_of_alleles',
                'allele1', 'allele2', 'file_start_position', 'size_in_bytes'
            }
            
            missing = required_columns - column_names
            if missing:
                raise ValueError(f"Invalid BGI file: missing columns {missing}")
                
        except sqlite3.Error as e:
            raise ValueError(f"Error reading BGI file: {e}")
    
    def get_variant_count(self) -> int:
        """Get total number of variants in the index."""
        if self._variant_count is None:
            self._variant_count = self.cursor.execute(
                "SELECT COUNT(*) FROM Variant"
            ).fetchone()[0]
        return self._variant_count
    
    def get_all_variants(self) -> np.ndarray:
        """
        Get metadata for all variants.
        
        Returns
        -------
        np.ndarray
            Structured array with variant metadata
        """
        query = """
        SELECT chromosome, position, rsid, number_of_alleles, 
               allele1, allele2, file_start_position, size_in_bytes
        FROM Variant 
        ORDER BY file_start_position
        """
        
        rows = self.cursor.execute(query).fetchall()
        return self._rows_to_structured_array(rows)
    
    def get_variants_in_region(self, chrom: str, start: int, end: int) -> np.ndarray:
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
        np.ndarray
            Structured array with variant metadata
        """
        query = """
        SELECT chromosome, position, rsid, number_of_alleles,
               allele1, allele2, file_start_position, size_in_bytes
        FROM Variant 
        WHERE chromosome = ? AND position >= ? AND position <= ?
        ORDER BY position
        """
        
        rows = self.cursor.execute(query, (chrom, start, end)).fetchall()
        return self._rows_to_structured_array(rows)
    
    def find_variants_by_filter(
        self, 
        positions: np.ndarray,
        alleles1: List[str],
        alleles2: List[str],
        rsids: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find variants matching position/allele/rsid combinations.
        
        This method is optimized for matching variants from .z files,
        handling allele order ambiguity.
        
        Parameters
        ----------
        positions : np.ndarray
            Positions to match
        alleles1 : List[str]
            First alleles
        alleles2 : List[str]
            Second alleles  
        rsids : List[str]
            RSIDs to match
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (matched_variants, original_indices)
        """
        # Get all variants for matching
        all_query = """
        SELECT chromosome, position, rsid, number_of_alleles,
               allele1, allele2, file_start_position, size_in_bytes
        FROM Variant
        """
        all_rows = self.cursor.execute(all_query).fetchall()
        
        # Build lookup structures
        pos_allele_map = {}  # (pos, allele1_sorted, allele2_sorted) -> row
        rsid_map = {}  # rsid -> row
        
        for row in all_rows:
            # Handle allele order by sorting
            a1, a2 = row['allele1'], row['allele2'] or ''
            alleles_key = tuple(sorted([a1, a2]))
            pos_key = (row['position'], alleles_key)
            pos_allele_map[pos_key] = row
            
            # Also index by rsid
            rsid_map[row['rsid']] = row
        
        # Match variants
        matched_rows = []
        matched_indices = []
        
        for i, (pos, a1, a2, rsid) in enumerate(zip(positions, alleles1, alleles2, rsids)):
            # Try position + alleles match first
            alleles_key = tuple(sorted([a1, a2]))
            pos_key = (int(pos), alleles_key)
            
            if pos_key in pos_allele_map:
                matched_rows.append(pos_allele_map[pos_key])
                matched_indices.append(i)
            elif rsid in rsid_map:
                # Fall back to rsid match
                matched_rows.append(rsid_map[rsid])
                matched_indices.append(i)
        
        # Convert to structured arrays
        if matched_rows:
            matched_variants = self._rows_to_structured_array(matched_rows)
            return matched_variants, np.array(matched_indices, dtype=np.int32)
        else:
            # Return empty arrays with correct structure
            return self._empty_variant_array(), np.array([], dtype=np.int32)
    
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
        placeholders = ','.join('?' * len(indices))
        query = f"""
        SELECT file_start_position
        FROM (SELECT file_start_position, ROW_NUMBER() OVER (ORDER BY file_start_position) - 1 as idx FROM Variant)
        WHERE idx IN ({placeholders})
        """
        
        rows = self.cursor.execute(query, indices).fetchall()
        return np.array([r[0] for r in rows], dtype=np.int64)
    
    def _rows_to_structured_array(self, rows: List[sqlite3.Row]) -> np.ndarray:
        """Convert SQLite rows to numpy structured array."""
        if not rows:
            return self._empty_variant_array()
        
        # Define dtype for structured array
        dtype = [
            ('chrom', 'U10'),
            ('pos', np.int32),
            ('rsid', 'U50'),
            ('n_alleles', np.int32),
            ('ref', 'U100'),
            ('alt', 'U100'),
            ('file_offset', np.int64),
            ('size_bytes', np.int32),
        ]
        
        # Create array
        n_variants = len(rows)
        arr = np.empty(n_variants, dtype=dtype)
        
        # Fill array
        for i, row in enumerate(rows):
            arr[i] = (
                row['chromosome'],
                row['position'],
                row['rsid'],
                row['number_of_alleles'],
                row['allele1'],
                row['allele2'] or '',
                row['file_start_position'],
                row['size_in_bytes']
            )
        
        return arr
    
    def _empty_variant_array(self) -> np.ndarray:
        """Create empty structured array with correct dtype."""
        dtype = [
            ('chrom', 'U10'),
            ('pos', np.int32),
            ('rsid', 'U50'),
            ('n_alleles', np.int32),
            ('ref', 'U100'),
            ('alt', 'U100'),
            ('file_offset', np.int64),
            ('size_bytes', np.int32),
        ]
        return np.empty(0, dtype=dtype)
    
    def close(self):
        """Close the database connection."""
        if hasattr(self, 'cursor') and self.cursor:
            self.cursor.close()
            self.cursor = None
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __del__(self):
        """Ensure connection is closed on deletion."""
        self.close()