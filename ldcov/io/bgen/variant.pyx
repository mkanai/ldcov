# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION
# BGEN variant parsing module

import struct
import numpy as np
cimport numpy as np
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from libc.string cimport memcpy
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

from ._bgen cimport VariantInfo, GenotypeData, PyMem_Malloc, PyMem_Free

np.import_array()


cdef class BgenVariant:
    """Represents a single variant from a BGEN file."""
    
    cdef VariantInfo info
    cdef object file_handle
    cdef int layout
    cdef int compression
    cdef uint32_t expected_n_samples
    cdef GenotypeData _genotype_data
    cdef bool _genotypes_loaded
    cdef object _preloaded_genotype_data
    cdef bool _skip_metadata_parsing
    
    def __init__(self, file_handle, uint64_t offset, int layout, int compression, uint32_t n_samples, 
                 preloaded_genotype_data=None, variant_metadata=None):
        """Initialize variant from file at given offset or from preloaded data."""
        self.file_handle = file_handle
        self.layout = layout
        self.compression = compression
        self.expected_n_samples = n_samples
        self._genotypes_loaded = False
        self._preloaded_genotype_data = preloaded_genotype_data
        self._skip_metadata_parsing = variant_metadata is not None
        
        if variant_metadata is not None:
            # Use provided metadata (from BGI) - skip file parsing
            self._init_from_metadata(variant_metadata, offset)
        else:
            # Original path - seek to variant position and read metadata
            file_handle.seek(offset)
            self.parse_variant_block()
    
    cdef void parse_variant_block(self) except *:
        """Parse variant metadata block."""
        cdef uint32_t variant_data_length
        cdef uint16_t varid_length, rsid_length, chrom_length
        cdef uint32_t n_alleles
        cdef uint32_t allele_length
        cdef uint32_t n_samples_from_variant
        cdef bytes data
        cdef int pos = 0
        
        # Store file offset
        self.info.file_offset = self.file_handle.tell()
        
        if self.layout == 1:  # Layout 1 (v1.1)
            # Read number of samples (4 bytes) - v1.1 specific
            data = self.file_handle.read(4)
            if len(data) < 4:
                raise ValueError("Incomplete variant data")
            n_samples_from_variant = struct.unpack('<I', data)[0]
            
            # Verify sample count matches
            if n_samples_from_variant != self.expected_n_samples:
                raise ValueError(f"Sample count mismatch: variant has {n_samples_from_variant}, expected {self.expected_n_samples}")
            
            # Read variant ID length and string
            data = self.file_handle.read(2)
            varid_length = struct.unpack('<H', data)[0]
            self.info.varid = self.file_handle.read(varid_length) if varid_length > 0 else b''
            
            # Read rsID length and string  
            data = self.file_handle.read(2)
            rsid_length = struct.unpack('<H', data)[0]
            self.info.rsid = self.file_handle.read(rsid_length) if rsid_length > 0 else b''
            
            # Read chromosome length and string
            data = self.file_handle.read(2)
            chrom_length = struct.unpack('<H', data)[0]
            self.info.chrom = self.file_handle.read(chrom_length) if chrom_length > 0 else b''
            
            # Read position (4 bytes)
            data = self.file_handle.read(4)
            self.info.pos = struct.unpack('<I', data)[0]
            
            # v1.1 is always biallelic
            self.info.n_alleles = 2
            
            # Read alleles
            self.info.alleles.clear()
            for i in range(2):
                data = self.file_handle.read(4)
                allele_length = struct.unpack('<I', data)[0]
                allele = self.file_handle.read(allele_length) if allele_length > 0 else b''
                self.info.alleles.push_back(allele)
            
            # Calculate genotype data length
            if self.compression == 0:
                # Uncompressed BGEN is not supported
                raise RuntimeError(
                    "Uncompressed BGEN files are not supported. "
                    "Please use compressed BGEN files (zlib or zstd). "
                    "You can compress your BGEN file using bgenix or qctool2."
                )
            else:
                # Compressed: read the length
                data = self.file_handle.read(4)
                self.info.geno_length = struct.unpack('<I', data)[0]
            
            # Store genotype offset (current position)
            self.info.geno_offset = self.file_handle.tell()
            
            # Skip genotype data for now (lazy loading)
            self.file_handle.seek(self.info.geno_offset + self.info.geno_length)
            
        elif self.layout == 2:  # Layout 2 (v1.2)
            # v1.2 format does NOT have a variant data block length
            # It goes straight to variant ID
            
            # Read variant ID length and string
            data = self.file_handle.read(2)
            varid_length = struct.unpack('<H', data)[0]
            self.info.varid = self.file_handle.read(varid_length) if varid_length > 0 else b''
            
            # Read rsID length and string  
            data = self.file_handle.read(2)
            rsid_length = struct.unpack('<H', data)[0]
            self.info.rsid = self.file_handle.read(rsid_length) if rsid_length > 0 else b''
            
            # Read chromosome length and string
            data = self.file_handle.read(2)
            chrom_length = struct.unpack('<H', data)[0]
            self.info.chrom = self.file_handle.read(chrom_length) if chrom_length > 0 else b''
            
            # Read position (4 bytes)
            data = self.file_handle.read(4)
            self.info.pos = struct.unpack('<I', data)[0]
            
            # Read number of alleles
            data = self.file_handle.read(2)
            self.info.n_alleles = struct.unpack('<H', data)[0]
            
            # Read each allele
            self.info.alleles.clear()
            for i in range(self.info.n_alleles):
                data = self.file_handle.read(4)
                if len(data) < 4:
                    raise ValueError(f"Incomplete allele length data for allele {i}")
                allele_length = struct.unpack('<I', data)[0]
                allele = self.file_handle.read(allele_length) if allele_length > 0 else b''
                self.info.alleles.push_back(allele)
            
            # Read genotype data block length
            data = self.file_handle.read(4)
            self.info.geno_length = struct.unpack('<I', data)[0]
            
            # Store genotype offset (current position)
            self.info.geno_offset = self.file_handle.tell()
            
            # Skip genotype data for now (lazy loading)
            self.file_handle.seek(self.info.geno_offset + self.info.geno_length)
            
        else:
            raise ValueError(f"Unsupported BGEN layout version: {self.layout}")
    
    cdef void _init_from_metadata(self, variant_metadata, uint64_t offset) except *:
        """Initialize variant from provided metadata dictionary (from BGI)."""
        # Store file offset
        self.info.file_offset = offset
        
        # Convert metadata to variant info structure
        self.info.varid = str(variant_metadata.get('varid', '')).encode('utf-8')
        self.info.rsid = str(variant_metadata.get('rsid', '')).encode('utf-8')  
        self.info.chrom = str(variant_metadata.get('chrom', '')).encode('utf-8')
        self.info.pos = int(variant_metadata.get('pos', 0))
        
        # Handle alleles - assume biallelic for now
        ref_allele = str(variant_metadata.get('ref', ''))
        alt_allele = str(variant_metadata.get('alt', ''))
        
        self.info.alleles.clear()
        self.info.alleles.push_back(ref_allele.encode('utf-8'))
        self.info.alleles.push_back(alt_allele.encode('utf-8'))
        self.info.n_alleles = 2
        
        # For preloaded data, we don't need file offsets since we have the data
        self.info.geno_offset = 0
        self.info.geno_length = 0
    
    @property
    def varid(self):
        return self.info.varid.decode('utf-8', errors='replace')
    
    @property
    def rsid(self):
        return self.info.rsid.decode('utf-8', errors='replace')
    
    @property
    def chrom(self):
        return self.info.chrom.decode('utf-8', errors='replace')
    
    @property
    def pos(self):
        return self.info.pos
    
    @property
    def alleles(self):
        return [a.decode('utf-8', errors='replace') for a in self.info.alleles]
    
    @property
    def n_alleles(self):
        return self.info.n_alleles
    
    @property
    def alt_dosage(self):
        """Get alt allele dosage for all samples."""
        if not self._genotypes_loaded:
            self._load_genotypes()
        
        # Create output array
        cdef np.ndarray[np.float32_t, ndim=1] dosages = np.zeros(self.expected_n_samples, dtype=np.float32)
        
        # Compute dosages
        self._genotype_data.compute_dosages(<float*>np.PyArray_DATA(dosages))
        
        return dosages
    
    def get_alt_dosage_filtered(self, sample_indices):
        """
        Get alt allele dosage for specified sample indices only.
        
        Parameters
        ----------
        sample_indices : array-like of int
            Indices of samples to extract dosages for
            
        Returns
        -------
        np.ndarray
            Dosages for the specified samples only
        """
        if not self._genotypes_loaded:
            self._load_genotypes()
        
        # Convert to numpy array if needed
        cdef np.ndarray[np.int32_t, ndim=1] indices = np.asarray(sample_indices, dtype=np.int32)
        cdef int n_indices = np.PyArray_DIM(indices, 0)
        
        # Create output array matching the number of requested samples
        cdef np.ndarray[np.float32_t, ndim=1] dosages = np.zeros(n_indices, dtype=np.float32)
        
        # Compute dosages only for requested samples
        self._genotype_data.compute_dosages_filtered(<int*>np.PyArray_DATA(indices), n_indices, <float*>np.PyArray_DATA(dosages))
        
        return dosages
    
    def get_dosage_for_samples(self, sample_indices=None):
        """
        Get alt allele dosage, optionally filtering to specific samples.
        
        This method automatically uses the efficient filtered computation when
        sample indices are provided.
        
        Parameters
        ----------
        sample_indices : array-like of int, optional
            Indices of samples to extract. If None, returns all samples.
            
        Returns
        -------
        np.ndarray
            Dosages for the requested samples
        """
        if sample_indices is not None:
            return self.get_alt_dosage_filtered(sample_indices)
        else:
            return self.alt_dosage
    
    @property
    def probabilities(self):
        """Get genotype probabilities."""
        if not self._genotypes_loaded:
            self._load_genotypes()
        
        # For diploid biallelic, we have 3 probabilities per sample
        cdef int n_probs = 3 if self.info.n_alleles == 2 else self.info.n_alleles * (self.info.n_alleles + 1) // 2
        cdef np.ndarray[np.float32_t, ndim=2] probs = np.zeros((self.expected_n_samples, n_probs), dtype=np.float32)
        
        # Copy probability data
        if self._genotype_data.probs != NULL:
            memcpy(np.PyArray_DATA(probs), self._genotype_data.probs, 
                   self.expected_n_samples * n_probs * sizeof(float))
        
        return probs
    
    cdef void _load_genotypes(self) except *:
        """Load and parse genotype data."""
        if self._genotypes_loaded:
            return
        
        cdef bytes geno_data
        
        if self._preloaded_genotype_data is not None:
            # Use pre-decompressed data - skip file I/O entirely
            if isinstance(self._preloaded_genotype_data, np.ndarray):
                # Convert numpy array to bytes
                geno_data = self._preloaded_genotype_data.tobytes()
            else:
                # Already bytes
                geno_data = self._preloaded_genotype_data
        else:
            # Original path - read from file
            self.file_handle.seek(self.info.geno_offset)
            geno_data = self.file_handle.read(self.info.geno_length)
            if len(geno_data) < self.info.geno_length:
                raise ValueError("Incomplete genotype data")
        
        # Initialize genotype data handler
        self._genotype_data = GenotypeData()
        self._genotype_data.n_samples = self.expected_n_samples
        self._genotype_data.n_alleles = self.info.n_alleles
        
        # Set compression type - if using preloaded data, it's already decompressed
        if self._preloaded_genotype_data is not None:
            self._genotype_data.compression = 0  # Treat as uncompressed since C++ already decompressed it
        else:
            self._genotype_data.compression = self.compression
        
        # Parse the genotype data based on layout
        # Note: parse_layout methods will allocate and store the raw data internally
        if self.layout == 1:
            self._genotype_data.parse_layout1(<uint8_t*>geno_data, len(geno_data))
        elif self.layout == 2:
            self._genotype_data.parse_layout2(<uint8_t*>geno_data, len(geno_data))
        else:
            raise ValueError(f"Unsupported layout version: {self.layout}")
        
        self._genotypes_loaded = True
    
    def __dealloc__(self):
        """Clean up allocated memory."""
        # GenotypeData will clean up its own memory in its __dealloc__ method
        pass
    
    def __repr__(self):
        return f'BgenVariant("{self.varid}", "{self.rsid}", "{self.chrom}", {self.pos}, {self.alleles})'