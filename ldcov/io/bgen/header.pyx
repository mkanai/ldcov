# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
# BGEN header parsing module

import struct
from libc.stdint cimport uint32_t, uint16_t
from libc.string cimport memcpy

from ._bgen cimport HeaderInfo, BGEN_COMPRESSED_ZLIB, BGEN_COMPRESSED_ZSTD, BGEN_LAYOUT_V11, BGEN_LAYOUT_V12


cdef class BgenHeader:
    """Parse and store BGEN file header information."""
    
    cdef public HeaderInfo info
    cdef bytes extra_data
    
    def __init__(self, file_handle):
        """Initialize header by reading from file handle."""
        self.parse_header(file_handle)
    
    cdef void parse_header(self, object file_handle) except *:
        """Parse BGEN header from file."""
        cdef bytes header_data
        cdef uint32_t header_length
        cdef uint32_t flags
        cdef uint32_t offset_to_variants
        
        # First, read just the offset and header length to determine version
        initial_data = file_handle.read(8)
        if len(initial_data) < 8:
            raise ValueError("Invalid BGEN file: could not read initial header")
        
        offset_to_variants = struct.unpack('<I', initial_data[0:4])[0]
        header_length = struct.unpack('<I', initial_data[4:8])[0]
        
        # Read the rest of the header based on header length
        remaining_header = file_handle.read(header_length - 8)
        if len(remaining_header) < header_length - 8:
            raise ValueError("Invalid BGEN file: incomplete header")
        
        # Combine for easier parsing
        header_data = initial_data + remaining_header
        
        # Now check if this is v1.1 or v1.2 by looking for magic number
        # In v1.2, magic "bgen" is at bytes 16-19
        # In v1.1, there's no magic number
        pos = 8  # Start after offset and header_length
        
        self.info.nvariants = struct.unpack('<I', header_data[pos:pos+4])[0]
        pos += 4
        
        self.info.nsamples = struct.unpack('<I', header_data[pos:pos+4])[0]
        pos += 4
        
        # Check if we have the magic number (v1.2) or not (v1.1)
        if header_length >= 20:
            magic = header_data[pos:pos+4]
            if magic == b'bgen':
                # v1.2 format - has magic number
                pos += 4
                # Read any extra data in header
                if header_length > 20:
                    self.extra_data = header_data[20:header_length]
                else:
                    self.extra_data = b''
            else:
                # v1.1 format - no magic number
                # The next data is already part of extra data
                if header_length > 16:
                    self.extra_data = header_data[16:header_length]
                else:
                    self.extra_data = b''
        else:
            # v1.1 format with minimal header
            self.extra_data = b''
        
        # Read flags (4 bytes after header)
        flags_data = file_handle.read(4)
        if len(flags_data) < 4:
            raise ValueError("Invalid BGEN file: could not read flags")
        
        self.info.flags = struct.unpack('<I', flags_data)[0]
        
        # Store the offset to variant data
        self.info.offset = offset_to_variants
        
        # Decode flags
        compression_flag = self.info.flags & 3
        if compression_flag == 0:
            self.info.compression = 0
        elif compression_flag == 1:
            self.info.compression = BGEN_COMPRESSED_ZLIB
        elif compression_flag == 2:
            self.info.compression = BGEN_COMPRESSED_ZSTD
        else:
            raise ValueError(f"Unknown compression type: {compression_flag}")
        
        layout_flag = (self.info.flags >> 2) & 15
        if layout_flag == 1:
            self.info.layout = BGEN_LAYOUT_V11
        elif layout_flag == 2:
            self.info.layout = BGEN_LAYOUT_V12
        else:
            raise ValueError(f"Unsupported layout version: {layout_flag}")
        
        self.info.has_sample_ids = bool((self.info.flags >> 31) & 1)
    
    @property
    def offset(self):
        return self.info.offset
    
    @property
    def nvariants(self):
        return self.info.nvariants
    
    @property
    def nsamples(self):
        return self.info.nsamples
    
    @property
    def compression(self):
        if self.info.compression == 0:
            return None
        elif self.info.compression == BGEN_COMPRESSED_ZLIB:
            return 'zlib'
        elif self.info.compression == BGEN_COMPRESSED_ZSTD:
            return 'zstd'
        else:
            return 'unknown'
    
    @property
    def layout(self):
        return self.info.layout
    
    @property
    def has_sample_ids(self):
        return self.info.has_sample_ids
    
    def __repr__(self):
        return (f"BgenHeader(nvariants={self.nvariants}, nsamples={self.nsamples}, "
                f"compression={self.compression}, layout={self.layout}, "
                f"has_sample_ids={self.has_sample_ids})")