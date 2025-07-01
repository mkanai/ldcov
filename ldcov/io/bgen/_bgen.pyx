# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION
# distutils: language = c++
# Implementation of BGEN data structures

import struct
import numpy as np
cimport numpy as np
from libc.stdint cimport uint8_t, uint16_t, uint32_t
from libc.string cimport memcpy
from libc.math cimport NAN

# Prefetch intrinsic for better cache utilization
cdef extern from *:
    """
    #ifdef __GNUC__
        static inline void prefetch(const void* addr) {
            __builtin_prefetch(addr, 0, 1);
        }
    #else
        static inline void prefetch(const void* addr) {
            (void)addr;  // No-op on non-GCC compilers
        }
    #endif
    """
    void prefetch(const void* addr) nogil

# SIMD intrinsics for dosage computation
cdef extern from *:
    """
    // Platform-specific SIMD implementations
    #if defined(__x86_64__) || defined(_M_X64)
        #include <immintrin.h>
    #elif defined(__aarch64__) || defined(__arm__)
        #include <arm_neon.h>
    #endif
    
    void compute_dosages_simd(const float* probs, float* output, int n_samples) {
        int i;
        
        #if defined(__AVX2__)
        // x86_64 with AVX2: Process 8 samples at a time (256-bit registers)
        const __m256 two = _mm256_set1_ps(2.0f);
        
        for (i = 0; i <= n_samples - 8; i += 8) {
            // Load P(AB) values for 8 samples
            __m256 p_ab = _mm256_set_ps(
                probs[(i+7)*3 + 1], probs[(i+6)*3 + 1], probs[(i+5)*3 + 1], probs[(i+4)*3 + 1],
                probs[(i+3)*3 + 1], probs[(i+2)*3 + 1], probs[(i+1)*3 + 1], probs[i*3 + 1]
            );
            
            // Load P(BB) values for 8 samples
            __m256 p_bb = _mm256_set_ps(
                probs[(i+7)*3 + 2], probs[(i+6)*3 + 2], probs[(i+5)*3 + 2], probs[(i+4)*3 + 2],
                probs[(i+3)*3 + 2], probs[(i+2)*3 + 2], probs[(i+1)*3 + 2], probs[i*3 + 2]
            );
            
            // Compute dosage = P(AB) + 2*P(BB) using FMA
            __m256 dosage = _mm256_fmadd_ps(two, p_bb, p_ab);
            
            // Store results
            _mm256_storeu_ps(&output[i], dosage);
        }
        
        #elif defined(__ARM_NEON) || defined(__aarch64__)
        // ARM NEON: Process 4 samples at a time (128-bit registers)
        const float32x4_t two = vdupq_n_f32(2.0f);
        
        for (i = 0; i <= n_samples - 4; i += 4) {
            // Load P(AB) values for 4 samples
            float32x4_t p_ab = {probs[i*3 + 1], probs[(i+1)*3 + 1], 
                               probs[(i+2)*3 + 1], probs[(i+3)*3 + 1]};
            
            // Load P(BB) values for 4 samples
            float32x4_t p_bb = {probs[i*3 + 2], probs[(i+1)*3 + 2], 
                               probs[(i+2)*3 + 2], probs[(i+3)*3 + 2]};
            
            // Compute dosage = P(AB) + 2*P(BB)
            float32x4_t dosage = vmlaq_f32(p_ab, p_bb, two);
            
            // Store results
            vst1q_f32(&output[i], dosage);
        }
        
        #else
        // No SIMD: use scalar code for entire loop
        i = 0;
        #endif
        
        // Handle remaining samples with scalar code
        for (; i < n_samples; i++) {
            output[i] = probs[i*3 + 1] + 2.0f * probs[i*3 + 2];
        }
    }
    
    void compute_dosages_filtered_simd(const float* probs, const int* indices, 
                                                    float* output, int n_indices) {
        int j;
        
        #if defined(__AVX2__)
        // x86_64 with AVX2: Process 8 samples at a time
        const __m256 two = _mm256_set1_ps(2.0f);
        
        for (j = 0; j <= n_indices - 8; j += 8) {
            // Gather P(AB) values for 8 selected samples
            __m256 p_ab = _mm256_set_ps(
                probs[indices[j+7]*3 + 1], probs[indices[j+6]*3 + 1], 
                probs[indices[j+5]*3 + 1], probs[indices[j+4]*3 + 1],
                probs[indices[j+3]*3 + 1], probs[indices[j+2]*3 + 1], 
                probs[indices[j+1]*3 + 1], probs[indices[j]*3 + 1]
            );
            
            // Gather P(BB) values for 8 selected samples
            __m256 p_bb = _mm256_set_ps(
                probs[indices[j+7]*3 + 2], probs[indices[j+6]*3 + 2], 
                probs[indices[j+5]*3 + 2], probs[indices[j+4]*3 + 2],
                probs[indices[j+3]*3 + 2], probs[indices[j+2]*3 + 2], 
                probs[indices[j+1]*3 + 2], probs[indices[j]*3 + 2]
            );
            
            // Compute dosage = P(AB) + 2*P(BB)
            __m256 dosage = _mm256_fmadd_ps(two, p_bb, p_ab);
            
            // Store results
            _mm256_storeu_ps(&output[j], dosage);
        }
        
        #elif defined(__ARM_NEON) || defined(__aarch64__)
        // ARM NEON: Process 4 samples at a time
        const float32x4_t two = vdupq_n_f32(2.0f);
        
        for (j = 0; j <= n_indices - 4; j += 4) {
            // Gather P(AB) values for 4 selected samples
            float32x4_t p_ab = {probs[indices[j]*3 + 1], probs[indices[j+1]*3 + 1], 
                               probs[indices[j+2]*3 + 1], probs[indices[j+3]*3 + 1]};
            
            // Gather P(BB) values for 4 selected samples
            float32x4_t p_bb = {probs[indices[j]*3 + 2], probs[indices[j+1]*3 + 2], 
                               probs[indices[j+2]*3 + 2], probs[indices[j+3]*3 + 2]};
            
            // Compute dosage = P(AB) + 2*P(BB)
            float32x4_t dosage = vmlaq_f32(p_ab, p_bb, two);
            
            // Store results
            vst1q_f32(&output[j], dosage);
        }
        
        #else
        // No SIMD: use scalar code for entire loop
        j = 0;
        #endif
        
        // Handle remaining samples with scalar code
        for (; j < n_indices; j++) {
            int idx = indices[j];
            output[j] = probs[idx*3 + 1] + 2.0f * probs[idx*3 + 2];
        }
    }
    """
    void compute_dosages_simd(const float* probs, float* output, int n_samples) nogil
    void compute_dosages_filtered_simd(const float* probs, const int* indices, 
                                      float* output, int n_indices) nogil


np.import_array()

# Module-level buffer for decompression reuse
cdef uint8_t* _decompress_buffer = NULL
cdef uint32_t _decompress_buffer_size = 0

# Implementation of GenotypeData
cdef class GenotypeData:
    
    def __cinit__(self):
        self.raw_data = NULL
        self.probs = NULL
        self.ploidy = NULL
        self._initialized = False
        self.n_samples = 0
        self.n_alleles = 0
        self.data_length = 0
        self.compression = 0
        self.phased = False
        self.constant_ploidy = False
        self.max_ploidy = 0
        self.has_missing = False
    
    def __dealloc__(self):
        if self.raw_data != NULL:
            PyMem_Free(self.raw_data)
            self.raw_data = NULL
        if self.probs != NULL:
            PyMem_Free(self.probs)
            self.probs = NULL
        if self.ploidy != NULL:
            PyMem_Free(self.ploidy)
            self.ploidy = NULL
    
    cdef void parse_layout1(self, uint8_t* data, uint32_t length):
        """Parse layout 1 (v1.1) genotype data block."""
        cdef uint32_t uncompressed_length
        cdef uint8_t* uncompressed_data
        cdef uint32_t pos = 0
        cdef uint32_t i
        cdef uint16_t prob_aa, prob_ab, prob_bb
        
        # Store raw data
        self.data_length = length
        self.raw_data = <uint8_t*>PyMem_Malloc(length)
        if self.raw_data == NULL:
            raise MemoryError("Failed to allocate memory for genotype data")
        memcpy(self.raw_data, data, length)
        
        # Decompress if needed
        if self.compression != 0:
            self.decompress_data(has_length_prefix=False)  # v1.1 has no length prefix
            uncompressed_data = self.raw_data
            length = self.data_length
        else:
            uncompressed_data = data
        
        # v1.1 is always diploid (ploidy = 2)
        self.ploidy = <uint8_t*>PyMem_Malloc(self.n_samples * sizeof(uint8_t))
        if self.ploidy == NULL:
            raise MemoryError("Failed to allocate ploidy array")
        
        for i in range(self.n_samples):
            self.ploidy[i] = 2
        
        # v1.1 is always unphased
        self.phased = False
        
        # v1.1 has constant ploidy (always 2) and no missing samples
        self.constant_ploidy = True
        self.max_ploidy = 2
        self.has_missing = False
        
        # Allocate probability array (3 probs per sample for biallelic diploid)
        self.probs = <float*>PyMem_Malloc(self.n_samples * 3 * sizeof(float))
        if self.probs == NULL:
            raise MemoryError("Failed to allocate probability array")
        
        # Read probabilities (3 x 2 bytes per sample)
        for i in range(self.n_samples):
            # Read three 16-bit probabilities
            prob_aa = (<uint16_t*>(uncompressed_data + pos))[0]
            pos += 2
            prob_ab = (<uint16_t*>(uncompressed_data + pos))[0]
            pos += 2
            prob_bb = (<uint16_t*>(uncompressed_data + pos))[0]
            pos += 2
            
            # Convert to float probabilities
            self.probs[i * 3] = prob_aa / 32768.0
            self.probs[i * 3 + 1] = prob_ab / 32768.0
            self.probs[i * 3 + 2] = prob_bb / 32768.0
        
        self._initialized = True
    
    cdef void parse_layout2(self, uint8_t* data, uint32_t length):
        """Parse layout 2 (v1.2) genotype data block."""
        cdef uint32_t uncompressed_length
        cdef uint8_t* uncompressed_data
        cdef uint32_t pos = 0
        cdef uint32_t nn_samples
        cdef uint16_t allele_check
        cdef uint8_t min_ploidy, max_ploidy
        cdef uint8_t n_bits
        cdef uint32_t n_probs_per_sample
        cdef uint32_t total_probs
        cdef uint32_t i, j
        cdef uint8_t ploidy_byte
        cdef uint32_t non_missing_samples = 0
        cdef bint constant_ploidy
        cdef uint32_t prob_idx = 0
        cdef uint32_t sample_idx
        cdef float prob_0, prob_1, prob_remainder
        cdef uint32_t max_val
        cdef float factor
        cdef uint32_t prob_raw_0, prob_raw_1
        cdef uint8_t first, second
        cdef uint32_t max_probs = 3  # For biallelic diploid variants
        # Variables for bit-indexed reading
        cdef uint32_t bit_idx = 0  # Bit index for consecutive probability data
        cdef uint64_t probs_mask
        cdef uint32_t byte_offset
        cdef uint64_t value
        cdef uint32_t prob_raw
        cdef float inv_255 = 0.00392156862745098  # Pre-computed 1.0 / 255.0
        cdef int sum_val
        cdef uint16_t* prob16_ptr
        
        # Store raw data
        self.data_length = length
        self.raw_data = <uint8_t*>PyMem_Malloc(length)
        if self.raw_data == NULL:
            raise MemoryError("Failed to allocate memory for genotype data")
        memcpy(self.raw_data, data, length)
        
        # For v1.2 format, handle decompression
        # The data passed to this function does NOT include the 4-byte length prefix
        # that precedes the genotype block in the file, but compressed data in v1.2
        # DOES start with a 4-byte uncompressed size field
        if self.compression != 0:
            # Compressed data needs decompression
            # v1.2 compressed data starts with uncompressed size (4 bytes) then compressed data
            self.decompress_data(has_length_prefix=True)
            uncompressed_data = self.raw_data
            length = self.data_length
        else:
            # Uncompressed data can be used directly
            uncompressed_data = data
            uncompressed_length = length
        
        # Reset position for uncompressed data
        pos = 0
        
        # Parse v1.2 format exactly following reference implementation:
        # 1. Number of samples (4 bytes)
        nn_samples = (<uint32_t*>(uncompressed_data + pos))[0]
        pos += 4
        if nn_samples != self.n_samples:
            raise ValueError(f"Number of samples doesn't match! Expected {self.n_samples}, got {nn_samples}")
        
        # 2. Number of alleles (2 bytes) 
        allele_check = (<uint16_t*>(uncompressed_data + pos))[0]
        pos += 2
        if allele_check != self.n_alleles:
            raise ValueError(f"Number of alleles doesn't match! Expected {self.n_alleles}, got {allele_check}")
        
        # 3. Min and max ploidy (1 byte each)
        min_ploidy = uncompressed_data[pos]
        pos += 1
        max_ploidy = uncompressed_data[pos]
        pos += 1
        
        # Check if constant ploidy
        # Note: constant_ploidy only means min==max, but doesn't guarantee no missing samples
        # Missing samples can still exist with ploidy byte having bit 7 set
        constant_ploidy = (min_ploidy == max_ploidy)
        self.constant_ploidy = constant_ploidy
        self.max_ploidy = max_ploidy
        
        # Allocate ploidy array
        self.ploidy = <uint8_t*>PyMem_Malloc(self.n_samples * sizeof(uint8_t))
        if self.ploidy == NULL:
            raise MemoryError("Failed to allocate ploidy array")
        
        # 4. Parse ploidy for each sample and count non-missing samples
        self.has_missing = False
        for i in range(self.n_samples):
            ploidy_byte = uncompressed_data[pos + i]
            
            # Check for missing data (bit 7 set)
            if ploidy_byte & 0x80:
                self.ploidy[i] = 0  # Missing
                self.has_missing = True
            else:
                if constant_ploidy:
                    # For constant ploidy, use the max_ploidy value
                    self.ploidy[i] = max_ploidy
                else:
                    # For variable ploidy, extract from the byte (lower 6 bits)
                    self.ploidy[i] = ploidy_byte & 0x3F
                non_missing_samples += 1
            
        
        # Skip past ploidy data (always n_samples bytes)
        pos += self.n_samples
        
        # For ldcov, we only support diploid genotypes
        if min_ploidy != 2 or max_ploidy != 2:
            raise ValueError(f"Only diploid genotypes are supported. Found ploidy range: {min_ploidy}-{max_ploidy}")
        
        # 5. Phased flag (1 byte)
        self.phased = uncompressed_data[pos] != 0
        pos += 1
        
        # 6. Bit depth (1 byte)
        n_bits = uncompressed_data[pos]
        pos += 1
        
        # Validate bit depth
        if n_bits not in [8, 16, 32]:
            raise ValueError(f"Unsupported bit depth: {n_bits}")
        
        # Calculate max value and factor for this bit depth
        if n_bits == 32:
            max_val = 4294967295  # 2^32 - 1
            factor = 2.3283064370807974e-10  # Pre-computed 1.0 / 4294967295
        elif n_bits == 16:
            max_val = 65535  # 2^16 - 1 
            factor = 1.5259021896696422e-05  # Pre-computed 1.0 / 65535
        elif n_bits == 8:
            max_val = 255  # 2^8 - 1
            factor = 0.00392156862745098  # Pre-computed 1.0 / 255
        else:
            max_val = (1 << n_bits) - 1  # 2^n_bits - 1
            factor = 1.0 / <float>max_val
        
        # Initialize probs_mask for bit-indexed reading now that n_bits is known
        probs_mask = (1ULL << n_bits) - 1
        
        # For biallelic diploid variants, we need 3 probabilities per sample
        if self.n_alleles != 2:
            raise ValueError(f"Only biallelic variants are supported. Found {self.n_alleles} alleles")
        
        n_probs_per_sample = 3  # AA, AB, BB
        total_probs = self.n_samples * n_probs_per_sample
        
        # Allocate probability array for all samples (initialize to NaN for missing)
        self.probs = <float*>PyMem_Malloc(total_probs * sizeof(float))
        if self.probs == NULL:
            raise MemoryError("Failed to allocate probability array")
        
        # Initialize all probabilities to NaN
        for i in range(total_probs):
            self.probs[i] = NAN
        
        # 7. Read probability data - CRITICAL: only for non-missing samples
        # Following reference implementation: probabilities_layout2() function
        cdef uint32_t prob_offset = 0  # Offset for probability data
        
        # For constant ploidy with 8-bit depth and 3 probabilities per sample (fast path)
        if constant_ploidy and max_probs == 3 and n_bits == 8:
            # Fast path: Read probability data for ALL samples (no missing samples when constant_ploidy=True)
            # This matches reference implementation exactly
            
            for i in range(self.n_samples):
                # Read 2 bytes for this sample (all samples present when constant_ploidy=True)
                first = uncompressed_data[pos + prob_offset]
                second = uncompressed_data[pos + prob_offset + 1]
                prob_offset += 2
                
                # Convert using pre-computed factor
                prob_0 = <float>first * inv_255      # P(AA)
                prob_1 = <float>second * inv_255     # P(AB)
                sum_val = 255 - first - second
                prob_remainder = <float>sum_val * inv_255  # P(BB)
                
                # Store probabilities for this sample
                sample_idx = i * 3
                self.probs[sample_idx] = prob_0      # P(AA)
                self.probs[sample_idx + 1] = prob_1  # P(AB)
                self.probs[sample_idx + 2] = prob_remainder  # P(BB)
        else:
            # General path for other bit depths or when there are missing samples
            # Use bit-indexed reading like reference implementation
            # CRITICAL: Read probability data for ALL samples (including missing ones)
            # Missing samples are identified by ploidy bytes, not absent probability data
            bit_idx = 0  # Reset bit index for consecutive probability data
            
            if n_bits == 16 and constant_ploidy and max_probs == 3:
                # Optimized path for 16-bit data
                prob16_ptr = <uint16_t*>(uncompressed_data + pos)
                for i in range(self.n_samples):
                    # Read 2 uint16 probabilities
                    prob_raw_0 = prob16_ptr[0]
                    prob_raw_1 = prob16_ptr[1]
                    prob16_ptr += 2
                    
                    # Convert to float probabilities using pre-computed factor
                    prob_0 = <float>prob_raw_0 * factor  # factor already computed for 16-bit
                    prob_1 = <float>prob_raw_1 * factor
                    prob_remainder = 1.0 - prob_0 - prob_1
                    
                    # Store probabilities
                    sample_idx = i * 3
                    self.probs[sample_idx] = prob_0      # P(AA)
                    self.probs[sample_idx + 1] = prob_1  # P(AB)
                    self.probs[sample_idx + 2] = prob_remainder  # P(BB)
            else:
                # General bit-indexed path for other cases
                for i in range(self.n_samples):
                    # Read 2 probabilities using bit indexing for ALL samples
                    # Read first probability
                    byte_offset = pos + bit_idx // 8
                    value = (<uint64_t*>(uncompressed_data + byte_offset))[0]
                    prob_raw = (value >> (bit_idx % 8)) & probs_mask
                    prob_0 = <float>prob_raw * factor  # P(AA)
                    bit_idx += n_bits
                    
                    # Read second probability  
                    byte_offset = pos + bit_idx // 8
                    value = (<uint64_t*>(uncompressed_data + byte_offset))[0]
                    prob_raw = (value >> (bit_idx % 8)) & probs_mask
                    prob_1 = <float>prob_raw * factor  # P(AB)
                    bit_idx += n_bits
                    
                    # Calculate third probability as remainder
                    prob_remainder = 1.0 - prob_0 - prob_1  # P(BB) = 1 - P(AA) - P(AB)
                    
                    # Store probabilities for this sample
                    sample_idx = i * 3
                    self.probs[sample_idx] = prob_0      # P(AA)
                    self.probs[sample_idx + 1] = prob_1  # P(AB)
                    self.probs[sample_idx + 2] = prob_remainder  # P(BB)
            
            # AFTER reading all probability data, set missing samples to NaN
            # This matches the reference implementation approach
            for i in range(self.n_samples):
                if self.ploidy[i] == 0:
                    # Missing sample - set probabilities to NaN AFTER reading all data
                    sample_idx = i * 3
                    self.probs[sample_idx] = NAN      # P(AA)
                    self.probs[sample_idx + 1] = NAN  # P(AB)
                    self.probs[sample_idx + 2] = NAN  # P(BB)
            
        
        self._initialized = True
    
    cdef void decompress_data(self, bint has_length_prefix=True):
        """Decompress genotype data using C++ decompression."""
        cdef uint32_t compressed_length
        cdef uint32_t offset = 0
        cdef uint32_t expected_size
        cdef size_t decompressed_size
        cdef int ret
        
        global _decompress_buffer, _decompress_buffer_size
        
        if has_length_prefix:
            # Skip the uncompressed length field (v1.2)
            compressed_length = self.data_length - 4
            offset = 4
            # Read expected uncompressed size for buffer allocation
            expected_size = (<uint32_t*>self.raw_data)[0]
        else:
            # No length prefix (v1.1)
            compressed_length = self.data_length
            offset = 0
            # For v1.1, estimate size (6 bytes per sample)
            expected_size = self.n_samples * 6
        
        # Ensure our reusable buffer is large enough
        if _decompress_buffer_size < expected_size:
            if _decompress_buffer != NULL:
                PyMem_Free(_decompress_buffer)
            _decompress_buffer_size = expected_size + 1024  # Add some padding
            _decompress_buffer = <uint8_t*>PyMem_Malloc(_decompress_buffer_size)
            if _decompress_buffer == NULL:
                raise MemoryError("Failed to allocate decompression buffer")
        
        decompressed_size = _decompress_buffer_size
        
        # Use C++ decompression
        if self.compression == BGEN_COMPRESSED_ZLIB:
            ret = decompress_zlib(self.raw_data + offset, compressed_length,
                                  _decompress_buffer, &decompressed_size)
            if ret != 0:
                raise RuntimeError(f"zlib decompression failed with error code: {ret}")
        elif self.compression == BGEN_COMPRESSED_ZSTD:
            ret = decompress_zstd(self.raw_data + offset, compressed_length,
                                  _decompress_buffer, &decompressed_size)
            if ret != 0:
                raise RuntimeError(f"zstd decompression failed with error code: {ret}")
        else:
            raise ValueError(f"Unknown compression type: {self.compression}")
        
        # Update data pointer and length
        self.data_length = decompressed_size
        
        # Allocate new buffer for decompressed data
        if self.raw_data != NULL:
            PyMem_Free(self.raw_data)
        
        self.raw_data = <uint8_t*>PyMem_Malloc(self.data_length)
        if self.raw_data == NULL:
            raise MemoryError("Failed to allocate memory for uncompressed data")
        
        # Copy decompressed data from buffer
        memcpy(self.raw_data, _decompress_buffer, self.data_length)
    
    cdef void compute_dosages(self, float* output) nogil:
        """Compute alt allele dosages from probabilities."""
        cdef uint32_t i
        cdef float p_aa, p_ab, p_bb
        cdef float* prob_ptr
        
        if not self._initialized or self.probs == NULL:
            return
        
        # For biallelic variants, dosage = 0*P(AA) + 1*P(AB) + 2*P(BB)
        if self.n_alleles == 2:
            # Fast path for constant ploidy with no missing samples
            if self.constant_ploidy and self.max_ploidy == 2 and not self.has_missing:
                # Use SIMD-optimized function for best performance
                compute_dosages_simd(self.probs, output, self.n_samples)
            else:
                # Variable ploidy - need to check each sample
                for i in range(self.n_samples):
                    # Check for non-diploid samples (missing samples have ploidy = 0)
                    # Only compute dosages for diploid samples (ploidy = 2)
                    if self.ploidy[i] != 2:
                        output[i] = NAN  # Missing samples should be NaN
                    else:
                        prob_ptr = &self.probs[i * 3]
                        output[i] = prob_ptr[1] + 2.0 * prob_ptr[2]
        else:
            # For multiallelic, we would need more complex logic
            # For now, just set to 0
            for i in range(self.n_samples):
                output[i] = 0.0
    
    cdef void compute_dosages_filtered(self, int* sample_indices, int n_indices, float* output) nogil:
        """Compute alt allele dosages only for specified sample indices."""
        cdef int j
        cdef uint32_t sample_idx
        cdef float p_aa, p_ab, p_bb
        cdef float* prob_ptr
        cdef int remainder
        cdef bint all_valid = True
        
        if not self._initialized or self.probs == NULL:
            return
        
        # For biallelic variants, dosage = 0*P(AA) + 1*P(AB) + 2*P(BB)
        if self.n_alleles == 2:
            # Check if we can use the fast SIMD path
            # This requires all samples to be valid diploid samples
            if self.constant_ploidy and self.max_ploidy == 2 and not self.has_missing:
                # First, verify all indices are valid
                for j in range(n_indices):
                    if sample_indices[j] >= <int>self.n_samples:
                        all_valid = False
                        break
                
                if all_valid:
                    # Use SIMD-optimized function
                    compute_dosages_filtered_simd(self.probs, sample_indices, output, n_indices)
                    return
            
            # Fallback path with ploidy checking
            # Process in chunks of 4 for better CPU pipelining
            remainder = n_indices % 4
            
            # Process groups of 4
            for j in range(0, n_indices - remainder, 4):
                # Prefetch next iteration's data
                if j + 4 < n_indices:
                    prefetch(&self.probs[sample_indices[j + 4] * 3])
                    
                # Unrolled loop for better performance
                sample_idx = sample_indices[j]
                if sample_idx < self.n_samples and self.ploidy[sample_idx] == 2:
                    prob_ptr = &self.probs[sample_idx * 3]
                    output[j] = prob_ptr[1] + 2.0 * prob_ptr[2]
                else:
                    output[j] = NAN if sample_idx < self.n_samples else 0.0
                
                sample_idx = sample_indices[j + 1]
                if sample_idx < self.n_samples and self.ploidy[sample_idx] == 2:
                    prob_ptr = &self.probs[sample_idx * 3]
                    output[j + 1] = prob_ptr[1] + 2.0 * prob_ptr[2]
                else:
                    output[j + 1] = NAN if sample_idx < self.n_samples else 0.0
                
                sample_idx = sample_indices[j + 2]
                if sample_idx < self.n_samples and self.ploidy[sample_idx] == 2:
                    prob_ptr = &self.probs[sample_idx * 3]
                    output[j + 2] = prob_ptr[1] + 2.0 * prob_ptr[2]
                else:
                    output[j + 2] = NAN if sample_idx < self.n_samples else 0.0
                
                sample_idx = sample_indices[j + 3]
                if sample_idx < self.n_samples and self.ploidy[sample_idx] == 2:
                    prob_ptr = &self.probs[sample_idx * 3]
                    output[j + 3] = prob_ptr[1] + 2.0 * prob_ptr[2]
                else:
                    output[j + 3] = NAN if sample_idx < self.n_samples else 0.0
            
            # Handle remaining samples
            for j in range(n_indices - remainder, n_indices):
                sample_idx = sample_indices[j]
                if sample_idx >= self.n_samples:
                    output[j] = 0.0
                elif self.ploidy[sample_idx] != 2:
                    output[j] = NAN
                else:
                    prob_ptr = &self.probs[sample_idx * 3]
                    output[j] = prob_ptr[1] + 2.0 * prob_ptr[2]
        else:
            # For multiallelic, we would need more complex logic
            # For now, just set to 0
            for j in range(n_indices):
                output[j] = 0.0


# Module cleanup function
def _cleanup_module():
    """Clean up module-level resources."""
    global _decompress_buffer, _decompress_buffer_size
    if _decompress_buffer != NULL:
        PyMem_Free(_decompress_buffer)
        _decompress_buffer = NULL
        _decompress_buffer_size = 0

# Register cleanup function with atexit
import atexit
atexit.register(_cleanup_module)

# Runtime detection of compression backend
def get_compression_backend():
    """
    Get information about the compression backend being used.
    
    Returns
    -------
    dict
        Dictionary with backend information including:
        - 'type': 'vendored' or 'system'
        - 'zlib': Description of zlib implementation
        - 'zstd': Description of zstd implementation
    """
    try:
        # Import the build configuration that was generated during installation
        from . import _build_config
        return _build_config.get_build_info()
    except ImportError:
        # Fallback if build config is missing (shouldn't happen in normal installs)
        return {
            'type': 'unknown',
            'zlib': 'Unknown',
            'zstd': 'Unknown',
            'note': 'Build configuration not found - compression backend unknown'
        }