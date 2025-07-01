#ifndef LDCOV_DECOMPRESS_H
#define LDCOV_DECOMPRESS_H

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Decompress zlib-compressed data
// Returns 0 on success, non-zero on error
int decompress_zlib(const uint8_t* compressed, size_t compressed_size, uint8_t* decompressed,
                    size_t* decompressed_size);

// Decompress zstd-compressed data
// Returns 0 on success, non-zero on error
int decompress_zstd(const uint8_t* compressed, size_t compressed_size, uint8_t* decompressed,
                    size_t* decompressed_size);

#ifdef __cplusplus
}
#endif

#endif  // LDCOV_DECOMPRESS_H