#include "decompress.h"
#include <zlib.h>
#include <zstd.h>
#include <algorithm>

extern "C" {

int decompress_zlib(const uint8_t* compressed, size_t compressed_size,
                    uint8_t* decompressed, size_t* decompressed_size) {
    // Initialize zlib decompression
    z_stream strm;
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.avail_in = 0;
    strm.next_in = Z_NULL;
    
    // Check for zlib header to determine format
    // BGEN v1.1 uses standard zlib (with header)
    // BGEN v1.2 uses raw deflate (no header)
    int ret;
    if (compressed_size >= 2 && compressed[0] == 0x78 && 
        (compressed[1] == 0x01 || compressed[1] == 0x5E || 
         compressed[1] == 0x9C || compressed[1] == 0xDA)) {
        // Standard zlib format detected (v1.1)
        ret = inflateInit(&strm);
    } else {
        // Raw deflate format (v1.2)
        ret = inflateInit2(&strm, -15);
    }
    
    if (ret != Z_OK) {
        return ret;
    }
    
    // Set input
    strm.avail_in = compressed_size;
    strm.next_in = const_cast<uint8_t*>(compressed);
    
    // Set output
    strm.avail_out = *decompressed_size;
    strm.next_out = decompressed;
    
    // Decompress
    ret = inflate(&strm, Z_FINISH);
    
    // Clean up
    inflateEnd(&strm);
    
    if (ret == Z_STREAM_END) {
        *decompressed_size = strm.total_out;
        return 0;  // Success
    }
    
    return ret;  // Error
}

int decompress_zstd(const uint8_t* compressed, size_t compressed_size,
                    uint8_t* decompressed, size_t* decompressed_size) {
    // Decompress using zstd
    size_t const result = ZSTD_decompress(
        decompressed, *decompressed_size,
        compressed, compressed_size
    );
    
    if (ZSTD_isError(result)) {
        return -1;  // Error
    }
    
    *decompressed_size = result;
    return 0;  // Success
}

} // extern "C"