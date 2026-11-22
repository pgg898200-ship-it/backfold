// Stable, public LZ4 frame ABI used only if the optional development header is absent.
// We use default settings and pass NULL for all option structures.
#pragma once
#include <stddef.h>
extern "C" {
struct LZ4F_cctx_s;
struct LZ4F_dctx_s;
struct LZ4F_preferences_t_placeholder;
struct LZ4F_compressOptions_t_placeholder;
struct LZ4F_decompressOptions_t_placeholder;
unsigned LZ4F_getVersion(void);
const char* LZ4_versionString(void);
unsigned LZ4F_isError(size_t);
const char* LZ4F_getErrorName(size_t);
size_t LZ4F_createCompressionContext(LZ4F_cctx_s**, unsigned);
size_t LZ4F_createDecompressionContext(LZ4F_dctx_s**, unsigned);
size_t LZ4F_freeCompressionContext(LZ4F_cctx_s*);
size_t LZ4F_freeDecompressionContext(LZ4F_dctx_s*);
size_t LZ4F_compressBound(size_t, const LZ4F_preferences_t_placeholder*);
size_t LZ4F_compressBegin(LZ4F_cctx_s*, void*, size_t, const LZ4F_preferences_t_placeholder*);
size_t LZ4F_compressUpdate(LZ4F_cctx_s*, void*, size_t, const void*, size_t,
                          const LZ4F_compressOptions_t_placeholder*);
size_t LZ4F_flush(LZ4F_cctx_s*, void*, size_t, const LZ4F_compressOptions_t_placeholder*);
size_t LZ4F_decompress(LZ4F_dctx_s*, void*, size_t*, const void*, size_t*,
                       const LZ4F_decompressOptions_t_placeholder*);
}
