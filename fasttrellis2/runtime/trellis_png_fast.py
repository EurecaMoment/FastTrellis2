"""Lossless PNG with the system libdeflate codec and one output allocation."""
import ctypes
import ctypes.util
import struct

import numpy as np

from trellis_png import encode_png as fallback, _chunk


_name = ctypes.util.find_library('deflate')
_codec = ctypes.CDLL(_name) if _name else None
if _codec is not None:
    _codec.libdeflate_alloc_compressor.argtypes = [ctypes.c_int]
    _codec.libdeflate_alloc_compressor.restype = ctypes.c_void_p
    _codec.libdeflate_free_compressor.argtypes = [ctypes.c_void_p]
    _codec.libdeflate_zlib_compress_bound.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _codec.libdeflate_zlib_compress_bound.restype = ctypes.c_size_t
    _codec.libdeflate_zlib_compress.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                             ctypes.c_void_p, ctypes.c_size_t]
    _codec.libdeflate_zlib_compress.restype = ctypes.c_size_t
    _codec.libdeflate_crc32.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t]
    _codec.libdeflate_crc32.restype = ctypes.c_uint32


def encode_png(image, compress_level=1, workers=8):
    if _codec is None or image.mode not in ('RGB', 'RGBA') or image.info:
        return fallback(image, compress_level, workers)
    pixels = getattr(image,'_trellis_pixels',None)
    if pixels is None:pixels=np.asarray(image,dtype=np.uint8)
    height, width, channels = pixels.shape
    flat = pixels.reshape(height, width * channels)
    filtered = np.empty((height, width * channels + 1), dtype=np.uint8)
    if compress_level == 0:
        filtered[:, 0] = 0
        filtered[:, 1:] = flat
    else:
        filtered[:, 0] = 1
        filtered[:, 1:1 + channels] = flat[:, :channels]
        np.subtract(flat[:, channels:], flat[:, :-channels], out=filtered[:, 1 + channels:])
    compressor = _codec.libdeflate_alloc_compressor(compress_level)
    if not compressor:
        raise MemoryError('Cannot initialize PNG compressor')
    try:
        bound = _codec.libdeflate_zlib_compress_bound(compressor, filtered.nbytes)
        output = np.empty(bound + 57, dtype=np.uint8)
        size = _codec.libdeflate_zlib_compress(compressor, filtered.ctypes.data, filtered.nbytes,
                                              output.ctypes.data + 41, bound)
        if not size:
            raise RuntimeError('PNG compressor exceeded its advertised output bound')
    finally:
        _codec.libdeflate_free_compressor(compressor)
    header = struct.pack('>IIBBBBB', width, height, 8, 6 if channels == 4 else 2, 0, 0, 0)
    prefix = b'\x89PNG\r\n\x1a\n' + _chunk(b'IHDR', header) + struct.pack('>I', size) + b'IDAT'
    output[:41] = np.frombuffer(prefix, np.uint8)
    crc = _codec.libdeflate_crc32(0, output.ctypes.data + 37, size + 4)
    suffix = struct.pack('>I', crc) + _chunk(b'IEND', b'')
    output[41 + size:57 + size] = np.frombuffer(suffix, np.uint8)
    return output[:57 + size]


encode_png.backend_name = 'libdeflate-single-output-buffer'
