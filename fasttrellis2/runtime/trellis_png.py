"""Standard, lossless PNG encoding with parallel DEFLATE stripes."""
import ctypes
import io
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def _chunk(kind, data):
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(data, zlib.crc32(kind)))


def encode_png(image, compress_level=1, workers=8):
    # Metadata and uncommon modes retain the original Pillow encoder.
    if image.mode not in ('RGB', 'RGBA') or image.info or compress_level != 1:
        with io.BytesIO() as buffer:
            image.save(buffer, format='PNG', compress_level=compress_level, optimize=False)
            return buffer.getvalue()
    pixels = np.asarray(image, dtype=np.uint8)
    height, width, channels = pixels.shape
    flat = pixels.reshape(height, width * channels)
    filtered = np.empty((height, width * channels + 1), dtype=np.uint8)
    filtered[:, 0] = 1
    filtered[:, 1:1 + channels] = flat[:, :channels]
    np.subtract(flat[:, channels:], flat[:, :-channels], out=filtered[:, 1 + channels:])
    slices = np.array_split(filtered, min(workers, height), axis=0)

    def compress(task):
        index, stripe = task
        data = memoryview(stripe)
        encoder = zlib.compressobj(level=1, wbits=-15)
        encoded = encoder.compress(data)
        encoded += encoder.flush(zlib.Z_FINISH if index == len(slices) - 1 else zlib.Z_FULL_FLUSH)
        return encoded, zlib.adler32(data), stripe.nbytes

    with ThreadPoolExecutor(max_workers=workers) as pool:
        encoded = list(pool.map(compress, enumerate(slices)))
    libz = ctypes.CDLL('libz.so.1')
    combine = libz.adler32_combine
    combine.argtypes = [ctypes.c_ulong, ctypes.c_ulong, ctypes.c_long]
    combine.restype = ctypes.c_ulong
    checksum = 1
    for _, stripe_sum, size in encoded:
        checksum = combine(checksum, stripe_sum, size)
    data = b'\x78\x01' + b''.join(part[0] for part in encoded) + struct.pack('>I', checksum)
    header = struct.pack('>IIBBBBB', width, height, 8, 6 if channels == 4 else 2, 0, 0, 0)
    return b'\x89PNG\r\n\x1a\n' + _chunk(b'IHDR', header) + _chunk(b'IDAT', data) + _chunk(b'IEND', b'')
