"""Lossless export acceleration; geometry, UVs and PBR pixels are preserved."""
from concurrent.futures import ThreadPoolExecutor
import io
import time


TEXTURE_ATTRIBUTES = ('baseColorTexture', 'metallicRoughnessTexture',
                      'normalTexture', 'occlusionTexture', 'emissiveTexture')


def inpaint_pbr_channels(base_color, metallic, roughness, alpha, mask_inv):
    """Run the original four independent OpenCV operations concurrently."""
    import cv2
    inputs = [(base_color, 3), (metallic, 1), (roughness, 1), (alpha, 1)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(cv2.inpaint, value, mask_inv, radius, cv2.INPAINT_TELEA)
                   for value, radius in inputs]
        results = [future.result() for future in futures]
    return results[0], results[1][..., None], results[2][..., None], results[3][..., None]


def export_glb_lossless(mesh, destination, compress_level=1):
    """Pre-encode owned textures in parallel, then use the normal GLB exporter.

    PNG compression level changes encoded bytes, never decoded pixels. The
    per-image adapters only supply the already-encoded PNG to trimesh; no global
    image encoder or exporter is patched. Every adapter is restored on exit.
    """
    if type(compress_level) is not int or not 0 <= compress_level <= 9:
        raise ValueError('PNG compression level must be an integer from 0 to 9')
    images = []
    seen = set()
    material = getattr(mesh.visual, 'material', None)
    materials = getattr(material, 'materials', None) or [material]
    for material in materials:
        for name in TEXTURE_ATTRIBUTES:
            image = getattr(material, name, None)
            if image is not None and id(image) not in seen:
                images.append(image)
                seen.add(id(image))

    if __package__:
        from .trellis_png import encode_png
    else:
        from trellis_png import encode_png

    def encode(image):
        return encode_png(image, compress_level=compress_level)

    started = time.perf_counter()
    if images:
        with ThreadPoolExecutor(max_workers=min(2, len(images))) as executor:
            encoded = list(executor.map(encode, images))
    else:
        encoded = []
    encoded_seconds = time.perf_counter() - started
    restored = []
    try:
        for image, png in zip(images, encoded):
            original = image.save
            had_instance_save = 'save' in image.__dict__
            saved_attribute = image.__dict__.get('save')

            def save_preencoded(fp, format=None, _png=png, _save=original, **parameters):
                if str(format).lower() == 'png' and hasattr(fp, 'write') and not parameters:
                    fp.write(_png)
                    return
                return _save(fp, format=format, **parameters)

            restored.append((image, had_instance_save, saved_attribute))
            image.save = save_preencoded
        packed_started = time.perf_counter()
        mesh.export(str(destination), include_normals=True)
        packed_seconds = time.perf_counter() - packed_started
    finally:
        for image, had_instance_save, saved_attribute in restored:
            if had_instance_save:
                image.save = saved_attribute
            else:
                del image.save
    return {'png_encode_seconds': encoded_seconds, 'glb_pack_and_write_seconds': packed_seconds,
            'png_compress_level': compress_level, 'png_images': len(images),
            'png_encoder': 'parallel-deflate-with-pillow-fallback',
            'serialization_seconds': time.perf_counter() - started}
