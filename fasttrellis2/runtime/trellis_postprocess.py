# Adapted from Microsoft TRELLIS.2 o_voxel/postprocess.py.
from typing import *
from contextlib import nullcontext
import json
import time
from tqdm import tqdm
import numpy as np
import torch
import cv2
from PIL import Image
import trimesh
import trimesh.visual
from flex_gemm.ops.grid_sample import grid_sample_3d
import nvdiffrast.torch as dr
import cumesh
if __package__:
    from .trellis_export import inpaint_pbr_channels
else:
    from trellis_export import inpaint_pbr_channels


def to_glb(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    decimation_target: int = 1000000,
    texture_size: int = 2048,
    remesh: bool = False,
    remesh_band: float = 1,
    remesh_project: float = 0.9,
    mesh_cluster_threshold_cone_half_angle_rad=np.radians(90.0),
    mesh_cluster_refine_iterations=0,
    mesh_cluster_global_iterations=1,
    mesh_cluster_smooth_strength=1,
    verbose: bool = False,
    use_tqdm: bool = False,
    timings: Optional[dict] = None,
    parallel_inpaint: bool = True,
    verify_inpaint: bool = False,
    check=None,
    project_texture_to_source: bool = True,
    fast_cleanup: bool = False,
    texture_padding: str = 'telea',
    uv_mode: str = 'xatlas',
    simplify_options: Optional[dict] = None,
    texture_sampling: str = 'sparse',
    uv_group_face_cap: int = 4096,
    bvh_factory=None,
    exact_inpaint=None,
    on_uv_cpu_start=None,
    deferred_material=None,
    on_texture_mask_ready=None,
    geometry_scope=None,
    exact_channel_extract=None,
    geometry_export_prepare=None,
    material_pixels=None,
):
    """
    Convert an extracted mesh to a GLB file.
    Performs cleaning, optional remeshing, UV unwrapping, and texture baking from a volume.

    Args:
        vertices: (N, 3) tensor of vertex positions
        faces: (M, 3) tensor of vertex indices
        attr_volume: (L, C) features of a sprase tensor for attribute interpolation
        coords: (L, 3) tensor of coordinates for each voxel
        attr_layout: dictionary of slice objects for each attribute
        aabb: (2, 3) tensor of minimum and maximum coordinates of the volume
        voxel_size: (3,) tensor of size of each voxel
        grid_size: (3,) tensor of number of voxels in each dimension
        decimation_target: target number of vertices for mesh simplification
        texture_size: size of the texture for baking
        remesh: whether to perform remeshing
        remesh_band: size of the remeshing band
        remesh_project: projection factor for remeshing
        mesh_cluster_threshold_cone_half_angle_rad: threshold for cone-based clustering in uv unwrapping
        mesh_cluster_refine_iterations: number of iterations for refining clusters in uv unwrapping
        mesh_cluster_global_iterations: number of global iterations for clustering in uv unwrapping
        mesh_cluster_smooth_strength: strength of smoothing for clustering in uv unwrapping
        verbose: whether to print verbose messages
        use_tqdm: whether to use tqdm to display progress bar
    """
    timings = {} if timings is None else timings
    torch.cuda.synchronize()
    stage_started = time.perf_counter()

    def mark(name):
        nonlocal stage_started
        torch.cuda.synchronize()
        now = time.perf_counter()
        timings[name] = now - stage_started
        stage_started = now
        print('POSTPROCESS_TIMING ' + json.dumps({name: timings[name]}), flush=True)
        if check is not None:
            check()

    # --- Input Normalization (AABB, Voxel Size, Grid Size) ---
    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=coords.device)
    assert isinstance(aabb, torch.Tensor), f"aabb must be a list, tuple, np.ndarray, or torch.Tensor, but got {type(aabb)}"
    assert aabb.dim() == 2, f"aabb must be a 2D tensor, but got {aabb.shape}"
    assert aabb.size(0) == 2, f"aabb must have 2 rows, but got {aabb.size(0)}"
    assert aabb.size(1) == 3, f"aabb must have 3 columns, but got {aabb.size(1)}"

    # Calculate grid dimensions based on AABB and voxel size
    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size, voxel_size, voxel_size]
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=coords.device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        assert grid_size is not None, "Either voxel_size or grid_size must be provided"
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size, grid_size]
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size = torch.tensor(grid_size, dtype=torch.int32, device=coords.device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size

    # Assertions for dimensions
    assert isinstance(voxel_size, torch.Tensor)
    assert voxel_size.dim() == 1 and voxel_size.size(0) == 3
    assert isinstance(grid_size, torch.Tensor)
    assert grid_size.dim() == 1 and grid_size.size(0) == 3

    if use_tqdm:
        pbar = tqdm(total=6, desc="Extracting GLB")
    if verbose:
        print(f"Original mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    # Move data to GPU
    vertices = vertices.cuda()
    faces = faces.cuda()

    # Initialize CUDA mesh handler
    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)

    mark('input_setup_seconds')
    with geometry_scope() if geometry_scope is not None else nullcontext():
        # --- Initial Mesh Cleaning ---
        # Fills holes as much as we can before processing
        mesh.fill_holes(max_hole_perimeter=3e-2)
        if verbose:
            print(f"After filling holes: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        vertices, faces = mesh.read()
        if use_tqdm:
            pbar.update(1)

        mark('hole_fill_seconds')
        # Build BVH for the current mesh to guide remeshing
        if use_tqdm:
            pbar.set_description("Building BVH")
        if verbose:
            print(f"Building BVH for current mesh...", end='', flush=True)
        build_bvh = cumesh.cuBVH if bvh_factory is None else bvh_factory
        bvh = build_bvh(vertices, faces) if remesh or project_texture_to_source else None
        if use_tqdm:
            pbar.update(1)
        if verbose:
            print("Done")

        if use_tqdm:
            pbar.set_description("Cleaning mesh")
        if verbose:
            print("Cleaning mesh...")

        mark('bvh_build_seconds')
        # --- Branch 1: Standard Pipeline (Simplification & Cleaning) ---
        if not remesh:
            if fast_cleanup:
                mesh.simplify(decimation_target, verbose=verbose, options=simplify_options or {})
                mesh.remove_duplicate_faces()
                mesh.remove_degenerate_faces()
            else:
                # Step 1: Aggressive simplification (3x target)
                mesh.simplify(decimation_target * 3, verbose=verbose)
            if verbose and not fast_cleanup:
                print(f"After inital simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

            # Step 2: Clean up topology (duplicates, non-manifolds, isolated parts)
            if not fast_cleanup:
                mesh.remove_duplicate_faces()
                mesh.repair_non_manifold_edges()
                mesh.remove_small_connected_components(1e-5)
                mesh.fill_holes(max_hole_perimeter=3e-2)
            if verbose and not fast_cleanup:
                print(f"After initial cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

            # Step 3: Final simplification to target count
            if not fast_cleanup:
                mesh.simplify(decimation_target, verbose=verbose)
            if verbose and not fast_cleanup:
                print(f"After final simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

            # Step 4: Final Cleanup loop
            if not fast_cleanup:
                mesh.remove_duplicate_faces()
                mesh.repair_non_manifold_edges()
                mesh.remove_small_connected_components(1e-5)
                mesh.fill_holes(max_hole_perimeter=3e-2)
            if verbose and not fast_cleanup:
                print(f"After final cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

            # Step 5: Unify face orientations
            if not fast_cleanup:
                mesh.unify_face_orientations()

        # --- Branch 2: Remeshing Pipeline ---
        else:
            center = aabb.mean(dim=0)
            scale = (aabb[1] - aabb[0]).max().item()
            resolution = grid_size.max().item()

            # Perform Dual Contouring remeshing (rebuilds topology)
            mesh.init(*cumesh.remeshing.remesh_narrow_band_dc(
                vertices, faces,
                center = center,
                scale = (resolution + 3 * remesh_band) / resolution * scale,
                resolution = resolution,
                band = remesh_band,
                project_back = remesh_project, # Snaps vertices back to original surface
                verbose = verbose,
                bvh = bvh,
            ))
            if verbose:
                print(f"After remeshing: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

            # Simplify and clean the remeshed result (similar logic to above)
            mesh.simplify(decimation_target, verbose=verbose)
            if verbose:
                print(f"After simplifying: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

        if use_tqdm:
            pbar.update(1)
        if verbose:
            print("Done")


    mark('geometry_cleanup_seconds')
    uv_cleanup_futures = []
    # --- UV Parameterization ---
    if use_tqdm:
        pbar.set_description("Parameterizing new mesh")
    if verbose:
        print("Parameterizing new mesh...")

    if uv_mode != 'xatlas':
        raise ValueError('FastTrellis2 uses xatlas UVs')
    if uv_mode == 'xatlas':
        if __package__:
            from .trellis_uv_exact import unwrap_bounded
        else:
            from trellis_uv_exact import unwrap_bounded
        timings['uv_dispatch_seconds'] = time.perf_counter() - stage_started
        uv_call_started = time.perf_counter()
        out_vertices, out_faces, out_uvs, out_vmaps = unwrap_bounded(
            mesh,
            compute_charts_kwargs={
                "threshold_cone_half_angle_rad": mesh_cluster_threshold_cone_half_angle_rad,
                "refine_iterations": mesh_cluster_refine_iterations,
                "global_iterations": mesh_cluster_global_iterations,
                "smooth_strength": mesh_cluster_smooth_strength,
            },
            return_vmaps=True,
            verbose=verbose, timings=timings,
            on_cpu_start=on_uv_cpu_start, cleanup_futures=uv_cleanup_futures,
            face_cap=uv_group_face_cap,
        )
        timings['uv_function_seconds'] = time.perf_counter() - uv_call_started
    # The UV CPU branch and material GPU branch join before baking. The
    # placeholder coords used for device selection must never reach sampling.
    if deferred_material is not None:
        material_wait_started = time.perf_counter()
        material = deferred_material()
        attr_volume, coords, attr_layout = material['attrs'], material['coords'], material['layout']
        timings['material_join_wait_seconds'] = time.perf_counter() - material_wait_started
    out_vertices = out_vertices.cuda()
    out_faces = out_faces.cuda()
    out_uvs = out_uvs.cuda()
    out_vmaps = out_vmaps.cuda()
    mesh.compute_vertex_normals()
    out_normals = mesh.read_vertex_normals()[out_vmaps]
    geometry_future=(geometry_export_prepare(out_vertices,out_faces,out_uvs,out_normals)
                     if geometry_export_prepare is not None else None)

    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")

    mark('uv_unwrap_seconds')
    # --- Texture Baking (Attribute Sampling) ---
    if use_tqdm:
        pbar.set_description("Sampling attributes")
    if verbose:
        print("Sampling attributes...", end='', flush=True)

    # Setup differentiable rasterizer context
    ctx = dr.RasterizeCudaContext()
    texture_events = []
    def texture_event(label):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        texture_events.append((label, event))
    texture_event('start')
    # Prepare UV coordinates for rasterization (rendering in UV space)
    uvs_rast = torch.cat([out_uvs * 2 - 1, torch.zeros_like(out_uvs[:, :1]), torch.ones_like(out_uvs[:, :1])], dim=-1).unsqueeze(0)
    rast = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)

    # Rasterize in chunks to save memory
    for i in range(0, out_faces.shape[0], 100000):
        rast_chunk, _ = dr.rasterize(
            ctx, uvs_rast, out_faces[i:i+100000],
            resolution=[texture_size, texture_size],
        )
        mask_chunk = rast_chunk[..., 3:4] > 0
        rast_chunk[..., 3:4] += i # Store face ID in alpha channel
        rast = torch.where(mask_chunk, rast_chunk, rast)
    texture_event('rasterize')

    # Mask of valid pixels in texture
    mask = rast[0, ..., 3] > 0
    mask_cpu = None
    mask_inv = None
    if on_texture_mask_ready is not None:
        if texture_padding != 'telea':
            raise ValueError('Deferred inpainting requires the native Telea mask')
        mask_cpu = mask.cpu().numpy()
        mask_inv = (~mask_cpu).astype(np.uint8)
        on_texture_mask_ready(mask_inv)

    # Interpolate 3D positions in UV space (finding 3D coord for every texel)
    pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
    valid_pos = pos[mask]
    texture_event('interpolate_and_mask')

    # Map these positions back to the *original* high-res mesh to get accurate attributes
    # This corrects geometric errors introduced by simplification/remeshing
    if project_texture_to_source:
        _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
        orig_tri_verts = vertices[faces[face_id.long()]] # (N_new, 3, 3)
        valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
    texture_event('source_projection')

    # Trilinear sampling from the attribute volume (Color, Material props)
    attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device='cuda')
    if texture_sampling == 'dense128':
        # Low-resolution material field for the explicitly lower-quality fast
        # profile. Average occupied source voxels, extend into a two-cell band,
        # then use the hardware-backed dense grid sampler without a BVH.
        import torch.nn.functional as functional
        resolution = 128
        cell = ((coords.float() + .5) / grid_size.float() * resolution).long().clamp(0, resolution - 1)
        flat = (cell[:, 0] * resolution + cell[:, 1]) * resolution + cell[:, 2]
        volume = torch.zeros((attr_volume.shape[1], resolution ** 3), device=coords.device)
        counts = torch.zeros(resolution ** 3, device=coords.device)
        volume.index_add_(1, flat, attr_volume.float().T)
        counts.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
        volume = (volume / counts.clamp_min(1)[None]).reshape(1, -1, resolution, resolution, resolution)
        occupied = (counts > 0).float().reshape(1, 1, resolution, resolution, resolution)
        for _ in range(2):
            weight = functional.avg_pool3d(occupied, 3, stride=1, padding=1)
            extended = functional.avg_pool3d(volume, 3, stride=1, padding=1) / weight.clamp_min(1e-8)
            volume = torch.where(occupied.bool(), volume, extended)
            occupied = (weight > 0).float()
        sample_grid = ((valid_pos - aabb[0]) / (aabb[1] - aabb[0]) * 2 - 1)[:, [2, 1, 0]]
        sampled = functional.grid_sample(volume, sample_grid.reshape(1, 1, 1, -1, 3),
            mode='bilinear', align_corners=False).reshape(attr_volume.shape[1], -1).T
        attrs[mask] = sampled
        del volume, counts, occupied
    elif texture_sampling == 'sparse':
        attrs[mask] = grid_sample_3d(
            attr_volume,
            torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
            shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
            grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
            mode='trilinear',
        )
    else:
        raise ValueError('texture_sampling must be sparse or dense128')
    texture_event('volume_sample')
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")

    mark('texture_bake_seconds')
    timings['texture_gpu_intervals'] = {
        name: previous.elapsed_time(event) / 1000
        for (_, previous), (name, event) in zip(texture_events, texture_events[1:])
    }
    # --- Texture Post-Processing & Material Construction ---
    if use_tqdm:
        pbar.set_description("Finalizing mesh")
    if verbose:
        print("Finalizing mesh...", end='', flush=True)

    mask = mask.cpu().numpy() if mask_cpu is None else mask_cpu

    # Extract channels based on layout (BaseColor, Metallic, Roughness, Alpha)
    if exact_channel_extract is None:
        base_color = np.clip(attrs[..., attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        metallic = np.clip(attrs[..., attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        roughness = np.clip(attrs[..., attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        alpha = np.clip(attrs[..., attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    else:
        base_color, metallic, roughness, alpha = exact_channel_extract(attrs, attr_layout)
    alpha_mode = 'OPAQUE'

    # Inpainting: fill gaps (dilation) to prevent black seams at UV boundaries
    if texture_padding == 'dilate':
        # Coarse geometry may query outside occupied voxels; treat missing
        # samples as gaps instead of publishing zero-valued black PBR pixels.
        mask = mask & (alpha[..., 0] > 0)
    if mask_inv is None:
        mask_inv = (~mask).astype(np.uint8)
    if verify_inpaint:
        audit_started = time.perf_counter()
        expected = [cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA),
            cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None],
            cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None],
            cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]]
        timings['audit_serial_inpaint_seconds'] = time.perf_counter() - audit_started
    inpaint_started = time.perf_counter()
    if texture_padding == 'dilate':
        packed = np.concatenate([base_color, metallic, roughness, alpha], axis=-1)
        packed = np.where(mask[..., None], packed, 0)
        valid = mask.astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        for _ in range(4):
            expanded = cv2.dilate(valid, kernel)
            frontier = (expanded != 0) & (valid == 0)
            padded = cv2.dilate(packed, kernel)
            packed[frontier] = padded[frontier]
            valid = expanded
        base_color, metallic, roughness, alpha = packed[..., :3], packed[..., 3:4], packed[..., 4:5], packed[..., 5:6]
    elif texture_padding != 'telea':
        raise ValueError('texture_padding must be telea or dilate')
    elif parallel_inpaint:
        inpaint = inpaint_pbr_channels if exact_inpaint is None else exact_inpaint
        base_color, metallic, roughness, alpha = inpaint(
            base_color, metallic, roughness, alpha, mask_inv)
    else:
        base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
        metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
        roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
        alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    timings['inpaint_seconds'] = time.perf_counter() - inpaint_started
    if verify_inpaint:
        comparisons = [np.array_equal(a, b) for a, b in zip(expected, [base_color, metallic, roughness, alpha])]
        timings['audit_inpaint_pixels_identical'] = all(comparisons)
        if not all(comparisons):
            raise ValueError('Parallel inpainting changed PBR pixels')
        del expected
        print('INPAINT_AUDIT ' + json.dumps({'serial_seconds': timings['audit_serial_inpaint_seconds'],
            'parallel_seconds': timings['inpaint_seconds'], 'all_pixels_identical': True}), flush=True)


    mark('texture_finish_seconds')
    assembly_started = time.perf_counter()
    # Create PBR material
    # Standard PBR packs Metallic and Roughness into Blue and Green channels
    packed_pixels=None if material_pixels is None else material_pixels()
    if packed_pixels is None:
        base_pixels=np.concatenate([base_color,alpha],axis=-1)
        mr_pixels=np.concatenate([np.zeros_like(metallic),roughness,metallic],axis=-1)
    else:
        base_pixels,mr_pixels=packed_pixels
    base_image=Image.fromarray(base_pixels)
    mr_image=Image.fromarray(mr_pixels)
    # These images are immutable between assembly and export in this request.
    # Keep the same raw pixels available to the lossless encoder without an
    # extra Pillow-to-NumPy payload copy.
    base_image._trellis_pixels=base_pixels
    mr_image._trellis_pixels=mr_pixels
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=base_image,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=mr_image,
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode=alpha_mode,
        doubleSided=True if not remesh else False,
    )

    timings['material_image_and_object_seconds'] = time.perf_counter() - assembly_started
    # --- Coordinate System Conversion & Final Object ---
    if geometry_future is None:
        vertices_np = out_vertices.cpu().numpy()
        faces_np = out_faces.cpu().numpy()
        uvs_np = out_uvs.cpu().numpy()
        normals_np = out_normals.cpu().numpy()

        # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
        vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
        normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
        uvs_np[:, 1] = 1 - uvs_np[:, 1] # Flip UV V-coordinate

        textured_mesh = trimesh.Trimesh(
            vertices=vertices_np,
            faces=faces_np,
            vertex_normals=normals_np,
            process=False,
            visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material)
        )
    else:
        geometry_join=time.perf_counter()
        textured_mesh=geometry_future.result()
        textured_mesh.visual.material=material
        timings['geometry_export_join_seconds']=time.perf_counter()-geometry_join
    cleanup_started = time.perf_counter()
    for cleanup in uv_cleanup_futures:
        cleanup.result()
    timings['uv_cleanup_wait_seconds'] = time.perf_counter() - cleanup_started
    progress_started = time.perf_counter()

    if use_tqdm:
        pbar.update(1)
        pbar.close()
    if verbose:
        print("Done")

    timings['final_progress_output_seconds'] = time.perf_counter() - progress_started
    mark('material_assembly_seconds')
    return textured_mesh
