"""Prepare this request's export geometry while its material is being baked."""
from concurrent.futures import ThreadPoolExecutor

import torch
import trimesh

from trellis_glb_direct import prepare_geometry


class PipelinedGeometryExport:
    def __init__(self):
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='trellis-export-geometry')
        self.copy_stream=torch.cuda.Stream()

    def __call__(self,vertices,faces,uvs,normals):
        ready=torch.cuda.Event()
        ready.record()
        def work():
            with torch.inference_mode(),torch.cuda.device(vertices.device),torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(ready)
                vertices_np,faces_np,uvs_np,normals_np=[value.cpu().numpy() for value in (vertices,faces,uvs,normals)]
            vertices_np[:,1],vertices_np[:,2]=vertices_np[:,2],-vertices_np[:,1]
            normals_np[:,1],normals_np[:,2]=normals_np[:,2],-normals_np[:,1]
            uvs_np[:,1]=1-uvs_np[:,1]
            mesh=trimesh.Trimesh(vertices=vertices_np,faces=faces_np,vertex_normals=normals_np,
                                process=False,visual=trimesh.visual.TextureVisuals(uv=uvs_np))
            mesh._trellis_prepared_export=prepare_geometry(mesh)
            return mesh
        return self.executor.submit(work)

    def close(self):
        self.executor.shutdown(wait=True,cancel_futures=True)
        self.copy_stream.synchronize()
