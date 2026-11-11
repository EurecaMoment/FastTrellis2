"""Stream a native single-mesh PBR GLB without redundant scene serialization."""
from concurrent.futures import ThreadPoolExecutor
import json
import struct
import time

import numpy as np
import trimesh

from trellis_export import export_glb_lossless
from trellis_png import encode_png


def prepare_geometry(mesh):
    begin=time.perf_counter()
    vertices=np.ascontiguousarray(mesh.vertices,dtype='<f4')
    normals=np.ascontiguousarray(trimesh.util.unitize(mesh.vertex_normals),dtype='<f4')
    uv=np.array(mesh.visual.uv,dtype='<f4',copy=True,order='C')
    uv[:,1]=1-uv[:,1]
    faces=np.ascontiguousarray(mesh.faces,dtype='<u4').reshape(-1)
    bounds=(vertices.min(axis=0).tolist(),vertices.max(axis=0).tolist())
    return (vertices,normals,uv,faces),bounds,time.perf_counter()-begin


def export_glb_direct(mesh,destination,compress_level=1):
    material=getattr(mesh.visual,'material',None)
    # The normal exporter remains the path for richer scenes/materials.
    if (not isinstance(material,trimesh.visual.material.PBRMaterial)
        or mesh.metadata or mesh.units is not None
        or material.baseColorTexture is None or material.metallicRoughnessTexture is None
        or any(getattr(material,name,None) is not None for name in
               ('normalTexture','occlusionTexture','emissiveTexture'))
        or getattr(mesh.visual,'uv',None) is None):
        return export_glb_lossless(mesh,destination,compress_level)
    started=time.perf_counter()
    prepared=getattr(mesh,'_trellis_prepared_export',None)
    with ThreadPoolExecutor(max_workers=3) as executor:
        geom_future=executor.submit(prepare_geometry,mesh) if prepared is None else None
        images=[executor.submit(encode_png,image,compress_level=compress_level) for image in
                (material.baseColorTexture,material.metallicRoughnessTexture)]
        arrays,bounds,geometry_seconds=geom_future.result() if geom_future is not None else prepared
        encoded=[future.result() for future in images]
    preparation=time.perf_counter()-started
    parts=[]
    views=[]
    accessors=[]
    offset=0
    def buffer_view(value,target=None):
        nonlocal offset
        data=memoryview(value).cast('B')
        length=data.nbytes
        item=dict(buffer=0,byteOffset=offset,byteLength=length)
        if target is not None:item['target']=target
        views.append(item)
        parts.append(data)
        padding=(-length)%4
        if padding:parts.append(memoryview(b'\0'*padding))
        offset+=length+padding
        return len(views)-1
    for name,array,kind in zip(('POSITION','NORMAL','TEXCOORD_0','indices'),arrays,('VEC3','VEC3','VEC2','SCALAR')):
        view=buffer_view(array,34963 if name=='indices' else 34962)
        accessor=dict(bufferView=view,byteOffset=0,componentType=5125 if name=='indices' else 5126,
            count=len(array),type=kind)
        if name=='POSITION':accessor.update(min=bounds[0],max=bounds[1])
        accessors.append(accessor)
    image_views=[buffer_view(value) for value in encoded]
    factor=material.baseColorFactor
    if factor is None:factor=[1.,1.,1.,1.]
    elif np.asarray(factor).dtype==np.uint8:factor=(np.asarray(factor,dtype=np.float64)/255).tolist()
    else:factor=np.asarray(factor).tolist()
    pbr=dict(baseColorTexture=dict(index=0),metallicRoughnessTexture=dict(index=1),baseColorFactor=factor,
        metallicFactor=1.0 if material.metallicFactor is None else float(material.metallicFactor),
        roughnessFactor=1.0 if material.roughnessFactor is None else float(material.roughnessFactor))
    mat=dict(pbrMetallicRoughness=pbr,doubleSided=bool(material.doubleSided),
        alphaMode=material.alphaMode or 'OPAQUE')
    if material.alphaCutoff is not None:mat['alphaCutoff']=float(material.alphaCutoff)
    if material.emissiveFactor is not None:mat['emissiveFactor']=np.asarray(material.emissiveFactor).tolist()
    if material.name:mat['name']=material.name
    tree=dict(asset=dict(version='2.0',generator='TRELLIS exact PBR export'),scene=0,
        scenes=[dict(nodes=[0])],nodes=[dict(mesh=0,name='geometry_0')],
        meshes=[dict(primitives=[dict(attributes=dict(POSITION=0,NORMAL=1,TEXCOORD_0=2),
            indices=3,material=0,mode=4)])],accessors=accessors,bufferViews=views,
        buffers=[dict(byteLength=offset)],materials=[mat],
        images=[dict(bufferView=view,mimeType='image/png') for view in image_views],
        textures=[dict(source=i) for i in range(2)])
    document=json.dumps(tree,separators=(',',':')).encode('utf-8')
    document+=b' '*((-len(document))%4)
    length=12+8+len(document)+8+offset
    packed_started=time.perf_counter()
    with open(destination,'wb') as handle:
        handle.write(struct.pack('<4sII',b'glTF',2,length))
        handle.write(struct.pack('<I4s',len(document),b'JSON'))
        handle.write(document)
        handle.write(struct.pack('<I4s',offset,b'BIN\0'))
        for part in parts:handle.write(part)
    return dict(png_encode_seconds=preparation,geometry_prepare_seconds=geometry_seconds,
        geometry_prepare_overlapped=prepared is not None,
        glb_pack_and_write_seconds=time.perf_counter()-packed_started,png_compress_level=compress_level,
        png_images=2,png_encoder=getattr(encode_png,'backend_name','parallel-deflate-with-pillow-fallback'),
        serialization_seconds=time.perf_counter()-started,serializer='direct-native-mesh-pbr')
