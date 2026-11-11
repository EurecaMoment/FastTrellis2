"""Overlap native FMM traversal with native dependency-ordered CUDA colors."""
import time
import importlib

import cv2
import numpy as np
import torch

from trellis_inpaint_exact import ExactTelea


class StreamingTelea(ExactTelea):
    def __init__(self,blocks=432,batch=524288,module_name='trellis_telea_stream',async_copy=False,pack_material=False):
        super().__init__(blocks=blocks)
        self.streaming=importlib.import_module(module_name)
        self.batch=batch
        self.pending_stream=None
        self.async_copy=async_copy
        self.upload_stream=torch.cuda.Stream() if async_copy else None
        self.pack_material=pack_material
        self.last_material_pixels=None

    def start_plan(self,mask):
        if self.pending_stream is not None:
            raise RuntimeError('Previous streaming plan has not been consumed')
        if not np.any(mask) or np.all(mask):return
        started=time.perf_counter()
        flags=np.pad((mask!=0).astype(np.uint8)*2,1)
        rank=np.where(flags==2,np.int32(2147483647),np.int32(0))
        band=cv2.dilate(flags,cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)))-flags
        band[[0,-1],:]=0
        band[:,[0,-1]]=0
        times=[]
        outer=[]
        for radius in (3,1):
            distance=np.full(flags.shape,1e6,np.float32)
            distance[band!=0]=0
            outside=cv2.dilate(flags,np.ones((2*radius+1,2*radius+1),np.uint8))
            outside=cv2.subtract(cv2.subtract(outside,flags),band)
            outside[[0,-1],:]=0
            outside[:,[0,-1]]=0
            times.append(distance)
            outer.append(outside)
        plan=self.streaming.StreamingPlan(flags,times[0],times[1],band,outer[0],outer[1],self.batch)
        self.pending_stream=(mask,plan,rank,times,started)

    def cancel_plan(self):
        pending=self.pending_stream
        self.pending_stream=None
        if pending is not None:pending[1].close()
        super().cancel_plan()

    def __call__(self,base_color,metallic,roughness,alpha,mask):
        started=time.perf_counter()
        self.last_material_pixels=None
        inputs=(base_color,metallic,roughness,alpha)
        if not np.any(mask) or np.all(mask):
            self.last_timing=dict(seconds=time.perf_counter()-started,filled=0)
            return tuple(np.array(value,copy=True) for value in inputs)
        if self.pending_stream is None:self.start_plan(mask)
        expected,plan,rank,times,plan_started=self.pending_stream
        self.pending_stream=None
        if expected is not mask:
            plan.close()
            raise RuntimeError('Streaming plan belongs to another mask')
        retained=[]
        try:
            scalar=np.stack([value.reshape(mask.shape) for value in inputs[1:]],axis=-1)
            device_inputs=[torch.from_numpy(np.ascontiguousarray(value)).cuda() for value in (base_color,scalar)]
            outputs=[value.clone() for value in device_inputs]
            rank=torch.from_numpy(rank).cuda()
            done=[torch.zeros_like(rank) for _ in range(2)]
            counters=[torch.zeros(1,dtype=torch.int32,device='cuda') for _ in range(2)]
            plan.wait_outer()
            times=[torch.from_numpy(value).cuda() for value in times]
            current=torch.cuda.current_stream()
            first_render_offset=None
            chunks=0
            while True:
                chunk=plan.next()
                if chunk is None:break
                first,positions,values=chunk
                # All changes to shared rank/distance arrays follow the prior
                # two color kernels. The streams never read a concurrently
                # modified dependency map.
                pinned=None
                if self.async_copy:
                    # Upload only independent chunk buffers while the previous
                    # color kernels run. Shared dependency maps are updated
                    # after the same color-stream joins as the original path.
                    pinned=(torch.from_numpy(positions).pin_memory(),torch.from_numpy(values).pin_memory())
                    with torch.cuda.stream(self.upload_stream):
                        positions=pinned[0].to('cuda',non_blocking=True)
                        values=pinned[1].to('cuda',non_blocking=True)
                    current.wait_stream(self.upload_stream)
                    for stream in self.streams:current.wait_stream(stream)
                else:
                    for stream in self.streams:current.wait_stream(stream)
                    positions=torch.from_numpy(positions).cuda()
                    values=torch.from_numpy(values).cuda()
                self.streaming.update(positions.data_ptr(),values.data_ptr(),rank.data_ptr(),
                    times[0].data_ptr(),times[1].data_ptr(),first,positions.numel(),current.cuda_stream)
                for counter in counters:counter.zero_()
                for i,(stream,radius) in enumerate(zip(self.streams,(3,1))):
                    stream.wait_stream(current)
                    self.streaming.render(device_inputs[i].data_ptr(),outputs[i].data_ptr(),rank.data_ptr(),
                        times[i].data_ptr(),positions.data_ptr(),done[i].data_ptr(),counters[i].data_ptr(),
                        self.weights[radius].data_ptr(),*mask.shape,positions.numel(),radius,
                        stream.cuda_stream,self.blocks,first)
                if first_render_offset is None:first_render_offset=time.perf_counter()-plan_started
                # Raw-pointer inputs are retained through both stream joins.
                retained.append((positions,values,pinned))
                chunks+=1
            for stream in self.streams:current.wait_stream(stream)
            if self.pack_material:
                rgba=torch.cat((outputs[0],outputs[1][...,2:3]),dim=-1)
                mr=torch.cat((torch.zeros_like(outputs[1][...,0:1]),outputs[1][...,1:2],outputs[1][...,0:1]),dim=-1)
                rgba_cpu,mr_cpu=[value.cpu().numpy() for value in (rgba,mr)]
                self.last_material_pixels=(rgba_cpu,mr_cpu)
                result=(rgba_cpu[...,:3],mr_cpu[...,2:3],mr_cpu[...,1:2],rgba_cpu[...,3:4])
            else:
                rgb,scalars=[value.cpu().numpy() for value in outputs]
                result=(rgb,scalars[...,0:1],scalars[...,1:2],scalars[...,2:3])
            details=plan.stats()
            self.last_timing=dict(seconds=time.perf_counter()-started,
                pipeline_wall_seconds=time.perf_counter()-plan_started,first_render_offset=first_render_offset,
                chunks=chunks,backend='streamed-original-telea',**details)
            return result
        finally:
            for stream in self.streams:stream.synchronize()
            if self.upload_stream is not None:
                self.upload_stream.synchronize()
                torch.cuda.current_stream().synchronize()
            plan.close()
