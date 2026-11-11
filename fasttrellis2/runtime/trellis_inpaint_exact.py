"""Original OpenCV Telea ordering with dependency-ordered CUDA arithmetic.

This backend keeps the native radius 3 RGB and radius 1 material channels.
The extension is built before service startup, never inside a request.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch


class ExactTelea:
    def __init__(self, blocks=216):
        import trellis_telea_gpu
        self.extension=trellis_telea_gpu
        self.blocks=blocks
        self.streams=(torch.cuda.Stream(),torch.cuda.Stream())
        self.weights={}
        for radius in (1,3):
            value=np.zeros(radius*radius+1,np.float32)
            for squared in range(1,len(value)):
                value[squared]=1.0/(np.float64(squared)*np.sqrt(np.float64(squared)))
            self.weights[radius]=torch.from_numpy(value).cuda()
        self.last_timing={}
        self.plan_executor=None
        self.pending_plan=None

    def start_plan(self,mask):
        """Start exact CPU FMM as soon as rasterization supplies the mask."""
        if self.pending_plan is not None:
            raise RuntimeError('Previous inpaint plan has not been consumed')
        if not np.any(mask) or np.all(mask):
            return
        if self.plan_executor is None:
            self.plan_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='trellis-inpaint-plan')
        def work():
            started=time.perf_counter()
            prepared=self.prepare(mask)
            return prepared,started,time.perf_counter()
        self.pending_plan=(mask,self.plan_executor.submit(work))

    def close(self):
        self.cancel_plan()
        if self.plan_executor is not None:
            self.plan_executor.shutdown(wait=True,cancel_futures=True)
        self.plan_executor=None

    def cancel_plan(self):
        pending=self.pending_plan
        self.pending_plan=None
        if pending is not None and not pending[1].cancel():
            pending[1].exception()

    def prepare(self,mask):
        flags=np.pad((mask!=0).astype(np.uint8)*2,1)
        band=cv2.dilate(flags,cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)))-flags
        band[[0,-1],:]=0
        band[:,[0,-1]]=0
        times=[]
        outside=[]
        for radius in (3,1):
            distance=np.full(flags.shape,1e6,np.float32)
            distance[band!=0]=0
            outer=cv2.dilate(flags,np.ones((2*radius+1,2*radius+1),np.uint8))
            outer=cv2.subtract(cv2.subtract(outer,flags),band)
            outer[[0,-1],:]=0
            outer[:,[0,-1]]=0
            times.append(distance)
            outside.append(outer)
        rank=np.empty(flags.shape,np.int32)
        order=np.empty(np.count_nonzero(mask),np.int32)
        details=self.extension.plan(flags,times[0],times[1],band,outside[0],outside[1],rank,order)
        return rank,order[:details['filled']],times,details

    def render(self,image,rank,order,distance,radius):
        tensor=torch.from_numpy(np.ascontiguousarray(image)).cuda()
        result=tensor.clone()
        done=torch.zeros_like(rank)
        counter=torch.zeros(1,device='cuda',dtype=torch.int32)
        height,width=image.shape[:2]
        self.extension.render(tensor.data_ptr(),result.data_ptr(),rank.data_ptr(),distance.data_ptr(),
            order.data_ptr(),done.data_ptr(),counter.data_ptr(),self.weights[radius].data_ptr(),
            height,width,order.numel(),radius,torch.cuda.current_stream().cuda_stream,self.blocks)
        # Keep raw-pointer inputs alive until this request has synchronized.
        return result,(tensor,done,counter)

    def __call__(self,base_color,metallic,roughness,alpha,mask):
        started=time.perf_counter()
        inputs=(base_color,metallic,roughness,alpha)
        if mask.ndim!=2 or base_color.shape!=(*mask.shape,3):
            raise ValueError('Telea input dimensions do not match')
        if any(value.dtype!=np.uint8 or value.shape[:2]!=mask.shape for value in inputs):
            raise ValueError('Telea requires matching uint8 PBR channels')
        if not np.any(mask) or np.all(mask):
            self.last_timing={'seconds':time.perf_counter()-started,'filled':0}
            return tuple(np.array(value,copy=True) for value in inputs)
        plan_overlap=0.0
        if self.pending_plan is None:
            rank,order,times,details=self.prepare(mask)
            planning=time.perf_counter()-started
        else:
            expected_mask,future=self.pending_plan
            self.pending_plan=None
            if expected_mask is not mask:
                raise RuntimeError('Deferred inpaint plan belongs to another mask')
            prepared,plan_started,plan_ended=future.result()
            rank,order,times,details=prepared
            planning=plan_ended-plan_started
            plan_overlap=max(0.0,min(started,plan_ended)-plan_started)
        plan_wait=time.perf_counter()-started
        scalar_image=np.stack([value.reshape(mask.shape) for value in inputs[1:]],axis=-1)
        rank=torch.from_numpy(rank).cuda()
        order=torch.from_numpy(order).cuda()
        times=[torch.from_numpy(value).cuda() for value in times]
        current=torch.cuda.current_stream()
        outputs=[]
        retained=[]
        for stream,image,distance,radius in zip(self.streams,(base_color,scalar_image),times,(3,1)):
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                value,keep=self.render(image,rank,order,distance,radius)
            outputs.append(value)
            retained.append(keep)
        for stream in self.streams:current.wait_stream(stream)
        rgb,scalars=[value.cpu().numpy() for value in outputs]
        self.last_timing={'seconds':time.perf_counter()-started,'planning_seconds':planning,
                          'planning_wait_seconds':plan_wait,'planning_overlap_seconds':plan_overlap,
                          'filled':details['filled'],'backend':'original-telea-fmm-cuda'}
        return rgb,scalars[...,0:1],scalars[...,1:2],scalars[...,2:3]
