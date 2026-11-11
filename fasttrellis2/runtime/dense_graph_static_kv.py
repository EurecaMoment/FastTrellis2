"""Prepare fixed dense execution graphs while refreshing every image's KV."""
import time
import torch
import torch.nn.functional as functional


class DenseGraphStaticKV:
    def __init__(self, model, conditioning_shape):
        self.model=model
        self.original=model.forward
        self.raw_forward=type(model).forward.__get__(model,type(model))
        self.entries=[]
        self.conditions={}
        self.projections=[block.cross_attn.to_kv for block in model.blocks]
        self.refreshes=0
        self.replays=0
        started=time.perf_counter()
        for _slot in range(2):
            x=torch.zeros(1,model.in_channels,*([model.resolution]*3),device=model.device)
            t=torch.full((1,),1000.0,device=model.device)
            cond=torch.zeros(conditioning_shape,device=model.device,dtype=model.dtype)
            buffers=[functional.linear(cond,linear.weight,linear.bias) for linear in self.projections]
            restores=[]
            try:
                for linear,buffer in zip(self.projections,buffers):
                    restores.append((linear,linear.forward))
                    linear.forward=lambda value,_buffer=buffer:_buffer
                stream=torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):self.raw_forward(x,t,cond)
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):
                    result=self.raw_forward(x,t,cond)
                torch.cuda.synchronize()
                self.entries.append((graph,x,t,cond,buffers,result))
            finally:
                for linear,forward in restores:linear.forward=forward
        self.preparation_seconds=time.perf_counter()-started
        model.forward=self.forward

    def clear(self):
        self.conditions.clear()

    def forward(self,x,t,cond):
        if x.shape!=self.entries[0][1].shape or cond.shape!=self.entries[0][3].shape:
            return self.original(x,t,cond)
        key=id(cond)
        if key not in self.conditions:
            if len(self.conditions)==len(self.entries):
                return self.original(x,t,cond)
            index=len(self.conditions)
            entry=self.entries[index]
            converted=cond.to(dtype=self.model.dtype)
            entry[3].copy_(converted)
            for linear,buffer in zip(self.projections,entry[4]):
                buffer.copy_(functional.linear(converted,linear.weight,linear.bias))
            self.conditions[key]=(cond,index)
            self.refreshes+=1
        index=self.conditions[key][1]
        graph,static_x,static_t,_,_,result=self.entries[index]
        static_x.copy_(x)
        static_t.copy_(t)
        graph.replay()
        self.replays+=1
        return result.clone()

    def close(self):
        self.model.forward=self.original
        self.clear()
        self.entries.clear()
