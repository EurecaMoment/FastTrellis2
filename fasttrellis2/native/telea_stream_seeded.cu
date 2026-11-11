// Exact Telea arithmetic over the original FMM dependency order.
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

__device__ __forceinline__ bool known(const int* ranks,int row,int col,int stride,int current) {
    return ranks[row*stride+col]<current;
}

__device__ __forceinline__ int read_pixel(
    const unsigned char* original,const volatile unsigned char* output,
    const int* ranks,int* done,int row,int col,int width,int current,int channel) {
    const int p=(row+1)*(width+2)+col+1;
    const int rank=ranks[p];
    const int offset=(row*width+col)*3+channel;
    if(rank==0||rank>=current) return original[offset];
    // A producer fences all three byte stores before publishing the flag.
    // Read-only volatile polling avoids a contended read-modify-write for
    // every colour channel and every already-completed neighbour.
    while(reinterpret_cast<volatile int*>(done)[p]==0) { __nanosleep(64); }
    return output[offset];
}

__global__ void render_kernel(
    const unsigned char* original,unsigned char* output,const int* ranks,
    const float* distance,const int* sequence,int* done,int* counter,
    const float* weights,int height,int width,int count,int radius,int first) {
    const int lane=threadIdx.x&31;
    const unsigned full=0xffffffffu;
    const int stride=width+2;
    while(true) {
        int index=lane==0?atomicAdd(counter,1):0;
        index=__shfl_sync(full,index,0);
        if(index>=count) break;
        const int p=sequence[index],current=first+index+1;
        const int i=p/stride,j=p%stride;
        if(lane<3) {
            const int channel=lane;
            float gx=0,gy=0;
            if(known(ranks,i,j+1,stride,current)) {
                if(known(ranks,i,j-1,stride,current)) gx=(distance[p+1]-distance[p-1])*0.5f;
                else gx=distance[p+1]-distance[p];
            } else if(known(ranks,i,j-1,stride,current)) gx=distance[p]-distance[p-1];
            if(known(ranks,i+1,j,stride,current)) {
                if(known(ranks,i-1,j,stride,current)) gy=(distance[p+stride]-distance[p-stride])*0.5f;
                else gy=distance[p+stride]-distance[p];
            } else if(known(ranks,i-1,j,stride,current)) gy=distance[p]-distance[p-stride];
            float ia=0,jx=0,jy=0,sum=1.0e-20f;
            for(int k=i-radius;k<=i+radius;++k) {
                const int km=k-1+(k==1),kp=k-1-(k==height);
                for(int l=j-radius;l<=j+radius;++l) {
                    const int lm=l-1+(l==1),lp=l-1-(l==width);
                    const int squared=(l-j)*(l-j)+(k-i)*(k-i);
                    if(k<=0||l<=0||k>=height+1||l>=width+1||squared>radius*radius)continue;
                    if(!known(ranks,k,l,stride,current))continue;
                    const float rx=float(j-l),ry=float(i-k);
                    const float delta=distance[k*stride+l]-distance[p];
                    const float lev=float(1.0/(1.0+fabs(double(delta))));
                    float direction=rx*gx+ry*gy;
                    if(fabs(double(direction))<=0.01)direction=0.000001f;
                    const float w=float(fabs(double(weights[squared]*lev*direction)));
                    float ix=0,iy=0;
                    if(known(ranks,k,l+1,stride,current)) {
                        if(known(ranks,k,l-1,stride,current))
                            ix=float(read_pixel(original,output,ranks,done,km,lp+1,width,current,channel)-
                                     read_pixel(original,output,ranks,done,km,lm-1,width,current,channel))*2.0f;
                        else ix=float(read_pixel(original,output,ranks,done,km,lp+1,width,current,channel)-
                                      read_pixel(original,output,ranks,done,km,lm,width,current,channel));
                    } else if(known(ranks,k,l-1,stride,current))
                        ix=float(read_pixel(original,output,ranks,done,km,lp,width,current,channel)-
                                 read_pixel(original,output,ranks,done,km,lm-1,width,current,channel));
                    if(known(ranks,k+1,l,stride,current)) {
                        if(known(ranks,k-1,l,stride,current))
                            iy=float(read_pixel(original,output,ranks,done,kp+1,lm,width,current,channel)-
                                     read_pixel(original,output,ranks,done,km-1,lm,width,current,channel))*2.0f;
                        else iy=float(read_pixel(original,output,ranks,done,kp+1,lm,width,current,channel)-
                                      read_pixel(original,output,ranks,done,km,lm,width,current,channel));
                    } else if(known(ranks,k-1,l,stride,current))
                        iy=float(read_pixel(original,output,ranks,done,kp,lm,width,current,channel)-
                                 read_pixel(original,output,ranks,done,km-1,lm,width,current,channel));
                    ia+=w*float(read_pixel(original,output,ranks,done,k-1,l-1,width,current,channel));
                    jx-=w*(ix*rx);
                    jy-=w*(iy*ry);
                    sum+=w;
                }
            }
            const float sat=ia/sum+(jx+jy)/(sqrtf(jx*jx+jy*jy)+1.0e-20f);
            const int rounded=__double2int_rn(double(sat)+0.5);
            output[((i-1)*width+j-1)*3+channel]=static_cast<unsigned char>(min(255,max(0,rounded)));
            __threadfence();
        }
        __syncwarp(full);
        if(lane==0)atomicExch(done+p,1);
        __syncwarp(full);
    }
}

void launch_stream_render(uintptr_t image,uintptr_t output,uintptr_t rank,uintptr_t distance,
                   uintptr_t order,uintptr_t done,uintptr_t counter,uintptr_t weights,
                   int height,int width,int count,int radius,uintptr_t stream,int blocks,int first) {
    render_kernel<<<blocks,128,0,reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const unsigned char*>(image),reinterpret_cast<unsigned char*>(output),
        reinterpret_cast<const int*>(rank),reinterpret_cast<const float*>(distance),
        reinterpret_cast<const int*>(order),reinterpret_cast<int*>(done),reinterpret_cast<int*>(counter),
        reinterpret_cast<const float*>(weights),height,width,count,radius,first);
    const cudaError_t result=cudaGetLastError();
    if(result!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(result));
}

__global__ void update_plan_kernel(const int *positions,const float *values,int *rank,
    float *distance3,float *distance1,int first,int count) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const int p=positions[i];
    rank[p]=first+i+1;
    distance3[p]=values[i];
    distance1[p]=values[i];
}
void update_stream_plan(uintptr_t positions,uintptr_t values,uintptr_t rank,uintptr_t distance3,
    uintptr_t distance1,int first,int count,uintptr_t stream) {
    update_plan_kernel<<<(count+255)/256,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const int*>(positions),reinterpret_cast<const float*>(values),
        reinterpret_cast<int*>(rank),reinterpret_cast<float*>(distance3),
        reinterpret_cast<float*>(distance1),first,count);
    const auto result=cudaGetLastError();
    if(result!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(result));
}
