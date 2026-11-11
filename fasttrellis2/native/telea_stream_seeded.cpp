#include <immintrin.h>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <memory>
#include <atomic>
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective icvers.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/* ////////////////////////////////////////////////////////////////////
//
//  Geometrical transforms on images and matrices: rotation, zoom etc.
//
// */


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <chrono>
#include <thread>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <queue>
#include <type_traits>
#include <vector>
namespace py = pybind11;
using uchar = unsigned char;
// OpenCV's math.h exposes the float overloads in the global namespace.
using std::sqrt;
using std::fabs;
#define MIN(a,b) ((a) > (b) ? (b) : (a))
namespace cv {
struct Point2f { float x, y; };
template<typename T, int Count> struct Vec {
    T values[Count];
    T& operator[](int index) { return values[index]; }
    const T& operator[](int index) const { return values[index]; }
};
struct Mat {
    int rows, cols, count;
    size_t step;
    uint8_t* data;
    template<typename T> T& at(int row, int col) {
        return reinterpret_cast<T*>(data + row * step)[col];
    }
    template<typename T> const T& at(int row, int col) const {
        return reinterpret_cast<const T*>(data + row * step)[col];
    }
    int channels() const { return count; }
};
template<typename T> T saturate_cast(double value) {
    const long rounded = std::lrint(value);
    return static_cast<T>(std::min(255l, std::max(0l, rounded)));
}
}
using namespace cv;
#define sqrt(value) ::std::sqrt(value)
#define fabs(value) ::std::fabs(static_cast<double>(value))
template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type round_cast(float val) {
   return cv::saturate_cast<T>(val);
}

template<typename T>
typename std::enable_if<!std::is_floating_point<T>::value, T>::type round_cast(float val) {
   return cv::saturate_cast<T>(val + 0.5);
}

inline float
min4( float a, float b, float c, float d )
{
    a = MIN(a,b);
    c = MIN(c,d);
    return MIN(a,c);
}

#define KNOWN  0  //known outside narrow band
#define BAND   1  //narrow band (known)
#define INSIDE 2  //unknown
#define CHANGE 3  //servise

typedef struct CvHeapElem
{
    float T;
    int i,j;
    int order;  // to keep insertion order

    bool operator > (const CvHeapElem& rhs) const {
        if (T > rhs.T) {
            return true;
        } else if (T < rhs.T) {
            return false;
        }
        return order > rhs.order;
    }
}
CvHeapElem;




struct CompactHeapElem {
    float T;
    uint16_t i,j;
    int order;
    explicit CompactHeapElem(const CvHeapElem &value)
        :T(value.T),i(uint16_t(value.i)),j(uint16_t(value.j)),order(value.order) {}
    CvHeapElem expand() const { return {T,int(i),int(j),order}; }
};
static_assert(sizeof(CompactHeapElem)==12,"Unexpected heap node padding");

struct SharedSeed { int i,j; };
static std::vector<SharedSeed> initialSeeds(const Mat &initial) {
    std::vector<SharedSeed> result;
    result.reserve(size_t(initial.rows)*initial.cols/2);
    for(int i=1;i<initial.rows-1;++i)
        for(int j=1;j<initial.cols-1;++j)
            if(initial.at<uchar>(i,j))result.push_back({i,j});
    return result;
}
static std::vector<SharedSeed> outsideSeeds(Mat &outer,const Mat &initial) {
    std::vector<SharedSeed> result;
    result.reserve(size_t(initial.rows)*initial.cols/8);
    const int cols=initial.cols;
    const __m512i inside=_mm512_set1_epi8(INSIDE),changed=_mm512_set1_epi8(CHANGE);
    for(int i=1;i<initial.rows-1;++i) {
        int j=0;
        for(;j+64<=cols;j+=64) {
            const int p=i*cols+j;
            const __m512i band=_mm512_loadu_si512(initial.data+p);
            const __mmask64 seeds=_mm512_cmpneq_epi8_mask(band,_mm512_setzero_si512());
            if(!seeds)continue;
            const __mmask64 near=
                _mm512_cmpeq_epi8_mask(_mm512_loadu_si512(outer.data+p-cols),inside)|
                _mm512_cmpeq_epi8_mask(_mm512_loadu_si512(outer.data+p-1),inside)|
                _mm512_cmpeq_epi8_mask(_mm512_loadu_si512(outer.data+p+cols),inside)|
                _mm512_cmpeq_epi8_mask(_mm512_loadu_si512(outer.data+p+1),inside);
            uint64_t selected=seeds&near;
            // BAND/KNOWN/CHANGE are equivalent to the distance solver. Mark
            // all original seeds for the unchanged final sign-negation pass.
            _mm512_mask_storeu_epi8(outer.data+p,seeds,changed);
            while(selected) {
                const unsigned n=__builtin_ctzll(selected);
                result.push_back({i,j+int(n)});
                selected&=selected-1;
            }
        }
        for(;j<cols;++j)if(initial.at<uchar>(i,j)) {
            const bool near=outer.at<uchar>(i-1,j)==INSIDE || outer.at<uchar>(i,j-1)==INSIDE ||
                            outer.at<uchar>(i+1,j)==INSIDE || outer.at<uchar>(i,j+1)==INSIDE;
            outer.at<uchar>(i,j)=CHANGE;
            if(near)result.push_back({i,j});
        }
    }
    return result;
}
class CvPriorityQueueFloat {
    const std::vector<SharedSeed> *seeds=nullptr;
    size_t seed_head=0;

    std::array<std::vector<CompactHeapElem>,33> buckets;
    std::vector<CompactHeapElem> redistribution;
    std::array<uint32_t,33> minima;
    std::priority_queue<CvHeapElem,std::vector<CvHeapElem>,std::greater<CvHeapElem>> fallback;
    uint32_t last=0;
    size_t remaining=0, head=0;
    int next_order=0;
    bool ordinary=false;
    static uint32_t key(float value) {
        if (value==0) value=0;
        uint32_t bits;
        std::memcpy(&bits,&value,sizeof(bits));
        return bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
    }
    static int bucket(uint32_t value,uint32_t previous) {
        const uint32_t different=value^previous;
        return different ? 32-__builtin_clz(different) : 0;
    }
    void switch_to_ordinary() {
        ordinary=true;
        if(seeds) {
            while(seed_head<seeds->size()) {
                const auto value=(*seeds)[seed_head];
                fallback.push(CvHeapElem{0.0f,value.i,value.j,int(seed_head++)});
                ++remaining;
            }
            seeds=nullptr;
        }

        for (int b=0;b<33;++b) {
            for (size_t n=(b==0?head:0);n<buckets[b].size();++n)
                fallback.push(buckets[b][n].expand());
            buckets[b].clear();
        }
        head=0;
    }
    bool pop(CvHeapElem &result) {

        if(!ordinary && seeds && seed_head<seeds->size()) {
            const auto value=(*seeds)[seed_head];
            result=CvHeapElem{0.0f,value.i,value.j,int(seed_head++)};
            return true;
        }
        if (!remaining) return false;
        if (ordinary) {
            result=fallback.top();
            fallback.pop();
        } else {
            if (head==buckets[0].size()) {
                buckets[0].clear();
                head=0;
                int b=1;
                while (b<33 && buckets[b].empty()) ++b;
                if (b==33) return false;
                last=minima[b];
                // Lower buckets are empty. Stable redistribution preserves the
                // original insertion-order tie break for equal float distances.
                redistribution.clear();
                redistribution.swap(buckets[b]);
                minima[b]=UINT32_MAX;
                for (const auto &value:redistribution) {
                    uint32_t distance=key(value.T);
                    int target=bucket(distance,last);
                    minima[target]=std::min(minima[target],distance);
                    buckets[target].push_back(value);
                }
            }
            result=buckets[0][head++].expand();
        }
        --remaining;
        return true;
    }
public:
    CvPriorityQueueFloat() { minima.fill(UINT32_MAX); }
    void AddSeeds(const std::vector<SharedSeed> &initial) {
        seeds=&initial;
        seed_head=0;
        next_order=int(initial.size());
        last=key(0.0f);
    }

    bool Add(const Mat &flags) {
        for (int i=0;i<flags.rows;++i)
            for (int j=0;j<flags.cols;++j)
                if (flags.at<uchar>(i,j)!=0) Push(i,j,0);
        return true;
    }
    bool Push(int i,int j,float time) {
        CvHeapElem value{time,i,j,next_order++};
        uint32_t distance=key(time);
        // The normal FMM traversal is monotone. Handle any exception using
        // the unmodified heap, including all nodes already in radix buckets.
        if (!ordinary && (std::isnan(time) || distance<last || uint32_t(i)>65535 || uint32_t(j)>65535)) switch_to_ordinary();
        if (ordinary) fallback.push(value);
        else {
            int b=bucket(distance,last);
            buckets[b].push_back(CompactHeapElem(value));
            minima[b]=std::min(minima[b],distance);
        }
        ++remaining;
        return true;
    }
    bool Pop(int *i,int *j) {
        CvHeapElem value;
        if (!pop(value)) return false;
        *i=value.i;*j=value.j;
        return true;
    }
    bool Pop(int *i,int *j,float *time) {
        CvHeapElem value;
        if (!pop(value)) return false;
        *i=value.i;*j=value.j;*time=value.T;
        return true;
    }
};

static inline float VectorScalMult(const cv::Point2f& v1, const cv::Point2f& v2)
{
   return v1.x*v2.x+v1.y*v2.y;
}

static inline float VectorLength(const cv::Point2f& v1)
{
    return v1.x*v1.x+v1.y*v1.y;
}

///////////////////////////////////////////////////////////////////////////////////////////
//HEAP::iterator Heap_Iterator;
//HEAP Heap;

static float FastMarching_solve(int i1,int j1,int i2,int j2, const Mat &f, const Mat &t)
{
    double sol, a11, a22, m12;
    a11=t.at<float>(i1,j1);
    a22=t.at<float>(i2,j2);
    m12=MIN(a11,a22);

    if( f.at<uchar>(i1,j1) != INSIDE )
        if( f.at<uchar>(i2,j2) != INSIDE )
            if( fabs(a11-a22) >= 1.0 )
                sol = 1+m12;
            else
                sol = (a11+a22+sqrt((double)(2-(a11-a22)*(a11-a22))))*0.5;
        else
            sol = 1+a11;
    else if( f.at<uchar>(i2,j2) != INSIDE )
        sol = 1+a22;
    else
        sol = 1+m12;

    return (float)sol;
}

/////////////////////////////////////////////////////////////////////////////////////



static float FastMarching_min4(int i, int j, const Mat& f, const Mat& t) {
    const int up = i-1, down = i+1, left = j-1, right = j+1;
    const bool ku=f.at<uchar>(up,j)!=INSIDE, kd=f.at<uchar>(down,j)!=INSIDE;
    const bool kl=f.at<uchar>(i,left)!=INSIDE, kr=f.at<uchar>(i,right)!=INSIDE;
    const double u=t.at<float>(up,j), d=t.at<float>(down,j);
    const double l=t.at<float>(i,left), r=t.at<float>(i,right);
    const bool vertical=ku||kd, horizontal=kl||kr;
    const double a=ku?(kd?MIN(u,d):u):d;
    const double b=kl?(kr?MIN(l,r):l):r;
    if (vertical && horizontal) {
        const double delta=a-b;
        if (fabs(delta)>=1.0) return float(1+MIN(a,b));
        return float((a+b+sqrt(double(2-delta*delta)))*0.5);
    }
    if (vertical) return float(1+a);
    if (horizontal) return float(1+b);
    return float(1+MIN(MIN(u,d),MIN(l,r)));
}

static void
icvCalcFMM(Mat &f, Mat &t, CvPriorityQueueFloat *Heap, bool negate) {
   int i, j, ii = 0, jj = 0, q;
   float dist;

   while (Heap->Pop(&ii,&jj)) {

      unsigned known=(negate)?CHANGE:KNOWN;
      f.at<uchar>(ii,jj) = (uchar)known;

      for (q=0; q<4; q++) {
         i=0; j=0;
         if     (q==0) {i=ii-1; j=jj;}
         else if(q==1) {i=ii;   j=jj-1;}
         else if(q==2) {i=ii+1; j=jj;}
         else {i=ii;   j=jj+1;}
         if ((i<=0)||(j<=0)||(i>f.rows)||(j>f.cols)) continue;

         if (f.at<uchar>(i,j)==INSIDE) {
            dist = FastMarching_min4(i,j,f,t);
            t.at<float>(i,j) = dist;
            f.at<uchar>(i,j) = BAND;
            Heap->Push(i,j,dist);
         }
      }
   }

   if (negate) {
      for (i=0; i<f.rows; i++) {
         for(j=0; j<f.cols; j++) {
            if (f.at<uchar>(i,j) == CHANGE) {
               f.at<uchar>(i,j) = KNOWN;
               t.at<float>(i,j) = -t.at<float>(i,j);
            }
         }
      }
   }
}




static float stream_min4(int pixel,int stride,const float *state) {
    const float uf=state[pixel-stride],df=state[pixel+stride];
    const float lf=state[pixel-1],rf=state[pixel+1];
    const bool ku=uf>=0,kd=df>=0,kl=lf>=0,kr=rf>=0;
    // Original unvisited inner pixels have time 1e6. KNOWN and BAND are
    // equivalent in FastMarching_solve, so their separate flag is redundant.
    const double u=ku?uf:1e6f,d=kd?df:1e6f,l=kl?lf:1e6f,r=kr?rf:1e6f;
    const bool vertical=ku||kd,horizontal=kl||kr;
    const double a=ku?(kd?MIN(u,d):u):d;
    const double b=kl?(kr?MIN(l,r):l):r;
    if(vertical && horizontal) {
        const double delta=a-b;
        if(fabs(delta)>=1.0)return float(1+MIN(a,b));
        return float((a+b+sqrt(double(2-delta*delta)))*.5);
    }
    if(vertical)return float(1+a);
    if(horizontal)return float(1+b);
    return float(1+MIN(MIN(u,d),MIN(l,r)));
}

struct StreamChunk {
    int first;
    std::vector<int32_t> positions;
    std::vector<float> distances;
};

class StreamingPlan {
    using Bytes=py::array_t<uint8_t,py::array::c_style>;
    using Floats=py::array_t<float,py::array::c_style>;
    Bytes flags,band,outside3,outside1;
    Floats distance3,distance1;
    int rows,cols,batch;
    std::thread worker;
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<std::shared_ptr<StreamChunk>> queue;
    bool cancelled=false,finished=false;
    int outer_count=0,filled=0;
    std::exception_ptr error;
    double total_seconds=0,outer_seconds=0,first_chunk_seconds=0;

    bool publish(const std::shared_ptr<StreamChunk>& chunk) {
        std::unique_lock<std::mutex> lock(mutex);
        changed.wait(lock,[&]{return queue.size()<16 || cancelled;});
        if(cancelled)return false;
        queue.push_back(chunk);
        changed.notify_all();
        return true;
    }

    void run() {
        const auto started=std::chrono::steady_clock::now();
        Mat f{rows,cols,1,size_t(cols),flags.mutable_data()};
        Mat t3{rows,cols,1,size_t(cols)*sizeof(float),reinterpret_cast<uint8_t*>(distance3.mutable_data())};
        Mat t1{rows,cols,1,size_t(cols)*sizeof(float),reinterpret_cast<uint8_t*>(distance1.mutable_data())};
        Mat initial{rows,cols,1,size_t(cols),band.mutable_data()};
        Mat outer3{rows,cols,1,size_t(cols),outside3.mutable_data()};
        Mat outer1{rows,cols,1,size_t(cols),outside1.mutable_data()};
        std::vector<float> inside(distance3.data(),distance3.data()+rows*cols);
        for(int p=0;p<rows*cols;++p)
            if(flags.data()[p]==INSIDE)inside[p]=-1.0f;
        const auto seedPixels=initialSeeds(initial);
        auto outside=[&](Mat &outer,Mat &distance) {
            try {
                CvPriorityQueueFloat heap;
                const auto selected=outsideSeeds(outer,initial);
                heap.AddSeeds(selected);
                icvCalcFMM(outer,distance,&heap,true);
            } catch(...) {
                std::lock_guard<std::mutex> lock(mutex);
                error=std::current_exception();
            }
            std::lock_guard<std::mutex> lock(mutex);
            if(++outer_count==2)outer_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
            changed.notify_all();
        };
        std::thread a([&]{outside(outer3,t3);}),b([&]{outside(outer1,t1);});
        try {
            CvPriorityQueueFloat heap;
            heap.AddSeeds(seedPixels);
            auto chunk=std::make_shared<StreamChunk>();
            chunk->first=0;
            chunk->positions.reserve(batch);
            chunk->distances.reserve(batch);
            int ii,jj;
            bool stop=false;
            while(!stop && heap.Pop(&ii,&jj)) {

                for(int q=0;q<4;++q) {
                    int i=ii,j=jj;
                    if(q==0)--i;else if(q==1)--j;else if(q==2)++i;else ++j;
                    if(i<=0||j<=0||i>rows-1||j>cols-1)continue;
                    if(inside[i*cols+j]<0.0f) {
                        const float dist=stream_min4(i*cols+j,cols,inside.data());
                        inside[i*cols+j]=dist;
                        chunk->positions.push_back(i*cols+j);
                        chunk->distances.push_back(dist);
                        ++filled;

                        heap.Push(i,j,dist);
                        if(int(chunk->positions.size())==batch) {
                            if(first_chunk_seconds==0)first_chunk_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
                            if(!publish(chunk)) {stop=true;break;}
                            chunk=std::make_shared<StreamChunk>();
                            chunk->first=filled;
                            chunk->positions.reserve(batch);
                            chunk->distances.reserve(batch);
                        }
                    }
                }
            }
            if(!stop && !chunk->positions.empty())publish(chunk);
        } catch(...) {
            std::lock_guard<std::mutex> lock(mutex);
            error=std::current_exception();
        }
        a.join();b.join();
        std::lock_guard<std::mutex> lock(mutex);
        total_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        finished=true;
        changed.notify_all();
    }
public:
    StreamingPlan(Bytes f,Floats t3,Floats t1,Bytes initial,Bytes out3,Bytes out1,int chunk_size)
        :flags(f),band(initial),outside3(out3),outside1(out1),distance3(t3),distance1(t1),
         rows(f.shape(0)),cols(f.shape(1)),batch(chunk_size) {
        if(batch<=0)throw std::invalid_argument("Chunk size must be positive");
        worker=std::thread([this]{run();});
    }
    ~StreamingPlan() {close();}
    void wait_outer() {
        py::gil_scoped_release release;
        std::unique_lock<std::mutex> lock(mutex);
        changed.wait(lock,[&]{return outer_count==2 || error;});
        if(error)std::rethrow_exception(error);
    }
    py::object next() {
        std::shared_ptr<StreamChunk> chunk;
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> lock(mutex);
            changed.wait(lock,[&]{return !queue.empty() || finished || error;});
            if(error)std::rethrow_exception(error);
            if(!queue.empty()) {chunk=queue.front();queue.pop_front();changed.notify_all();}
        }
        if(!chunk)return py::none();
        auto *owner=new std::shared_ptr<StreamChunk>(chunk);
        py::capsule lifetime(owner,[](void *p){delete static_cast<std::shared_ptr<StreamChunk>*>(p);});
        const py::ssize_t n=chunk->positions.size();
        auto indices=py::array_t<int32_t>({n},{py::ssize_t(sizeof(int32_t))},chunk->positions.data(),lifetime);
        auto times=py::array_t<float>({n},{py::ssize_t(sizeof(float))},chunk->distances.data(),lifetime);
        return py::make_tuple(chunk->first,indices,times);
    }
    py::dict stats() {
        std::lock_guard<std::mutex> lock(mutex);
        py::dict result;
        result["finished"]=finished;
        result["filled"]=filled;
        result["cpu_stream_seconds"]=total_seconds;
        result["outside_ready_seconds"]=outer_seconds;
        result["first_chunk_seconds"]=first_chunk_seconds;
        return result;
    }
    void close() {
        {std::lock_guard<std::mutex> lock(mutex);cancelled=true;changed.notify_all();}
        if(worker.joinable())worker.join();
    }
};

void launch_stream_render(uintptr_t image,uintptr_t output,uintptr_t rank,uintptr_t distance,
    uintptr_t order,uintptr_t done,uintptr_t counter,uintptr_t weights,int height,int width,
    int count,int radius,uintptr_t stream,int blocks,int first);
void update_stream_plan(uintptr_t positions,uintptr_t values,uintptr_t rank,uintptr_t distance3,
    uintptr_t distance1,int first,int count,uintptr_t stream);

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    py::class_<StreamingPlan,std::shared_ptr<StreamingPlan>>(m,"StreamingPlan",py::module_local())
        .def(py::init<py::array_t<uint8_t,py::array::c_style>,py::array_t<float,py::array::c_style>,
            py::array_t<float,py::array::c_style>,py::array_t<uint8_t,py::array::c_style>,
            py::array_t<uint8_t,py::array::c_style>,py::array_t<uint8_t,py::array::c_style>,int>())
        .def("wait_outer",&StreamingPlan::wait_outer)
        .def("next",&StreamingPlan::next)
        .def("stats",&StreamingPlan::stats)
        .def("close",&StreamingPlan::close,py::call_guard<py::gil_scoped_release>());
    m.def("render",&launch_stream_render);
    m.def("update",&update_stream_plan);
}
