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



class CvPriorityQueueFloat {
    std::array<std::vector<CvHeapElem>,33> buckets;
    std::vector<CvHeapElem> redistribution;
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
        for (int b=0;b<33;++b) {
            for (size_t n=(b==0?head:0);n<buckets[b].size();++n)
                fallback.push(buckets[b][n]);
            buckets[b].clear();
        }
        head=0;
    }
    bool pop(CvHeapElem &result) {
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
            result=buckets[0][head++];
        }
        --remaining;
        return true;
    }
public:
    CvPriorityQueueFloat() { minima.fill(UINT32_MAX); }
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
        if (!ordinary && (std::isnan(time) || distance<last)) switch_to_ordinary();
        if (ordinary) fallback.push(value);
        else {
            int b=bucket(distance,last);
            buckets[b].push_back(value);
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
            dist = min4(FastMarching_solve(i-1,j,i,j-1,f,t),
                        FastMarching_solve(i+1,j,i,j-1,f,t),
                        FastMarching_solve(i-1,j,i,j+1,f,t),
                        FastMarching_solve(i+1,j,i,j+1,f,t));
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


py::dict plan(py::array_t<uint8_t, py::array::c_style> flags,
              py::array_t<float, py::array::c_style> distance3,
              py::array_t<float, py::array::c_style> distance1,
              py::array_t<uint8_t, py::array::c_style> band,
              py::array_t<uint8_t, py::array::c_style> outside3,
              py::array_t<uint8_t, py::array::c_style> outside1,
              py::array_t<int32_t, py::array::c_style> rank,
              py::array_t<int32_t, py::array::c_style> order) {
    const int rows=flags.shape(0), cols=flags.shape(1);
    Mat f{rows,cols,1,size_t(cols),flags.mutable_data()};
    Mat t3{rows,cols,1,size_t(cols)*sizeof(float),reinterpret_cast<uint8_t*>(distance3.mutable_data())};
    Mat t1{rows,cols,1,size_t(cols)*sizeof(float),reinterpret_cast<uint8_t*>(distance1.mutable_data())};
    Mat initial{rows,cols,1,size_t(cols),band.mutable_data()};
    Mat outer3{rows,cols,1,size_t(cols),outside3.mutable_data()};
    Mat outer1{rows,cols,1,size_t(cols),outside1.mutable_data()};
    int32_t *ranks=rank.mutable_data(), *sequence=order.mutable_data();
    int filled=0;
    double outer_seconds=0,inner_seconds=0;
    {
        py::gil_scoped_release release;
        auto start=std::chrono::steady_clock::now();
        for(int p=0;p<rows*cols;++p) ranks[p]=flags.data()[p]==INSIDE?INT32_MAX:0;
        // Every initially known direct neighbour of a masked pixel is on the
        // zero-distance band. The inner FMM therefore does not depend on the
        // negative distances farther outside, so all three traversals can run
        // independently with distinct storage.
        std::vector<float> inside_distance(distance3.data(),distance3.data()+rows*cols);
        Mat ti{rows,cols,1,size_t(cols)*sizeof(float),reinterpret_cast<uint8_t*>(inside_distance.data())};
        std::thread outside3_thread([&] {
            CvPriorityQueueFloat outside_heap3;
            outside_heap3.Add(initial);
            icvCalcFMM(outer3,t3,&outside_heap3,true);
        });
        std::thread outside1_thread([&] {
            CvPriorityQueueFloat outside_heap1;
            outside_heap1.Add(initial);
            icvCalcFMM(outer1,t1,&outside_heap1,true);
        });
        CvPriorityQueueFloat heap;
        heap.Add(initial);
        auto after_outer=std::chrono::steady_clock::now();
        int ii,jj;
        while(heap.Pop(&ii,&jj)) {
            f.at<uchar>(ii,jj)=KNOWN;
            for(int q=0;q<4;++q) {
                int i=ii,j=jj;
                if(q==0)--i; else if(q==1)--j; else if(q==2)++i; else ++j;
                if(i<=0||j<=0||i>rows-1||j>cols-1)continue;
                if(f.at<uchar>(i,j)==INSIDE) {
                    const float dist=min4(FastMarching_solve(i-1,j,i,j-1,f,ti),
                        FastMarching_solve(i+1,j,i,j-1,f,ti),
                        FastMarching_solve(i-1,j,i,j+1,f,ti),
                        FastMarching_solve(i+1,j,i,j+1,f,ti));
                    ti.at<float>(i,j)=dist;
                    const int p=i*cols+j;
                    sequence[filled++]=p;
                    ranks[p]=filled;
                    f.at<uchar>(i,j)=BAND;
                    heap.Push(i,j,dist);
                }
            }
        }
        outside3_thread.join();
        outside1_thread.join();
        for(int n=0;n<filled;++n) {
            const int p=sequence[n];
            distance3.mutable_data()[p]=inside_distance[p];
            distance1.mutable_data()[p]=inside_distance[p];
        }
        auto finish=std::chrono::steady_clock::now();
        outer_seconds=std::chrono::duration<double>(after_outer-start).count();
        inner_seconds=std::chrono::duration<double>(finish-after_outer).count();
    }
    py::dict result;
    result["filled"]=filled;
    result["outside_seconds"]=outer_seconds;
    result["inside_seconds"]=inner_seconds;
    return result;
}

void launch_render(uintptr_t image,uintptr_t output,uintptr_t rank,uintptr_t distance,
                   uintptr_t order,uintptr_t done,uintptr_t counter,uintptr_t weights,
                   int height,int width,int count,int radius,uintptr_t stream,int blocks);

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("plan",&plan);
    m.def("render",&launch_render);
}
