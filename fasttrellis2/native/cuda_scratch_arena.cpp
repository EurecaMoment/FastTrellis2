// Bounded request-independent device storage for the existing CuMesh kernels.
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <unordered_map>

namespace {
using Allocate = cudaError_t (*)(void **, size_t);
using Free = cudaError_t (*)(void *);
Allocate original_allocate = nullptr;
Free original_free = nullptr;
std::once_flag initialized;
std::mutex gate;
void *arena = nullptr;
size_t capacity = 0;
int arena_device = -1;
std::map<size_t, size_t> available;
std::unordered_map<void *, size_t> live;
thread_local bool enabled = false, caching = false, preserve_sync = false;
thread_local uint64_t allocations = 0, releases = 0, hits = 0;
thread_local double allocate_seconds = 0, free_seconds = 0;

void initialize() {
    std::call_once(initialized, [] {
        original_allocate = reinterpret_cast<Allocate>(dlsym(RTLD_NEXT, "cudaMalloc"));
        original_free = reinterpret_cast<Free>(dlsym(RTLD_NEXT, "cudaFree"));
        if (!original_allocate || !original_free) {
            void *handle = dlopen("libcudart.so.12", RTLD_NOW);
            original_allocate = reinterpret_cast<Allocate>(dlsym(handle, "cudaMalloc"));
            original_free = reinterpret_cast<Free>(dlsym(handle, "cudaFree"));
        }
    });
}

bool in_arena(const void *pointer) {
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    const auto begin = reinterpret_cast<uintptr_t>(arena);
    return arena && address >= begin && address - begin < capacity;
}
}

extern "C" int trellis_pool_prepare_arena(size_t bytes) {
    initialize();
    if (!original_allocate || !bytes) return int(cudaErrorInvalidValue);
    std::lock_guard<std::mutex> lock(gate);
    if (arena) return capacity == bytes ? int(cudaSuccess) : int(cudaErrorInvalidValue);
    cudaError_t status = cudaGetDevice(&arena_device);
    if (status != cudaSuccess) return int(status);
    status = original_allocate(&arena, bytes);
    if (status != cudaSuccess) return int(status);
    capacity = bytes;
    available.emplace(0, bytes);
    return int(cudaSuccess);
}

extern "C" cudaError_t cudaMalloc(void **pointer, size_t bytes) {
    initialize();
    if (!original_allocate) return cudaErrorUnknown;
    if (!enabled || !caching || !arena || !bytes) return original_allocate(pointer, bytes);
    const auto started = std::chrono::steady_clock::now();
    ++allocations;
    size_t rounded = 256;
    while (rounded < bytes && rounded <= (SIZE_MAX >> 1)) rounded <<= 1;
    if (rounded < bytes) rounded = bytes;
    bool allocated = false;
    int device = -1;
    const auto device_status = cudaGetDevice(&device);
    if (device_status != cudaSuccess) return device_status;
    if (device == arena_device) {
        std::lock_guard<std::mutex> lock(gate);
        auto best = available.end();
        for (auto it = available.begin(); it != available.end(); ++it) {
            if (it->second >= rounded && (best == available.end() || it->second < best->second)) {
                best = it;
                if (it->second == rounded) break;
            }
        }
        if (best != available.end()) {
            const size_t offset = best->first, size = best->second;
            available.erase(best);
            if (size > rounded) available.emplace(offset + rounded, size - rounded);
            *pointer = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(arena) + offset);
            live.emplace(*pointer, rounded);
            ++hits;
            allocated = true;
        }
    }
    const auto result = allocated ? cudaSuccess : original_allocate(pointer, bytes);
    allocate_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return result;
}

extern "C" cudaError_t cudaFree(void *pointer) {
    initialize();
    if (!original_free) return cudaErrorUnknown;
    const auto started = std::chrono::steady_clock::now();
    if (enabled) ++releases;
    cudaError_t status = cudaSuccess;
    if (!in_arena(pointer)) {
        status = original_free(pointer);
    } else {
        // Native cudaFree is a completion barrier. Geometry keeps that barrier;
        // the existing UV scope permits same-stream reuse and joins at exit.
        if (!enabled || preserve_sync) {
            status = cudaDeviceSynchronize();
            if (status != cudaSuccess) return status;
        }
        std::lock_guard<std::mutex> lock(gate);
        const auto entry = live.find(pointer);
        if (entry == live.end()) return cudaErrorInvalidValue;
        size_t offset = reinterpret_cast<uintptr_t>(pointer) - reinterpret_cast<uintptr_t>(arena);
        size_t size = entry->second;
        live.erase(entry);
        auto after = available.lower_bound(offset);
        if (after != available.end() && offset + size == after->first) {
            size += after->second;
            after = available.erase(after);
        }
        if (after != available.begin()) {
            auto before = std::prev(after);
            if (before->first + before->second == offset) {
                offset = before->first;
                size += before->second;
                available.erase(before);
            }
        }
        available.emplace(offset, size);
    }
    if (enabled) free_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return status;
}

extern "C" void trellis_pool_begin(int mode) {
    initialize();
    allocations = releases = hits = 0;
    allocate_seconds = free_seconds = 0;
    caching = mode != 0;
    preserve_sync = mode == 2;
    enabled = true;
}

extern "C" void trellis_pool_end() { enabled = false; }
extern "C" int trellis_pool_sync_supported() { return 1; }

extern "C" void trellis_pool_stats(double *result) {
    result[0] = allocations;
    result[1] = releases;
    result[2] = hits;
    result[3] = allocate_seconds;
    result[4] = free_seconds;
    std::lock_guard<std::mutex> lock(gate);
    size_t spare = 0;
    for (const auto &entry : available) spare += entry.second;
    result[5] = double(spare);
}
