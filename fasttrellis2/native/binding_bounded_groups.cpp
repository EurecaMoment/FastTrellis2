extern "C" void trellis_prepare_xatlas_rng();
#include <torch/extension.h>
#include "xatlas.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>


namespace trellis_xatlas_candidate {


void check_tensor(const torch::Tensor& tensor, const std::string& name, torch::ScalarType type) {
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == type, name, " has incorrect data type");
}


bool ProgressCallbackTrampoline(trellis_xatlas_candidate::ProgressCategory category, int progress, void* userData) {
    // userData stores pointer to py::function
    auto* func = static_cast<py::function*>(userData);

    py::gil_scoped_acquire gil;

    try {
        (*func)(trellis_xatlas_candidate::StringForEnum(category), progress);
        return true; // true: continue
    } catch (py::error_already_set& e) {
        return false; // false: stop
    }
}



// Material IDs here are only chart constraints, not exported PBR materials.
// Every original index/position stays unchanged; the existing xatlas ComputeCharts
// and PackCharts still run, including its reseeding, merge and validity checks.
static std::vector<uint32_t> boundedFaceGroups(const std::vector<int32_t>& indices,
                                             const float *positions, uint32_t cap) {
    const uint32_t n=uint32_t(indices.size()/3);
    if(!cap || n<=cap)return {};
    struct Edge { uint64_t key; uint32_t edge; };
    std::vector<Edge> edges(3*n);
    std::vector<uint32_t> neighbors(3*n,UINT32_MAX), owner(n,UINT32_MAX), queue;
    queue.reserve(cap);
    for(uint32_t e=0;e<3*n;++e){
        const uint32_t a=uint32_t(indices[e]);
        const uint32_t b=uint32_t(indices[(e/3)*3+(e%3+1)%3]);
        edges[e]={uint64_t(std::min(a,b))<<32 | std::max(a,b),e};
    }
    std::sort(edges.begin(),edges.end(),[](const Edge&a,const Edge&b){
        return a.key<b.key || (a.key==b.key && a.edge<b.edge);
    });
    for(size_t first=0;first<edges.size();){
        size_t end=first+1;
        while(end<edges.size() && edges[end].key==edges[first].key)++end;
        if(end-first==2){
            neighbors[edges[first].edge]=edges[first+1].edge/3;
            neighbors[edges[first+1].edge]=edges[first].edge/3;
        }
        first=end;
    }
    // Bound connected face groups across the cleaned whole mesh so the
    // downstream chart optimizer avoids quadratic work on huge groups.
    uint32_t group=0;
    for(uint32_t seed=0;seed<n;++seed){
        if(owner[seed]!=UINT32_MAX)continue;
        queue.clear();queue.push_back(seed);owner[seed]=group;
        for(size_t cursor=0;cursor<queue.size();++cursor){
            const uint32_t face=queue[cursor];
            for(uint32_t j=0;j<3 && queue.size()<cap;++j){
                const uint32_t next=neighbors[3*face+j];
                if(next!=UINT32_MAX && owner[next]==UINT32_MAX){
                    owner[next]=group;queue.push_back(next);
                }
            }
        }
        ++group;
    }
    return owner;
}

class XAtlasWrapper {

private:
    trellis_xatlas_candidate::Atlas* m_atlas;

public:
    XAtlasWrapper() {
        m_atlas = trellis_xatlas_candidate::Create();
    }

    void Release() {
        auto *value=m_atlas;
        m_atlas=nullptr;
        py::gil_scoped_release release;
        if(value)trellis_xatlas_candidate::Destroy(value);
    }

    ~XAtlasWrapper() {
        if(m_atlas)trellis_xatlas_candidate::Destroy(m_atlas);
    }

    void AddMesh(
        const torch::Tensor& vertices,
        const torch::Tensor& faces,
        std::optional<const torch::Tensor> normals,
        std::optional<const torch::Tensor> uvs
    ) {
        check_tensor(vertices, "vertices", torch::kFloat32);
        check_tensor(faces, "faces", torch::kInt32);

        // 1. Construct mesh declaration
        trellis_xatlas_candidate::MeshDecl meshDecl;
        meshDecl.vertexCount = static_cast<uint32_t>(vertices.size(0));
        meshDecl.vertexPositionData = vertices.data_ptr<float>();
        meshDecl.vertexPositionStride = sizeof(float) * 3;
        meshDecl.indexCount = static_cast<uint32_t>(faces.size(0) * 3);
        meshDecl.indexData = faces.data_ptr<int32_t>();
        meshDecl.indexFormat = trellis_xatlas_candidate::IndexFormat::UInt32;
        if (normals.has_value()) {
            check_tensor(*normals, "normals", torch::kFloat32);
            meshDecl.vertexNormalData = normals->data_ptr<float>();
            meshDecl.vertexNormalStride = sizeof(float) * 3;
        }
        if (uvs.has_value()) {
            check_tensor(*uvs, "uvs", torch::kFloat32);
            meshDecl.vertexUvData = uvs->data_ptr<float>();
            meshDecl.vertexUvStride = sizeof(float) * 2;
        }

        // 2. Add mesh to atlas
        trellis_xatlas_candidate::AddMeshError result = trellis_xatlas_candidate::AddMesh(m_atlas, meshDecl);
        if (result != trellis_xatlas_candidate::AddMeshError::Success) {
            throw std::runtime_error("Adding mesh failed: " + std::string(trellis_xatlas_candidate::StringForEnum(result)));
        }
    }

    void AddMeshes(const torch::Tensor& vertices, const torch::Tensor& faces,
                   const torch::Tensor& vertexOffsets, const torch::Tensor& faceOffsets) {
        check_tensor(vertices, "vertices", torch::kFloat32);
        check_tensor(faces, "faces", torch::kInt32);
        check_tensor(vertexOffsets, "vertexOffsets", torch::kInt64);
        check_tensor(faceOffsets, "faceOffsets", torch::kInt64);
        TORCH_CHECK(vertexOffsets.numel() == faceOffsets.numel(), "offset counts differ");
        const auto* vo = vertexOffsets.data_ptr<int64_t>();
        const auto* fo = faceOffsets.data_ptr<int64_t>();
        const auto* fp = faces.data_ptr<int32_t>();
        const auto* vp = vertices.data_ptr<float>();
        const uint32_t count = vertexOffsets.numel() - 1;
        TORCH_CHECK(vo[0] == 0 && fo[0] == 0 && vo[count] == vertices.size(0)
                    && fo[count] == faces.size(0), "invalid cluster offsets");
        py::gil_scoped_release gil;
        std::vector<int32_t> indices;
        for (uint32_t i = 0; i < count; ++i) {
            TORCH_CHECK(vo[i+1] >= vo[i] && fo[i+1] >= fo[i], "offsets must be monotonic");
            const uint32_t vertexCount = vo[i+1] - vo[i];
            const uint32_t indexCount = (fo[i+1] - fo[i]) * 3;
            indices.resize(indexCount);
            for (uint32_t j = 0; j < indexCount; ++j) {
                const int64_t index = fp[fo[i] * 3 + j] - vo[i];
                TORCH_CHECK(index >= 0 && index < vertexCount, "cluster index out of range");
                indices[j] = static_cast<int32_t>(index);
            }
            trellis_xatlas_candidate::MeshDecl declaration;
            declaration.vertexCount = vertexCount;
            declaration.vertexPositionData = vp + vo[i] * 3;
            declaration.vertexPositionStride = sizeof(float) * 3;
            declaration.indexCount = indexCount;
            declaration.indexData = indices.data();
            declaration.indexFormat = trellis_xatlas_candidate::IndexFormat::UInt32;

            const char *capText=std::getenv("TRELLIS_XATLAS_PRE_GROUP_CAP");
            const uint32_t cap=capText?uint32_t(std::strtoul(capText,nullptr,10)):0;
            auto groups=boundedFaceGroups(indices,vp+vo[i]*3,cap);
            if(!groups.empty())declaration.faceMaterialData=groups.data();
            const auto result = trellis_xatlas_candidate::AddMesh(m_atlas, declaration, count);
            TORCH_CHECK(result == trellis_xatlas_candidate::AddMeshError::Success, "Adding cluster failed");
        }
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GetMeshes(
            const torch::Tensor& chartMap, const torch::Tensor& vertexOffsets) {
        check_tensor(chartMap, "chartMap", torch::kInt64);
        check_tensor(vertexOffsets, "vertexOffsets", torch::kInt64);
        TORCH_CHECK(vertexOffsets.numel() == m_atlas->meshCount + 1, "atlas cluster count changed");
        uint64_t vertices = 0, indices = 0;
        for (uint32_t i = 0; i < m_atlas->meshCount; ++i) {
            vertices += m_atlas->meshes[i].vertexCount;
            indices += m_atlas->meshes[i].indexCount;
        }
        auto mapping = torch::empty({static_cast<int64_t>(vertices)}, torch::kInt64);
        auto faces = torch::empty({static_cast<int64_t>(indices / 3), 3}, torch::kInt32);
        auto uv = torch::empty({static_cast<int64_t>(vertices), 2}, torch::kFloat32);
        const auto* cm = chartMap.data_ptr<int64_t>();
        const auto* vo = vertexOffsets.data_ptr<int64_t>();
        auto* mp = mapping.data_ptr<int64_t>();
        auto* fp = faces.data_ptr<int32_t>();
        auto* up = uv.data_ptr<float>();
        const float width = static_cast<float>(m_atlas->width);
        const float height = static_cast<float>(m_atlas->height);
        uint64_t vbase = 0, ibase = 0;
        py::gil_scoped_release gil;
        for (uint32_t i = 0; i < m_atlas->meshCount; ++i) {
            const auto& mesh = m_atlas->meshes[i];
            for (uint32_t j = 0; j < mesh.vertexCount; ++j) {
                const auto& vertex = mesh.vertexArray[j];
                mp[vbase + j] = cm[vo[i] + vertex.xref];
                up[(vbase + j) * 2] = width > 0 && height > 0 ? vertex.uv[0] / width : 0.0f;
                up[(vbase + j) * 2 + 1] = width > 0 && height > 0 ? vertex.uv[1] / height : 0.0f;
            }
            for (uint32_t j = 0; j < mesh.indexCount; ++j)
                fp[ibase + j] = static_cast<int32_t>(vbase + mesh.indexArray[j]);
            vbase += mesh.vertexCount;
            ibase += mesh.indexCount;
        }
        return std::make_tuple(mapping, faces, uv);
    }

    void ComputeCharts(trellis_xatlas_candidate::ChartOptions options, std::optional<py::function> progressCallback) {
        if (progressCallback.has_value()) {
            trellis_xatlas_candidate::SetProgressCallback(m_atlas, &ProgressCallbackTrampoline, &(*progressCallback));
        }

        {
            py::gil_scoped_release gil;
            trellis_xatlas_candidate::ComputeCharts(m_atlas, options);
        }

        trellis_xatlas_candidate::SetProgressCallback(m_atlas, nullptr, nullptr);
    }

    void PackCharts(trellis_xatlas_candidate::PackOptions options, std::optional<py::function> progressCallback) {
        if (progressCallback.has_value()) {
            trellis_xatlas_candidate::SetProgressCallback(m_atlas, &ProgressCallbackTrampoline, &(*progressCallback));
        }

        {
            py::gil_scoped_release gil;
            trellis_xatlas_candidate::PackCharts(m_atlas, options);
        }

        trellis_xatlas_candidate::SetProgressCallback(m_atlas, nullptr, nullptr);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GetMesh(uint32_t index) {
        if (index >= m_atlas->meshCount) {
            throw std::out_of_range("Mesh index " + std::to_string(index) + " out of bounds for atlas with " + std::to_string(m_atlas->meshCount) + " meshes.");
        }

        auto const& mesh = m_atlas->meshes[index];

        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

        auto mapping = torch::empty({(long)mesh.vertexCount}, options_int);
        auto faces = torch::empty({(long)mesh.indexCount / 3, 3}, options_int);
        auto uv = torch::empty({(long)mesh.vertexCount, 2}, options);

        int32_t* mappingPtr = mapping.data_ptr<int32_t>();
        int32_t* facesPtr = faces.data_ptr<int32_t>();
        float* uvPtr = uv.data_ptr<float>();

        float width = (float)m_atlas->width;
        float height = (float)m_atlas->height;

        for (uint32_t i = 0; i < mesh.vertexCount; ++i) {
            const auto& v = mesh.vertexArray[i];
            mappingPtr[i] = (int32_t)v.xref;

            if (width > 0 && height > 0) {
                uvPtr[i * 2 + 0] = v.uv[0] / width;
                uvPtr[i * 2 + 1] = v.uv[1] / height;
            } else {
                uvPtr[i * 2 + 0] = 0.0f;
                uvPtr[i * 2 + 1] = 0.0f;
            }
        }

        for (uint32_t i = 0; i < mesh.indexCount; ++i) {
            facesPtr[i] = (int32_t)mesh.indexArray[i];
        }

        return std::make_tuple(mapping, faces, uv);
    }
};


} // namespace trellis_xatlas_candidate


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prepare_rng", &trellis_prepare_xatlas_rng);
    m.doc() = "xatlas wrapper for PyTorch";

    py::class_<trellis_xatlas_candidate::ChartOptions>(m, "ChartOptions", py::module_local())
        .def(py::init<>())
        .def_readwrite("max_chart_area", &trellis_xatlas_candidate::ChartOptions::maxChartArea)
        .def_readwrite("max_boundary_length", &trellis_xatlas_candidate::ChartOptions::maxBoundaryLength)
        .def_readwrite("normal_deviation_weight", &trellis_xatlas_candidate::ChartOptions::normalDeviationWeight)
        .def_readwrite("roundness_weight", &trellis_xatlas_candidate::ChartOptions::roundnessWeight)
        .def_readwrite("straightness_weight", &trellis_xatlas_candidate::ChartOptions::straightnessWeight)
        .def_readwrite("normal_seam_weight", &trellis_xatlas_candidate::ChartOptions::normalSeamWeight)
        .def_readwrite("texture_seam_weight", &trellis_xatlas_candidate::ChartOptions::textureSeamWeight)
        .def_readwrite("max_cost", &trellis_xatlas_candidate::ChartOptions::maxCost)
        .def_readwrite("max_iterations", &trellis_xatlas_candidate::ChartOptions::maxIterations)
        .def_readwrite("use_input_mesh_uvs", &trellis_xatlas_candidate::ChartOptions::useInputMeshUvs)
        .def_readwrite("fix_winding", &trellis_xatlas_candidate::ChartOptions::fixWinding);

    py::class_<trellis_xatlas_candidate::PackOptions>(m, "PackOptions", py::module_local())
        .def(py::init<>())
        .def_readwrite("max_chart_size", &trellis_xatlas_candidate::PackOptions::maxChartSize)
        .def_readwrite("padding", &trellis_xatlas_candidate::PackOptions::padding)
        .def_readwrite("texels_per_unit", &trellis_xatlas_candidate::PackOptions::texelsPerUnit)
        .def_readwrite("resolution", &trellis_xatlas_candidate::PackOptions::resolution)
        .def_readwrite("bilinear", &trellis_xatlas_candidate::PackOptions::bilinear)
        .def_readwrite("block_align", &trellis_xatlas_candidate::PackOptions::blockAlign)
        .def_readwrite("brute_force", &trellis_xatlas_candidate::PackOptions::bruteForce)
        .def_readwrite("rotate_charts", &trellis_xatlas_candidate::PackOptions::rotateCharts)
        .def_readwrite("rotate_charts_to_axis", &trellis_xatlas_candidate::PackOptions::rotateChartsToAxis);

    py::class_<trellis_xatlas_candidate::XAtlasWrapper>(m, "Atlas", py::module_local())
        .def(py::init<>())
        .def("release", &trellis_xatlas_candidate::XAtlasWrapper::Release)
        .def("add_mesh", &trellis_xatlas_candidate::XAtlasWrapper::AddMesh)
        .def("add_meshes", &trellis_xatlas_candidate::XAtlasWrapper::AddMeshes)
        .def("get_meshes", &trellis_xatlas_candidate::XAtlasWrapper::GetMeshes)
        .def("compute_charts", &trellis_xatlas_candidate::XAtlasWrapper::ComputeCharts)
        .def("pack_charts", &trellis_xatlas_candidate::XAtlasWrapper::PackCharts)
        .def("get_mesh", &trellis_xatlas_candidate::XAtlasWrapper::GetMesh);
}
