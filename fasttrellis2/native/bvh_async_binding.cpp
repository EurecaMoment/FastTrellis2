// Call the installed BVH builder unchanged, releasing Python's interpreter lock.
#include <torch/extension.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

using Verts = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
using Trigs = Eigen::Matrix<uint32_t, Eigen::Dynamic, 3, Eigen::RowMajor>;

namespace cumesh { namespace cubvh {
class cuBVH {
public:
    cuBVH() {}
    virtual ~cuBVH() {}
    virtual void ray_trace(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor) = 0;
    virtual void unsigned_distance(at::Tensor, at::Tensor, at::Tensor, at::optional<at::Tensor>) = 0;
    virtual void signed_distance(at::Tensor, at::Tensor, at::Tensor, at::optional<at::Tensor>, uint32_t) = 0;
};
cuBVH* create_cuBVH(Eigen::Ref<const Verts>, Eigen::Ref<const Trigs>);
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("create", &cumesh::cubvh::create_cuBVH,
          pybind11::return_value_policy::take_ownership,
          pybind11::call_guard<pybind11::gil_scoped_release>());
}
