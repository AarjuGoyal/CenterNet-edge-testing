#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include "dcn_v2_cuda.hpp"

namespace py = pybind11;

PYBIND11_MODULE(dcn_v2_ext, m) {
    m.def("dcn_v2_cuda_forward", &dcn_v2_cuda_forward, "DCN forward pass (CUDA)");
    m.def("dcn_v2_cuda_backward", &dcn_v2_cuda_backward, "DCN backward pass (CUDA)");
    m.def("dcn_v2_psroi_pooling_cuda_forward", &dcn_v2_psroi_pooling_cuda_forward, "DCN pooling cuda forward pass (CUDA)");
    m.def("dcn_v2_psroi_pooling_cuda_backward", &dcn_v2_psroi_pooling_cuda_backward, "DCN pooling cuda backward pass (CUDA)");
}