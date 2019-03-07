#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<at::Tensor> masked_softmax_cuda_forward(
    at::Tensor input,
    at::Tensor mask,
    at::Tensor scale);

std::vector<at::Tensor> masked_softmax_cuda_backward(
    at::Tensor grad,
    at::Tensor output,
    at::Tensor mask,
    at::Tensor scale);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> masked_softmax_forward(
    at::Tensor input,
    at::Tensor mask,
    at::Tensor scale) {
  CHECK_INPUT(input);
  CHECK_INPUT(mask);

  return masked_softmax_cuda_forward(input, mask, scale);
}

std::vector<at::Tensor> masked_softmax_backward(
    at::Tensor grad,
    at::Tensor output,
    at::Tensor mask,
    at::Tensor scale) {
  CHECK_INPUT(grad);
  CHECK_INPUT(output);
  CHECK_INPUT(mask);

  return masked_softmax_cuda_backward(
      grad,
      output,
      mask,
      scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &masked_softmax_forward, "LLTM forward (CUDA)");
  m.def("backward", &masked_softmax_backward, "LLTM backward (CUDA)");
}
