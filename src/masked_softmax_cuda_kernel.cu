#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cfloat>
#include <vector>

using namespace at;

namespace {
template <typename scalar_t>
__global__ void __launch_bounds__(32) masked_softmax_cuda_forward_kernel(
    const scalar_t* __restrict__ input,
    const int* __restrict__ mask,
    scalar_t* __restrict__ output,
    unsigned int hidden_size,
    unsigned int m0,
    unsigned int m1,
    scalar_t scale) {

  const int tid = threadIdx.x;
  const unsigned int ibase = blockIdx.x * gridDim.y * gridDim.z * hidden_size +
                             blockIdx.y * gridDim.z * hidden_size +
                             blockIdx.z * hidden_size;

  const unsigned int mask_offset = blockIdx.x * (m0 > 1 ? m1 : 0) +
                                   blockIdx.z * (m1 > 1 ? 1 : 0);
  unsigned int mask_size = min(static_cast<unsigned int>(mask[mask_offset]),
                               hidden_size);
  unsigned shfl_mask = __ballot_sync(0xffffffff, threadIdx.x < mask_size);

  scalar_t max_x = -FLT_MAX;
  for (unsigned int i = tid; i < mask_size; i+=blockDim.x) {
    max_x = fmaxf(max_x, input[ibase + i] * scale);
  }
  for (unsigned int i = 16; i > 0; i >>= 1) {
    max_x = max(max_x, __shfl_xor_sync(shfl_mask, max_x, i));
  }

  scalar_t exp_sum = 0;
  for (unsigned int i = tid; i < mask_size; i+=blockDim.x) {
    exp_sum += std::exp(input[ibase + i] * scale - max_x);
  }
  for (unsigned int i = 16; i > 0; i >>= 1) {
    exp_sum += __shfl_xor_sync(shfl_mask, exp_sum, i);
  }

  for (unsigned int i = tid; i < mask_size; i+=blockDim.x) {
    output[ibase + i] = std::exp(input[ibase + i] * scale - max_x) / exp_sum;
  }
}

// d_input = output * (grad_output - output * sum(grad_output)) * scale
template <typename scalar_t>
__global__ void __launch_bounds__(32) masked_softmax_cuda_backward_kernel(
    scalar_t* __restrict__ d_input,
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ output,
    const int* __restrict__ mask,
    unsigned int hidden_size,
    unsigned int m0,
    unsigned int m1,
    scalar_t scale) {
  const int tid = threadIdx.x;
  const unsigned int ibase = blockIdx.x * gridDim.y * gridDim.z * hidden_size +
                             blockIdx.y * gridDim.z * hidden_size +
                             blockIdx.z * hidden_size;

  const unsigned int mask_offset = blockIdx.x * (m0 > 1 ? m1 : 0) +
                                   blockIdx.z * (m1 > 1 ? 1 : 0);
  unsigned int mask_size = min(static_cast<unsigned int>(mask[mask_offset]),
                               hidden_size);
  unsigned shfl_mask = __ballot_sync(0xffffffff, threadIdx.x < mask_size);

  scalar_t grad_sum = 0;
  for (unsigned int i = tid; i < mask_size; i+=blockDim.x) {
    scalar_t o = output[ibase + i];
    grad_sum += grad_output[ibase + i] * o;
  }
  for (unsigned int i = 16; i > 0; i >>= 1) {
    grad_sum += __shfl_xor_sync(shfl_mask, grad_sum, i);
  }

  for (unsigned int i = tid; i < mask_size; i+=blockDim.x) {
    scalar_t o = output[ibase + i];
    d_input[ibase + i] = o * (grad_output[ibase + i] - grad_sum) * scale;
  }
}
} // namespace

std::vector<at::Tensor> masked_softmax_cuda_forward(
    at::Tensor input,
    at::Tensor mask,
    at::Tensor scale) {
  AT_CHECK(input.dim() == 4, "input has an incorrect shape");
  AT_CHECK(mask.dim() == 2, "mask has an incorrect shape");
  AT_CHECK(mask.size(0) == 1 || mask.size(0) == input.size(0),
           "mask dim #0 has an incorrect shape");
  AT_CHECK(mask.size(1) == 1 || mask.size(1) == input.size(2),
           "mask dim #2 has an incorrect shape");

  auto output = at::zeros_like(input);

  const int threads = 32;
  const dim3 blocks(input.size(0), input.size(1), input.size(2));

  AT_DISPATCH_FLOATING_TYPES(input.type(), "masked_softmax_forward_cuda", ([&] {
    masked_softmax_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        input.data<scalar_t>(),
        mask.data<int>(),
        output.data<scalar_t>(),
        input.size(3),
        mask.size(0),
        mask.size(1),
        scale.item<scalar_t>());
  }));

  return {output};
}

std::vector<at::Tensor> masked_softmax_cuda_backward(
    at::Tensor grad_output,
    at::Tensor output,
    at::Tensor mask,
    at::Tensor scale) {
  auto d_input = at::zeros_like(output);

  const int threads = 32;
  const dim3 blocks(output.size(0), output.size(1), output.size(2));

  AT_DISPATCH_FLOATING_TYPES(output.type(), "masked_softmax_forward_cuda", ([&] {
    masked_softmax_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        d_input.data<scalar_t>(),
        grad_output.data<scalar_t>(),
        output.data<scalar_t>(),
        mask.data<int>(),
        output.size(3),
        mask.size(0),
        mask.size(1),
        scale.item<scalar_t>());
  }));

  return {d_input};
}
