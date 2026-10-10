#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

namespace {

template <typename scalar_t>
__global__ void utonia_pointrope_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> q,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> k,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> q_out,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> k_out,
    const float* __restrict__ positions,
    const float base,
    const float direction) {
  const int n = blockIdx.x;
  const int d = threadIdx.x;
  const int heads = q.size(1);
  const int head_dim = q.size(2);
  const int q_dim = head_dim / 6;

  extern __shared__ float shared[];
  float* q_shared = shared;
  float* k_shared = shared + head_dim;
  float* inv_freq = shared + 2 * head_dim;

  if (d < q_dim) {
    inv_freq[d] = powf(base, -static_cast<float>(d) / static_cast<float>(q_dim));
  }
  __syncthreads();

  const int axis = d / (2 * q_dim);
  const int local = d % (2 * q_dim);
  const int frequency_index = local % q_dim;
  const int u_index = axis * 2 * q_dim + frequency_index;
  const int v_index = u_index + q_dim;
  const float angle =
      direction * positions[n * 3 + axis] * inv_freq[frequency_index];
  const float c = cosf(angle);
  const float s = sinf(angle);

  for (int h = 0; h < heads; ++h) {
    q_shared[d] = static_cast<float>(q[n][h][d]);
    k_shared[d] = static_cast<float>(k[n][h][d]);
    __syncthreads();

    if (local < q_dim) {
      q_out[n][h][d] = static_cast<scalar_t>(
          q_shared[u_index] * c - q_shared[v_index] * s);
      k_out[n][h][d] = static_cast<scalar_t>(
          k_shared[u_index] * c - k_shared[v_index] * s);
    } else {
      q_out[n][h][d] = static_cast<scalar_t>(
          q_shared[v_index] * c + q_shared[u_index] * s);
      k_out[n][h][d] = static_cast<scalar_t>(
          k_shared[v_index] * c + k_shared[u_index] * s);
    }
    __syncthreads();
  }
}

}  // namespace

std::vector<torch::Tensor> utonia_pointrope_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor positions,
    double base,
    double direction) {
  const int n = static_cast<int>(q.size(0));
  const int heads = static_cast<int>(q.size(1));
  const int head_dim = static_cast<int>(q.size(2));
  auto q_out = torch::empty_like(q);
  auto k_out = torch::empty_like(k);

  if (n == 0 || heads == 0) {
    return {q_out, k_out};
  }

  const int threads = head_dim;
  const size_t shared_bytes =
      sizeof(float) * static_cast<size_t>(2 * head_dim + head_dim / 6);
  const dim3 grid(static_cast<unsigned int>(n), 1, 1);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      q.scalar_type(),
      "utonia_pointrope_cuda",
      [&] {
        utonia_pointrope_kernel<scalar_t><<<grid, threads, shared_bytes>>>(
            q.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            k.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            q_out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            k_out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            positions.data_ptr<float>(),
            static_cast<float>(base),
            static_cast<float>(direction));
      });

  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
  return {q_out, k_out};
}
