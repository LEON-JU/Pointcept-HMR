#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cfloat>
#include <vector>

namespace {

constexpr int THREADS = 256;

inline int GET_BLOCKS(const int n) {
  return (n + THREADS - 1) / THREADS;
}

template <typename scalar_t>
__global__ void gather_points_forward_kernel(
    const scalar_t* __restrict__ points,
    const int64_t* __restrict__ idx,
    scalar_t* __restrict__ out,
    int64_t m,
    int64_t c) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = m * c;
  if (linear >= total) {
    return;
  }
  int64_t row = linear / c;
  int64_t col = linear % c;
  int64_t src = idx[row];
  out[linear] = points[src * c + col];
}

template <typename scalar_t>
__global__ void gather_points_backward_kernel(
    const scalar_t* __restrict__ grad_out,
    const int64_t* __restrict__ idx,
    scalar_t* __restrict__ grad_points,
    int64_t m,
    int64_t c) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = m * c;
  if (linear >= total) {
    return;
  }
  int64_t row = linear / c;
  int64_t col = linear % c;
  int64_t dst = idx[row];
  atomicAdd(&grad_points[dst * c + col], grad_out[linear]);
}

template <typename scalar_t>
__global__ void group_points_forward_kernel(
    const scalar_t* __restrict__ points,
    const int64_t* __restrict__ idx,
    scalar_t* __restrict__ out,
    int64_t m,
    int64_t nsample,
    int64_t c) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = m * nsample * c;
  if (linear >= total) {
    return;
  }
  int64_t tmp = linear;
  int64_t col = tmp % c;
  tmp /= c;
  int64_t sample_idx = tmp % nsample;
  int64_t row = tmp / nsample;
  int64_t src = idx[row * nsample + sample_idx];
  out[linear] = points[src * c + col];
}

template <typename scalar_t>
__global__ void group_points_backward_kernel(
    const scalar_t* __restrict__ grad_out,
    const int64_t* __restrict__ idx,
    scalar_t* __restrict__ grad_points,
    int64_t m,
    int64_t nsample,
    int64_t c) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = m * nsample * c;
  if (linear >= total) {
    return;
  }
  int64_t tmp = linear;
  int64_t col = tmp % c;
  tmp /= c;
  int64_t sample_idx = tmp % nsample;
  int64_t row = tmp / nsample;
  int64_t dst = idx[row * nsample + sample_idx];
  atomicAdd(&grad_points[dst * c + col], grad_out[linear]);
}

template <typename scalar_t>
__global__ void fps_init_temp_kernel(scalar_t* __restrict__ temp, int64_t n) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= n) {
    return;
  }
  temp[linear] = FLT_MAX;
}

template <typename scalar_t>
__global__ void fps_kernel(
    const scalar_t* __restrict__ xyz,
    const int64_t* __restrict__ offset,
    scalar_t* __restrict__ temp,
    int64_t* __restrict__ out_idx,
    int64_t batch_size,
    int64_t npoint) {
  __shared__ scalar_t shared_best_dist[THREADS];
  __shared__ int shared_best_idx[THREADS];

  int batch_idx = blockIdx.x;
  if (batch_idx >= batch_size) {
    return;
  }
  int tid = threadIdx.x;
  int64_t start = batch_idx == 0 ? 0 : offset[batch_idx - 1];
  int64_t end = offset[batch_idx];
  int64_t n = end - start;
  if (n <= 0) {
    return;
  }

  int64_t old = 0;
  if (tid == 0) {
    out_idx[batch_idx * npoint] = start;
  }
  __syncthreads();

  for (int64_t j = 1; j < npoint; ++j) {
    scalar_t best = static_cast<scalar_t>(-1.0);
    int besti = 0;
    scalar_t x1 = xyz[(start + old) * 3 + 0];
    scalar_t y1 = xyz[(start + old) * 3 + 1];
    scalar_t z1 = xyz[(start + old) * 3 + 2];
    for (int64_t k = tid; k < n; k += blockDim.x) {
      scalar_t x2 = xyz[(start + k) * 3 + 0];
      scalar_t y2 = xyz[(start + k) * 3 + 1];
      scalar_t z2 = xyz[(start + k) * 3 + 2];
      scalar_t d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
      scalar_t cur = min(d, temp[start + k]);
      temp[start + k] = cur;
      if (cur > best) {
        best = cur;
        besti = static_cast<int>(k);
      }
    }
    shared_best_dist[tid] = best;
    shared_best_idx[tid] = besti;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride && shared_best_dist[tid] < shared_best_dist[tid + stride]) {
        shared_best_dist[tid] = shared_best_dist[tid + stride];
        shared_best_idx[tid] = shared_best_idx[tid + stride];
      }
      __syncthreads();
    }

    old = shared_best_idx[0];
    if (tid == 0) {
      out_idx[batch_idx * npoint + j] = start + old;
    }
    __syncthreads();
  }
}

template <typename scalar_t>
__global__ void ball_query_kernel(
    const scalar_t* __restrict__ xyz,
    const int64_t* __restrict__ offset,
    const scalar_t* __restrict__ new_xyz,
    const int64_t* __restrict__ new_offset,
    const float* __restrict__ radii,
    int64_t* __restrict__ out_idx,
    int64_t total_queries,
    int64_t batch_size,
    int64_t nsample) {
  int64_t q = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= total_queries) {
    return;
  }

  int batch_idx = 0;
  while (batch_idx < batch_size && q >= new_offset[batch_idx]) {
    ++batch_idx;
  }
  if (batch_idx >= batch_size) {
    return;
  }

  int64_t start = batch_idx == 0 ? 0 : offset[batch_idx - 1];
  int64_t end = offset[batch_idx];
  float radius = radii[q];
  float radius2 = radius * radius;
  scalar_t cx = new_xyz[q * 3 + 0];
  scalar_t cy = new_xyz[q * 3 + 1];
  scalar_t cz = new_xyz[q * 3 + 2];

  int64_t out_base = q * nsample;
  int found = 0;
  int64_t first = start;
  for (int64_t i = start; i < end && found < nsample; ++i) {
    scalar_t dx = xyz[i * 3 + 0] - cx;
    scalar_t dy = xyz[i * 3 + 1] - cy;
    scalar_t dz = xyz[i * 3 + 2] - cz;
    float dist2 = static_cast<float>(dx * dx + dy * dy + dz * dz);
    if (dist2 <= radius2) {
      if (found == 0) {
        first = i;
      }
      out_idx[out_base + found] = i;
      ++found;
    }
  }
  if (found == 0) {
    for (int64_t j = 0; j < nsample; ++j) {
      out_idx[out_base + j] = first;
    }
  } else {
    for (int64_t j = found; j < nsample; ++j) {
      out_idx[out_base + j] = first;
    }
  }
}

}  // namespace

std::vector<torch::Tensor> furthest_point_sample_packed_cuda(
    torch::Tensor xyz,
    torch::Tensor offset,
    int64_t npoint) {
  auto batch_size = offset.size(0);
  auto idx = torch::zeros({batch_size * npoint}, offset.options().dtype(torch::kLong));
  auto new_offset = torch::arange(1, batch_size + 1, offset.options().dtype(torch::kLong)) * npoint;
  auto temp = torch::zeros({xyz.size(0)}, xyz.options());

  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(xyz.scalar_type(), "furthest_point_sample_packed_cuda", [&] {
    fps_init_temp_kernel<scalar_t><<<GET_BLOCKS(temp.numel()), THREADS, 0, stream>>>(
        temp.data_ptr<scalar_t>(),
        temp.numel());
    fps_kernel<scalar_t><<<batch_size, THREADS, 0, stream>>>(
        xyz.data_ptr<scalar_t>(),
        offset.data_ptr<int64_t>(),
        temp.data_ptr<scalar_t>(),
        idx.data_ptr<int64_t>(),
        batch_size,
        npoint);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {idx, new_offset};
}

torch::Tensor ball_query_packed_cuda(
    torch::Tensor xyz,
    torch::Tensor offset,
    torch::Tensor new_xyz,
    torch::Tensor new_offset,
    torch::Tensor radii,
    int64_t nsample) {
  auto out = torch::zeros({new_xyz.size(0), nsample}, offset.options().dtype(torch::kLong));
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(xyz.scalar_type(), "ball_query_packed_cuda", [&] {
    ball_query_kernel<scalar_t><<<GET_BLOCKS(new_xyz.size(0)), THREADS, 0, stream>>>(
        xyz.data_ptr<scalar_t>(),
        offset.data_ptr<int64_t>(),
        new_xyz.data_ptr<scalar_t>(),
        new_offset.data_ptr<int64_t>(),
        radii.data_ptr<float>(),
        out.data_ptr<int64_t>(),
        new_xyz.size(0),
        offset.size(0),
        nsample);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor group_points_forward_cuda(torch::Tensor points, torch::Tensor idx) {
  auto out = torch::zeros({idx.size(0), idx.size(1), points.size(1)}, points.options());
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "group_points_forward_cuda", [&] {
    group_points_forward_kernel<scalar_t><<<GET_BLOCKS(out.numel()), THREADS, 0, stream>>>(
        points.data_ptr<scalar_t>(),
        idx.data_ptr<int64_t>(),
        out.data_ptr<scalar_t>(),
        idx.size(0),
        idx.size(1),
        points.size(1));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor group_points_backward_cuda(torch::Tensor grad_out, torch::Tensor idx, int64_t n_points) {
  auto grad_points = torch::zeros({n_points, grad_out.size(2)}, grad_out.options());
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(grad_out.scalar_type(), "group_points_backward_cuda", [&] {
    group_points_backward_kernel<scalar_t><<<GET_BLOCKS(grad_out.numel()), THREADS, 0, stream>>>(
        grad_out.data_ptr<scalar_t>(),
        idx.data_ptr<int64_t>(),
        grad_points.data_ptr<scalar_t>(),
        idx.size(0),
        idx.size(1),
        grad_out.size(2));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_points;
}

torch::Tensor gather_points_forward_cuda(torch::Tensor points, torch::Tensor idx) {
  auto out = torch::zeros({idx.size(0), points.size(1)}, points.options());
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "gather_points_forward_cuda", [&] {
    gather_points_forward_kernel<scalar_t><<<GET_BLOCKS(out.numel()), THREADS, 0, stream>>>(
        points.data_ptr<scalar_t>(),
        idx.data_ptr<int64_t>(),
        out.data_ptr<scalar_t>(),
        idx.size(0),
        points.size(1));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor gather_points_backward_cuda(torch::Tensor grad_out, torch::Tensor idx, int64_t n_points) {
  auto grad_points = torch::zeros({n_points, grad_out.size(1)}, grad_out.options());
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(grad_out.scalar_type(), "gather_points_backward_cuda", [&] {
    gather_points_backward_kernel<scalar_t><<<GET_BLOCKS(grad_out.numel()), THREADS, 0, stream>>>(
        grad_out.data_ptr<scalar_t>(),
        idx.data_ptr<int64_t>(),
        grad_points.data_ptr<scalar_t>(),
        idx.size(0),
        grad_out.size(1));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_points;
}
