#include <torch/extension.h>

#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK(x.scalar_type() == torch::kFloat, #x " must be float32")
#define CHECK_LONG(x) TORCH_CHECK(x.scalar_type() == torch::kLong, #x " must be int64")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> furthest_point_sample_packed_cuda(
    torch::Tensor xyz,
    torch::Tensor offset,
    int64_t npoint);
torch::Tensor ball_query_packed_cuda(
    torch::Tensor xyz,
    torch::Tensor offset,
    torch::Tensor new_xyz,
    torch::Tensor new_offset,
    torch::Tensor radii,
    int64_t nsample);
torch::Tensor group_points_forward_cuda(torch::Tensor points, torch::Tensor idx);
torch::Tensor group_points_backward_cuda(torch::Tensor grad_out, torch::Tensor idx, int64_t n_points);
torch::Tensor gather_points_forward_cuda(torch::Tensor points, torch::Tensor idx);
torch::Tensor gather_points_backward_cuda(torch::Tensor grad_out, torch::Tensor idx, int64_t n_points);

std::vector<torch::Tensor> furthest_point_sample_packed(
    torch::Tensor xyz,
    torch::Tensor offset,
    int64_t npoint) {
  CHECK_INPUT(xyz);
  CHECK_INPUT(offset);
  CHECK_FLOAT32(xyz);
  CHECK_LONG(offset);
  TORCH_CHECK(xyz.dim() == 2 && xyz.size(1) == 3, "xyz must have shape (N, 3)");
  TORCH_CHECK(offset.dim() == 1, "offset must have shape (B,)");
  TORCH_CHECK(npoint > 0, "npoint must be positive");
  return furthest_point_sample_packed_cuda(xyz, offset, npoint);
}

torch::Tensor ball_query_packed(
    torch::Tensor xyz,
    torch::Tensor offset,
    torch::Tensor new_xyz,
    torch::Tensor new_offset,
    torch::Tensor radii,
    int64_t nsample) {
  CHECK_INPUT(xyz);
  CHECK_INPUT(offset);
  CHECK_INPUT(new_xyz);
  CHECK_INPUT(new_offset);
  CHECK_INPUT(radii);
  CHECK_FLOAT32(xyz);
  CHECK_FLOAT32(new_xyz);
  CHECK_FLOAT32(radii);
  CHECK_LONG(offset);
  CHECK_LONG(new_offset);
  TORCH_CHECK(xyz.dim() == 2 && xyz.size(1) == 3, "xyz must have shape (N, 3)");
  TORCH_CHECK(new_xyz.dim() == 2 && new_xyz.size(1) == 3, "new_xyz must have shape (M, 3)");
  TORCH_CHECK(offset.dim() == 1, "offset must have shape (B,)");
  TORCH_CHECK(new_offset.dim() == 1, "new_offset must have shape (B,)");
  TORCH_CHECK(radii.dim() == 1, "radii must have shape (M,)");
  TORCH_CHECK(nsample > 0, "nsample must be positive");
  return ball_query_packed_cuda(xyz, offset, new_xyz, new_offset, radii, nsample);
}

torch::Tensor group_points_forward(torch::Tensor points, torch::Tensor idx) {
  CHECK_INPUT(points);
  CHECK_INPUT(idx);
  CHECK_FLOAT32(points);
  CHECK_LONG(idx);
  TORCH_CHECK(points.dim() == 2, "points must have shape (N, C)");
  TORCH_CHECK(idx.dim() == 2, "idx must have shape (M, nsample)");
  return group_points_forward_cuda(points, idx);
}

torch::Tensor group_points_backward(torch::Tensor grad_out, torch::Tensor idx, int64_t n_points) {
  CHECK_INPUT(grad_out);
  CHECK_INPUT(idx);
  CHECK_FLOAT32(grad_out);
  CHECK_LONG(idx);
  TORCH_CHECK(grad_out.dim() == 3, "grad_out must have shape (M, nsample, C)");
  TORCH_CHECK(idx.dim() == 2, "idx must have shape (M, nsample)");
  TORCH_CHECK(n_points >= 0, "n_points must be non-negative");
  return group_points_backward_cuda(grad_out, idx, n_points);
}

torch::Tensor gather_points_forward(torch::Tensor points, torch::Tensor idx) {
  CHECK_INPUT(points);
  CHECK_INPUT(idx);
  CHECK_FLOAT32(points);
  CHECK_LONG(idx);
  TORCH_CHECK(points.dim() == 2, "points must have shape (N, C)");
  TORCH_CHECK(idx.dim() == 1, "idx must have shape (M,)");
  return gather_points_forward_cuda(points, idx);
}

torch::Tensor gather_points_backward(torch::Tensor grad_out, torch::Tensor idx, int64_t n_points) {
  CHECK_INPUT(grad_out);
  CHECK_INPUT(idx);
  CHECK_FLOAT32(grad_out);
  CHECK_LONG(idx);
  TORCH_CHECK(grad_out.dim() == 2, "grad_out must have shape (M, C)");
  TORCH_CHECK(idx.dim() == 1, "idx must have shape (M,)");
  TORCH_CHECK(n_points >= 0, "n_points must be non-negative");
  return gather_points_backward_cuda(grad_out, idx, n_points);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("furthest_point_sample_packed", &furthest_point_sample_packed, "Packed furthest point sampling (CUDA)");
  m.def("ball_query_packed", &ball_query_packed, "Packed ball query (CUDA)");
  m.def("group_points_forward", &group_points_forward, "Group points forward (CUDA)");
  m.def("group_points_backward", &group_points_backward, "Group points backward (CUDA)");
  m.def("gather_points_forward", &gather_points_forward, "Gather points forward (CUDA)");
  m.def("gather_points_backward", &gather_points_backward, "Gather points backward (CUDA)");
}
