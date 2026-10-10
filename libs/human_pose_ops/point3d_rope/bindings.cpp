#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> utonia_pointrope_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor positions,
    double base,
    double direction);

std::vector<torch::Tensor> utonia_pointrope(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor positions,
    double base,
    double direction) {
  TORCH_CHECK(q.dim() == 3, "q must have shape (N,H,D)");
  TORCH_CHECK(k.dim() == 3, "k must have shape (N,H,D)");
  TORCH_CHECK(positions.dim() == 2, "positions must have shape (N,3)");
  TORCH_CHECK(q.sizes() == k.sizes(), "q and k must have identical shapes");
  TORCH_CHECK(q.size(0) == positions.size(0), "q and positions have different N");
  TORCH_CHECK(positions.size(1) == 3, "positions must have shape (N,3)");
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && positions.is_cuda(),
              "q, k, and positions must be CUDA tensors");
  TORCH_CHECK(q.scalar_type() == k.scalar_type(),
              "q and k must have identical dtypes");
  TORCH_CHECK(q.scalar_type() == at::kFloat ||
                  q.scalar_type() == at::kHalf ||
                  q.scalar_type() == at::kBFloat16,
              "q/k must be float32, float16, or bfloat16");
  TORCH_CHECK(positions.scalar_type() == at::kFloat,
              "positions must be float32");
  TORCH_CHECK(q.size(2) > 0 && q.size(2) % 6 == 0,
              "head dimension D must be positive and divisible by 6");
  TORCH_CHECK(q.size(2) <= 1024,
              "head dimension D must be <= 1024 for the CUDA kernel");

  return utonia_pointrope_cuda(
      q.contiguous(), k.contiguous(), positions.contiguous(), base, direction);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("apply", &utonia_pointrope, "Utonia continuous-coordinate 3D RoPE");
}
