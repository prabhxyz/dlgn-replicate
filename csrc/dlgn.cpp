#include <torch/extension.h>
#include <vector>

torch::Tensor dlgn_forward_cuda(torch::Tensor x,
                                torch::Tensor idx_l,
                                torch::Tensor idx_r,
                                torch::Tensor alpha);

std::vector<torch::Tensor> dlgn_backward_cuda(torch::Tensor grad_out,
                                              torch::Tensor x,
                                              torch::Tensor idx_l,
                                              torch::Tensor idx_r,
                                              torch::Tensor alpha);

torch::Tensor dlgn_forward(torch::Tensor x,
                           torch::Tensor idx_l,
                           torch::Tensor idx_r,
                           torch::Tensor alpha) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(idx_l.is_cuda() && idx_r.is_cuda(), "idx tensors must be CUDA");
  TORCH_CHECK(alpha.is_cuda(), "alpha must be CUDA");
  return dlgn_forward_cuda(x, idx_l, idx_r, alpha);
}

std::vector<torch::Tensor> dlgn_backward(torch::Tensor grad_out,
                                         torch::Tensor x,
                                         torch::Tensor idx_l,
                                         torch::Tensor idx_r,
                                         torch::Tensor alpha) {
  TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
  TORCH_CHECK(x.is_cuda() && idx_l.is_cuda() && idx_r.is_cuda() && alpha.is_cuda(),
              "all inputs must be CUDA");
  return dlgn_backward_cuda(grad_out, x, idx_l, idx_r, alpha);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &dlgn_forward, "DLGN forward (CUDA)");
  m.def("backward", &dlgn_backward, "DLGN backward (CUDA)");
}