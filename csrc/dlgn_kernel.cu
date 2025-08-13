#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int K = 3; // AND, OR, XOR

// Each thread computes one (b, g) pair for forward.
template <typename scalar_t>
__global__ void forward_kernel(
    const scalar_t* __restrict__ x,     // [B, M]
    const long* __restrict__ idx_l,     // [G]
    const long* __restrict__ idx_r,     // [G]
    const scalar_t* __restrict__ alpha, // [G, K]
    scalar_t* __restrict__ out,         // [B, G]
    int B, int M, int G)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= B * G) return;

    int b = t / G;
    int g = t % G;

    const long il = idx_l[g];
    const long ir = idx_r[g];

    scalar_t xl = x[b * M + il];
    scalar_t xr = x[b * M + ir];

    // Compute gate primitives
    scalar_t g_and = xl * xr;                    // product t-norm
    scalar_t g_or  = xl + xr - xl * xr;          // probabilistic sum
    scalar_t g_xor = xl + xr - (scalar_t)2.0 * xl * xr;

    // Softmax over alpha[g, :]
    scalar_t a0 = alpha[g * K + 0];
    scalar_t a1 = alpha[g * K + 1];
    scalar_t a2 = alpha[g * K + 2];
    scalar_t a_max = fmaxf(fmaxf(a0, a1), a2);
    scalar_t e0 = expf(a0 - a_max);
    scalar_t e1 = expf(a1 - a_max);
    scalar_t e2 = expf(a2 - a_max);
    scalar_t Z  = e0 + e1 + e2;

    scalar_t p0 = e0 / Z; // AND
    scalar_t p1 = e1 / Z; // OR
    scalar_t p2 = e2 / Z; // XOR

    scalar_t m = p0 * g_and + p1 * g_or + p2 * g_xor;

    out[b * G + g] = m;
}

// Backward:
// grad_out: [B, G]
// returns (grad_x: [B, M], grad_alpha: [G, K])
template <typename scalar_t>
__global__ void backward_kernel(
    const scalar_t* __restrict__ grad_out,   // [B, G]
    const scalar_t* __restrict__ x,          // [B, M]
    const long* __restrict__ idx_l,          // [G]
    const long* __restrict__ idx_r,          // [G]
    const scalar_t* __restrict__ alpha,      // [G, K]
    scalar_t* __restrict__ grad_x,           // [B, M]
    scalar_t* __restrict__ grad_alpha,       // [G, K]
    int B, int M, int G)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= B * G) return;

    int b = t / G;
    int g = t % G;

    const long il = idx_l[g];
    const long ir = idx_r[g];

    scalar_t xl = x[b * M + il];
    scalar_t xr = x[b * M + ir];
    scalar_t go = grad_out[b * G + g];

    // Gate primitives
    scalar_t g_and = xl * xr;
    scalar_t g_or  = xl + xr - xl * xr;
    scalar_t g_xor = xl + xr - (scalar_t)2.0 * xl * xr;

    // Softmax probs
    scalar_t a0 = alpha[g * K + 0];
    scalar_t a1 = alpha[g * K + 1];
    scalar_t a2 = alpha[g * K + 2];
    scalar_t a_max = fmaxf(fmaxf(a0, a1), a2);
    scalar_t e0 = expf(a0 - a_max);
    scalar_t e1 = expf(a1 - a_max);
    scalar_t e2 = expf(a2 - a_max);
    scalar_t Z  = e0 + e1 + e2;

    scalar_t p0 = e0 / Z; // AND
    scalar_t p1 = e1 / Z; // OR
    scalar_t p2 = e2 / Z; // XOR

    // Mixture output (for grad wrt alpha)
    scalar_t m = p0 * g_and + p1 * g_or + p2 * g_xor;

    // d(m)/d(xl) and d(m)/d(xr)
    scalar_t d_and_xl = xr;
    scalar_t d_or_xl  = (scalar_t)1.0 - xr;
    scalar_t d_xor_xl = (scalar_t)1.0 - (scalar_t)2.0 * xr;

    scalar_t d_and_xr = xl;
    scalar_t d_or_xr  = (scalar_t)1.0 - xl;
    scalar_t d_xor_xr = (scalar_t)1.0 - (scalar_t)2.0 * xl;

    scalar_t dm_dxl = p0 * d_and_xl + p1 * d_or_xl + p2 * d_xor_xl;
    scalar_t dm_dxr = p0 * d_and_xr + p1 * d_or_xr + p2 * d_xor_xr;

    // Accumulate into grad_x with atomics (features can fan out)
    atomicAdd(&grad_x[b * M + il], go * dm_dxl);
    atomicAdd(&grad_x[b * M + ir], go * dm_dxr);

    // Grad wrt alpha via softmax: ∂m/∂α_j = p_j * (g_j - m)
    scalar_t d_m_da0 = p0 * (g_and - m);
    scalar_t d_m_da1 = p1 * (g_or  - m);
    scalar_t d_m_da2 = p2 * (g_xor - m);

    atomicAdd(&grad_alpha[g * K + 0], go * d_m_da0);
    atomicAdd(&grad_alpha[g * K + 1], go * d_m_da1);
    atomicAdd(&grad_alpha[g * K + 2], go * d_m_da2);
}

} // namespace

torch::Tensor dlgn_forward_cuda(torch::Tensor x,
                                torch::Tensor idx_l,
                                torch::Tensor idx_r,
                                torch::Tensor alpha) {
    TORCH_CHECK(x.is_contiguous());
    TORCH_CHECK(alpha.size(1) == 3, "alpha must be [G, 3] for AND/OR/XOR");

    auto B = static_cast<int>(x.size(0));
    auto M = static_cast<int>(x.size(1));
    auto G = static_cast<int>(idx_l.size(0));

    auto y = torch::zeros({B, G}, x.options());

    const int threads = 256;
    const int blocks  = (B * G + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "dlgn_forward_cuda", ([&] {
        forward_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            idx_l.data_ptr<long>(),
            idx_r.data_ptr<long>(),
            alpha.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            B, M, G
        );
    }));

    return y;
}

std::vector<torch::Tensor> dlgn_backward_cuda(torch::Tensor grad_out,
                                              torch::Tensor x,
                                              torch::Tensor idx_l,
                                              torch::Tensor idx_r,
                                              torch::Tensor alpha) {
    auto B = static_cast<int>(x.size(0));
    auto M = static_cast<int>(x.size(1));
    auto G = static_cast<int>(idx_l.size(0));

    auto grad_x     = torch::zeros_like(x);
    auto grad_alpha = torch::zeros_like(alpha);

    const int threads = 256;
    const int blocks  = (B * G + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "dlgn_backward_cuda", ([&] {
        backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_out.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            idx_l.data_ptr<long>(),
            idx_r.data_ptr<long>(),
            alpha.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            grad_alpha.data_ptr<scalar_t>(),
            B, M, G
        );
    }));

    return {grad_x, grad_alpha};
}
