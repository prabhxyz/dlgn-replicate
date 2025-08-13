import torch
from .ops import dlgn_forward, dlgn_backward

class DLGNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx_l, idx_r, alpha):
        # Shapes:
        #   x:     [B, M]  (features in [0,1] ideally)
        #   idx_*: [G]     (int64 indices into feature dim)
        #   alpha: [G, K]  (logits over gate types; here K=3)
        x = x.contiguous()
        idx_l = idx_l.contiguous()
        idx_r = idx_r.contiguous()
        alpha = alpha.contiguous()

        y = dlgn_forward(x, idx_l, idx_r, alpha)
        ctx.save_for_backward(x, idx_l, idx_r, alpha)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, idx_l, idx_r, alpha = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_x, grad_alpha = dlgn_backward(grad_out, x, idx_l, idx_r, alpha)
        # idx tensors are not learnable
        return grad_x, None, None, grad_alpha
