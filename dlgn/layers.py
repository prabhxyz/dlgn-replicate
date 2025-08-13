import torch
import torch.nn as nn
from .autograd import DLGNFunction

# Small helper: K=3 hardcoded (AND, OR, XOR).
K = 3

class GateLayer(nn.Module):
    """
    A single layer of G two-input gates. Each gate g mixes AND/OR/XOR with a softmax over learnable logits alpha[g].
    Inputs are gathered from feature indices idx_l[g], idx_r[g] on the incoming tensor x[B, M].
    Output is y[B, G].
    """
    def __init__(self, num_inputs, idx_left, idx_right, init_alpha_scale=0.01, device=None):
        super().__init__()
        assert len(idx_left) == len(idx_right)
        self.num_inputs = num_inputs
        self.G = len(idx_left)
        self.register_buffer("idx_l", torch.tensor(idx_left, dtype=torch.long, device=device))
        self.register_buffer("idx_r", torch.tensor(idx_right, dtype=torch.long, device=device))

        alpha = torch.zeros(self.G, K, device=device)
        alpha.add_(init_alpha_scale * torch.randn_like(alpha))  # small random init near uniform
        self.alpha = nn.Parameter(alpha)

    def forward(self, x):
        # x: [B, M]
        return DLGNFunction.apply(x, self.idx_l, self.idx_r, self.alpha)


class DLGN(nn.Module):
    """
    Tiny example network: (GateLayer -> linear head).
    Stack more GateLayers for deeper networks, e.g., y1 -> GateLayer2 uses indices over G1 outputs, etc.
    """
    def __init__(self, num_inputs, idx_left, idx_right, hidden_out=None, device=None):
        super().__init__()
        self.gates = GateLayer(num_inputs, idx_left, idx_right, device=device)
        out_dim = len(idx_left)
        if hidden_out is None:
            self.head = nn.Linear(out_dim, 1)
        else:
            self.head = nn.Sequential(
                nn.Linear(out_dim, hidden_out),
                nn.ReLU(),
                nn.Linear(hidden_out, 1),
            )

    def forward(self, x):
        g = self.gates(x)
        return self.head(g)
