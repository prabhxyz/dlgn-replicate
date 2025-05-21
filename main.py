import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- base gate ----------
class DiffGate(nn.Module):
    """one gate"""
    def __init__(self, bias):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float))
        self.scale = nn.Parameter(torch.ones(1))  # learn temp

    def forward(self, *xs):
        # linear comb → squash
        s = torch.stack(xs, dim=-1).sum(-1)
        return torch.sigmoid(self.scale * (s + self.bias))


# ---------- hard-coded primitives ----------
class AND(DiffGate):
    def __init__(self):
        # want output ≈1 only when both ~1
        super().__init__(bias=-1.5)   # rough

class OR(DiffGate):
    def __init__(self):
        super().__init__(bias=-0.5)

class NAND(DiffGate):
    def __init__(self):
        super().__init__(bias=1.5)

class NOT(nn.Module):
    """flip"""
    def forward(self, x):
        return 1 - x


# ---------- small layer wrapper ----------
class LogicLayer(nn.Module):
    """stack gates in parallel"""
    def __init__(self, n_in, n_out, gate_type=AND):
        super().__init__()
        self.gates = nn.ModuleList([gate_type() for _ in range(n_out)])
        self.w = nn.Parameter(torch.randn(n_out, n_in))  # soft wiring weights

    def forward(self, x):
        outs = []
        for g, w_row in zip(self.gates, self.w):
            # weighted fan-in
            y = g(*(x * w_row.sigmoid()))
            outs.append(y)
        return torch.stack(outs, dim=-1)


# ---------- network ----------
class LogicNet(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.l1 = LogicLayer(in_dim, hidden, AND)
        self.l2 = LogicLayer(hidden, out_dim, OR)

    def forward(self, x):
        h = self.l1(x)
        y = self.l2(h)
        return y


# ---------- quick XOR demo ----------
if __name__ == "__main__":
    # data (truth table)
    X = torch.tensor([[0.,0.],
                      [0.,1.],
                      [1.,0.],
                      [1.,1.]])
    y = torch.tensor([[0.],
                      [1.],
                      [1.],
                      [0.]])

    net = LogicNet(in_dim=2, hidden=4, out_dim=1)
    opt = torch.optim.Adam(net.parameters(), lr=0.05)

    for step in range(5000):
        pred = net(X)
        loss = F.mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(step, loss.item())

    print("predictions:", net(X).detach().round())