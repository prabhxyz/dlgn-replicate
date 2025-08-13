import torch
import torch.nn as nn
from dlgn.layers import GateLayer, DLGN

def main():
    device = "cuda"
    torch.manual_seed(0)

    # Simple XOR toy: inputs in {0,1}^2, target x^y
    X = torch.tensor([[0.,0.],
                      [0.,1.],
                      [1.,0.],
                      [1.,1.]], device=device)
    y = torch.tensor([[0.],
                      [1.],
                      [1.],
                      [0.]], device=device)

    # One gate reading feature 0 and 1
    model = DLGN(num_inputs=2, idx_left=[0], idx_right=[1], device=device)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.BCEWithLogitsLoss()

    for step in range(300):
        opt.zero_grad()
        logits = model(X)  # [4,1]
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        if (step + 1) % 50 == 0:
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
            print(f"step {step+1:03d}  loss {loss.item():.4f}  preds {preds[:,0].tolist()}")

    print("alpha logits for the gate (AND, OR, XOR):")
    print(model.gates.alpha.detach().cpu().numpy())

if __name__ == "__main__":
    main()
