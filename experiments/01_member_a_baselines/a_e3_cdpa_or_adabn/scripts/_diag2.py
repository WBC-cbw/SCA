import sys
sys.path.insert(0, ".")
import torch, torch.nn as nn, numpy as np, h5py
from cdpa_common import CDPANet, load_b_e1_init, B_E1_MODELS, load_domain_data
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
data = load_domain_data("D:/sca_project/ASCAD_data/ASCAD.h5", "D:/sca_project/ASCAD_data/ASCAD_desync50.h5",
                        n_source_train=45000, n_target_adapt=45000)
xs = data["source_train_x"][:2000]; ys = data["source_train_y"][:2000]
m = CDPANet().cuda(); load_b_e1_init(m, B_E1_MODELS / "cnn_s42" / "ckpt.pt")
opt = torch.optim.RMSprop(m.parameters(), lr=1e-5)
ce = nn.CrossEntropyLoss()
m.train()
for step in range(10):
    xb = torch.from_numpy(xs[step*200:(step+1)*200]).cuda()
    yb = torch.from_numpy(ys[step*200:(step+1)*200]).cuda()
    logits = m(xb)
    loss = ce(logits, yb)
    print(step, "CE", round(float(loss), 4), "acc", round(float((logits.argmax(1)==yb).float().mean()), 4),
          "logit_max", round(float(logits.max()),2), "logit_std", round(float(logits.std()),2))
    opt.zero_grad(); loss.backward()
    gn = float(torch.sqrt(sum((p.grad.float()**2).sum() for p in m.parameters() if p.grad is not None)))
    print("   grad_norm", round(gn,1))
    opt.step()
