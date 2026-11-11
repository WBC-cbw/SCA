import sys
sys.path.insert(0, ".")
import torch, h5py, numpy as np
from cdpa_common import CDPANet, load_b_e1_init, B_E1_MODELS
m = CDPANet().cuda()
load_b_e1_init(m, B_E1_MODELS / "cnn_s42" / "ckpt.pt")
with h5py.File("D:/sca_project/ASCAD_data/ASCAD.h5", "r") as f:
    x = f["Profiling_traces/traces"][:2000].astype(np.float32)[:, None, :]
    y = f["Profiling_traces/labels"][:2000].astype(np.int64)
xb = torch.from_numpy(x).cuda(); yb = torch.from_numpy(y).cuda()
m.eval()
with torch.no_grad():
    logits = m(xb)
    print("CE", float(torch.nn.functional.cross_entropy(logits, yb)))
    print("acc", float((logits.argmax(1) == yb).float().mean()))
    print("logit mean/std", float(logits.mean()), float(logits.std()))
    reps = m.representations(xb)
    for k, v in reps.items():
        print(k, tuple(v.shape), float(v.norm(dim=1).mean()))
