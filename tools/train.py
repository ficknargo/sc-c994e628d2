"""Train the tiny tile CNN (29 classes: A-Z, BLANK, EMPTY, ROT) and export weights for js/cnn.js."""
import sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(8); torch.manual_seed(0)
C1, C2, C3, FC, NC = 16, 24, 48, 96, 29

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.c1 = nn.Conv2d(3, C1, 3, padding=1); s.b1 = nn.BatchNorm2d(C1)
        s.c2 = nn.Conv2d(C1, C2, 3, padding=1); s.b2 = nn.BatchNorm2d(C2)
        s.c3 = nn.Conv2d(C2, C3, 3, padding=1); s.b3 = nn.BatchNorm2d(C3)
        s.fc = nn.Linear(C3 * 16, FC); s.drop = nn.Dropout(0.3); s.out = nn.Linear(FC, NC)
    def forward(s, x):
        x = F.max_pool2d(F.relu(s.b1(s.c1(x))), 2)
        x = F.max_pool2d(F.relu(s.b2(s.c2(x))), 2)
        x = F.max_pool2d(F.relu(s.b3(s.c3(x))), 2)
        return s.out(s.drop(F.relu(s.fc(x.flatten(1)))))

def prep(X):  # uint8 NHWC -> float NCHW, per-crop normalisation (same as js/cnn.js)
    x = torch.from_numpy(X).float().permute(0, 3, 1, 2)
    m = x.mean((1, 2, 3), keepdim=True); sd = x.std((1, 2, 3), keepdim=True)
    return (x - m) / (sd + 8)

def augment(x):
    # random +-2px shift, contrast/brightness jitter on normalised input, slight noise
    n = x.shape[0]
    pad = F.pad(x, (2, 2, 2, 2), mode='replicate')
    dx, dy = np.random.randint(0, 5, 2)
    x = pad[:, :, dy:dy + 32, dx:dx + 32]
    x = x * torch.empty(n, 1, 1, 1).uniform_(0.7, 1.3) + torch.empty(n, 3, 1, 1).uniform_(-0.2, 0.2)
    return x + torch.randn_like(x) * 0.05

def evaluate(net, X, Y, bs=4096):
    net.eval(); P = []
    with torch.no_grad():
        for i in range(0, len(X), bs): P.append(net(prep(X[i:i + bs])).argmax(1))
    P = torch.cat(P).numpy(); return P

def export(net, path):
    def fold(conv, bn):
        g = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        return (conv.weight * g[:, None, None, None]).detach(), ((conv.bias - bn.running_mean) * g + bn.bias).detach()
    parts = [*fold(net.c1, net.b1), *fold(net.c2, net.b2), *fold(net.c3, net.b3), net.fc.weight, net.fc.bias, net.out.weight, net.out.bias]
    arr = np.concatenate([p.detach().numpy().astype(np.float32).ravel() for p in parts])
    arr.astype(np.float16).tofile(path); print('exported', path, arr.size, 'params')

if __name__ == '__main__':
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    d = np.load('tools/data/train.npz'); X, Y = d['X'], torch.from_numpy(d['Y'].astype(np.int64))
    v = np.load('tools/data/val.npz'); Xv, Yv = v['X'], v['Y']
    net = Net(); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    bs = 512; steps = epochs * (len(X) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=steps)
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(len(X)); t0 = time.time(); tl = 0
        for i in range(len(X) // bs):
            idx = perm[i * bs:(i + 1) * bs]
            x = augment(prep(X[idx])); y = Y[idx]
            loss = F.cross_entropy(net(x), y, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item()
        P = evaluate(net, Xv, Yv)
        print(f'epoch {ep} loss {tl / (len(X) // bs):.3f} val acc {(P == Yv).mean():.4f} ({time.time() - t0:.0f}s)', flush=True)
        torch.save(net.state_dict(), 'tools/data/net.pt')
    export(net, 'model/tiles-v1.f16')
