# ConvRot 式分组正则 Hadamard: 单层 W8A8 误差验证 (合成 11000x 离群)
import torch
torch.manual_seed(0)
import torch.nn as nn

N0 = 256   # 组大小

def reg_hadamard_4():
    # 4 阶正则 Hadamard: 每行/列和绝对值 = 2 = sqrt(4)
    return torch.tensor([[1., 1., 1., -1.],
                         [1., 1., -1., 1.],
                         [1., -1., 1., 1.],
                         [-1., 1., 1., 1.]])

def kron(a, b):
    return torch.einsum("ij,kl->ikjl", a, b).reshape(a.shape[0]*b.shape[0],
                                                     a.shape[1]*b.shape[1])

def reg_hadamard_256():
    h = reg_hadamard_4()
    for _ in range(3):   # 4^4 = 256
        h = kron(h, reg_hadamard_4())
    return h / 256.0 ** 0.5   # 正交化

R = reg_hadamard_256()
# 正交性 + 正则性自检
err_orth = (R @ R.T - torch.eye(256)).abs().max().item()
rowsum = R.sum(dim=1).abs()
print(f"正交性误差: {err_orth:.2e}  行和 |max|={rowsum.max():.3f} (应=√256=16) |min|={rowsum.min():.3f}")
# 随机符号对角化 (仍是正则正交)
d1 = torch.diag(torch.randn(256).sign() * 1.0)
d2 = torch.diag(torch.randn(256).sign() * 1.0)
R = d1 @ R @ d2

K, N = 4096, 12288
W = torch.randn(N, K) * 0.02
# 合成 massive activation: 每 token 少数通道巨大 (模拟 11000x)
X = torch.randn(64, K)
outlier_ch = torch.tensor([7, 113, 2049, 3998])
X[:, outlier_ch] *= 3000.0

Y_ref = torch.nn.functional.linear(X, W)

def w8a8_rtn(x, w, rotate=False):
    """per-channel W8 + per-token A8 (模拟 linear_dynamic)"""
    if rotate:
        x = x.view(-1, K // N0, N0) @ R        # 激活分组旋转
        w = (w.view(N, K // N0, N0) @ R).reshape(N, K)  # 权重列分组旋转
    # 权重 per-channel int8
    s = w.abs().amax(dim=1) / 127.0
    wq = torch.round(w / s.unsqueeze(1)).clamp(-127, 127)
    # 激活 per-token int8 (reduce_range=False)
    xa = x.abs().amax(dim=1, keepdim=True) / 127.0
    xq = torch.round(x / xa)
    y = (xq @ wq.to(torch.float32).T) * xa * s.unsqueeze(0)
    return y



# 配方 v3: 两级旋转 (组间 H16 打散跨组离群 + 组内 RHT256 均匀化) + 纯 per-token 动态量化
def two_level_rot(x):
    """[B,K] -> 旋转后 [B,K]; 列侧同变换用于 W"""
    b = x.shape[0]
    x1 = x.view(b, NG, N0)
    x1 = torch.einsum("bgn,hg->bhn", x1, H16)      # 组间 16x16
    x2 = (x1.reshape(b * NG, N0) @ R).reshape(b, K)  # 组内 256
    return x2

NG = K // N0
h4 = reg_hadamard_4()
h16 = kron(kron(h4, h4) / 4.0, torch.ones(1)) if False else kron(h4, h4) / 4.0
H16 = torch.diag(torch.randn(16).sign()) @ h16 @ torch.diag(torch.randn(16).sign())

Xr = two_level_rot(X)
Wr = two_level_rot(W)                                # W 行侧(=原列方向)两级旋转

def w8a8(x, w):
    s = w.abs().amax(dim=1) / 127.0
    wq = torch.round(w / s.unsqueeze(1)).clamp(-127, 127)
    xa = x.abs().amax(dim=1, keepdim=True) / 127.0
    xq = torch.round(x / xa)
    return (xq @ wq.to(torch.float32).T) * xa * s.unsqueeze(0)

# 等价性: 两级旋转是正交变换, X_rot @ W_rot^T == X @ W^T (权重列侧旋转)
eq = (Xr @ Wr.T - Y_ref).abs().max().item() / Y_ref.abs().max().item()
print(f"两级旋转等价性残差: {eq:.2e}")

y = w8a8(Xr, Wr)
rel = ((y - Y_ref).norm() / Y_ref.norm()).item()
print(f"两级旋转+per-token W8A8: 输出相对误差 = {rel:.4%}")
print(f"旋转后激活: max={Xr.abs().max():.1f} median={Xr.abs().median():.2f} ratio={Xr.abs().max()/Xr.abs().median():.0f}x")
print("RHT2_TEST_DONE")
