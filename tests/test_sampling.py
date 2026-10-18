# tests/test_sampling.py
import torch
from torchinfra.sampling import sample_token


def main():
    torch.manual_seed(0)
    logits = torch.tensor([1.0, 3.0, 2.0, 0.5], device="cuda")
    assert sample_token(logits) == 1                      # greedy
    assert sample_token(logits, temperature=0.0) == 1
    assert sample_token(logits, temperature=1e-9) == 1    # 趋近 greedy
    assert sample_token(logits, temperature=1.0, top_k=1) == 1

    # greedy 断言：temperature→0 时数值上不可达，用极小温度验证 argmax 收敛
    # top_p 支撑集：softmax([1,3,2,.5]) ≈ [.085,.631,.232,.052]
    # 排序 cumsum=[.631,.863,.948,1]，规则 cumsum-prob<=top_p：
    #   top_p=0.5 → 只留 idx1
    for seed in range(20):
        g = torch.Generator(device="cuda").manual_seed(seed)
        assert sample_token(logits, temperature=1.0, top_p=0.5, generator=g) == 1
    # 平坦分布 logits=[1.0,1.1,0.9,1.05]：probs≈[.246,.272,.223,.259]
    # 排序 [idx1,idx3,idx0,idx2]，cumsum-prob=[0,.272,.531,.777]
    #   top_p=0.6 → 支撑集 {1,3,0}（idx2 被裁掉）
    flat = torch.tensor([1.0, 1.1, 0.9, 1.05], device="cuda")
    g = torch.Generator(device="cuda").manual_seed(0)
    picks = {sample_token(flat, temperature=1.0, top_p=0.6, generator=g)
             for _ in range(300)}
    assert picks <= {1, 3, 0}, picks
    assert len(picks) >= 2, f"支撑集应有多于一个 token 被采到: {picks}"  # 采样真在分布内进行

    # 种子可复现
    g1 = torch.Generator(device="cuda").manual_seed(42)
    g2 = torch.Generator(device="cuda").manual_seed(42)
    a = [sample_token(logits, temperature=1.0, generator=g1) for _ in range(10)]
    b = [sample_token(logits, temperature=1.0, generator=g2) for _ in range(10)]
    assert a == b
    print("sampling OK")


if __name__ == "__main__":
    main()
