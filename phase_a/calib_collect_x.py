# SpinQuant 迷你版第一步: 收集每层 Linear 的输入样本 (每层 96 行随机采样)
import sys, gc, torch
sys.path.insert(0, "~/项目/qwen21dit/phase_a")
import torch.nn as nn
from transformers import AutoTokenizer
from encoder import load_encoder, TPL
CAL_PROMPTS = ["masterpiece, a serene mountain lake at dawn, mist over water, ultra detailed",
    "1girl, cute anime style, cherry blossoms, golden hour lighting, ultra detailed",
    "a cute orange cat sitting on a wooden table, soft light",
    " masterpiece, best quality, absurdres, 1girl, solo, long hair, detailed face, collarbone, seductive pose, lying on bed, soft lighting, nsfw, detailed skin texture",
    " scenic view of mountain range at sunset, dramatic clouds, wide angle, 8k",
    "1boy, handsome, suit, city street at night, neon lights, cinematic",
    " 美式动画风格, 一个女孩站在屋顶看城市夜景, 电影感构图",
    "a", "close-up photo of a mechanical watch, macro lens",
    "1girl, twintails, school uniform, classroom, anime screenshot style",
    "fantasy castle floating in the sky, waterfalls falling into clouds, concept art",
    "portrait of an old fisherman, dramatic side lighting, black and white",
    " masterpiece, best quality, 1girl, submissive pose, bdsm dungeon theme, dim lighting, nsfw",
    "a bowl of steaming ramen, shallow depth of field",
    "cyberpunk street market, rain, holographic signs",
    "watercolor painting of a small cottage in a meadow",
    "2girls, dancing, stage lights, concert, dynamic pose",
    "an astronaut floating above earth, photorealistic",
    "chibi cat warrior with tiny sword, sticker style",
    "dark fantasy knight in black armor, glowing runes, fog",
    "extreme long prompt with many clauses, masterpiece, best quality, ultra detailed, cinematic, 8k",
    "a single red leaf falling", "1girl, maid outfit, mansion interior, warm light",
    "snowy mountain peak at blue hour", "cute shiba inu with sunglasses, skateboard",
    " 水彩风格, 静物, 一束干花插在玻璃瓶里", "pixel art, cozy room, cat sleeping",
    "1girl, kimono, autumn maple leaves, japanese garden",
    "underwater coral reef, school of tropical fish",
    "brutal close-up of a tarantula face, macro photography"]
from safetensors.torch import save_file

OUT = "~/models/qwen21/int8_qwen3vl8b.xsamples.safetensors"
NSAMP = 96
TOPK = 48   # absmax 最大的行(massive 点)必进样本

tok_, enc_bf, _, _ = load_encoder()

samples = {}   # name -> [NSAMP, K] fp32
def make_hook(name):
    def hook(mod, inp, out):
        x = inp[0].detach().reshape(-1, x_dim[0] if False else inp[0].shape[-1]).float()
        n = x.shape[0]
        if n >= NSAMP:
            idx = torch.randperm(n)[:NSAMP]
        else:
            idx = torch.randint(0, n, (NSAMP,))
        # 就地采样: absmax top16 行 + 8 随机行 (每 prompt), 不累积全量
        rowmax = x.abs().amax(dim=1)
        top = torch.topk(rowmax, min(16, x.shape[0])).indices
        rnd = idx[:8]
        cur = x[torch.cat([top, rnd])]
        samples[name] = cur if name not in samples else torch.cat([samples[name], cur], 0)
    return hook

hooks, x_dim = [], [0]
for n, m in enc_bf.named_modules():
    if isinstance(m, nn.Linear) and n.startswith("model.language_model"):
        hooks.append(m.register_forward_hook(make_hook(n)))
print(f"hook {len(hooks)}")

with torch.no_grad():
    for i, p in enumerate(CAL_PROMPTS):
        ids = tok_(TPL.format(p), return_tensors="pt")["input_ids"]
        enc_bf(ids, output_hidden_states=True, use_cache=False)
        if (i + 1) % 8 == 0:
            print(f"  {i+1}/{len(CAL_PROMPTS)}")
for h in hooks:
    h.remove()

# 每层: absmax 最大的 TOPK 行 + 随机补齐到 NSAMP
out = {}
g = torch.Generator().manual_seed(7)
for k, v in samples.items():
    rowmax = v.abs().amax(dim=1)
    top = torch.topk(rowmax, min(TOPK, v.shape[0])).indices
    rest_pool = torch.tensor([i for i in range(v.shape[0]) if i not in set(top.tolist())])
    nrest = max(0, NSAMP - len(top))
    if len(rest_pool) and nrest:
        ridx = rest_pool[torch.randperm(len(rest_pool), generator=g)[:nrest]]
    else:
        ridx = torch.empty(0, dtype=torch.long)
    out[k] = v[torch.cat([top, ridx])].contiguous()
    samples[k] = None   # 释放
save_file(out, OUT)
tot = sum(v.numel() * 4 for v in out.values()) / 2**30
print(f"OK {OUT}: {len(out)} 层, {tot:.2f}GB")
del samples, out
print("XSAMPLES_DONE")
