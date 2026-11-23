# SmoothQuant 校准: hook 每个语言塔 Linear 的输入, 统计 per-channel absmax
# 输出: 每层 [K] fp32 向量 (safetensors)
from qwen21dit.paths import MODEL_DIR
import json, time, sys, gc, torch
from transformers import AutoTokenizer
from qwen21dit.encoder import load_encoder, TPL

OUT = str(MODEL_DIR / "int8_qwen3vl8b.calib.safetensors")

# 校准集: 长短混合、中英混合、真实使用风格
CAL_PROMPTS = [
    "masterpiece, a serene mountain lake at dawn, mist over water, ultra detailed",
    "1girl, cute anime style, cherry blossoms, golden hour lighting, ultra detailed",
    "a cute orange cat sitting on a wooden table, soft light",
    " masterpiece, best quality, absurdres, 1girl, solo, long hair, silver hair, red eyes, detailed face, pale skin, collarbone, bare shoulders, off shoulder dress, lace trim, garter straps, thighhighs, black panties, seductive pose, lying on bed, bedroom, soft lighting, depth of field, nsfw, detailed skin texture",
    " scenic view of mountain range at sunset, dramatic clouds, birds in the sky, wide angle, landscape photography, 8k",
    "1boy, handsome, suit, city street at night, neon lights, cinematic",
    " 美式动画风格, 一个女孩站在屋顶看城市夜景, 风吹起头发, 电影感构图",
    "a",
    "close-up photo of a mechanical watch, macro lens, extreme detail",
    "1girl, twintails, school uniform, classroom, sunlight through window, anime screenshot style, high quality, detailed eyes",
    "fantasy castle floating in the sky, waterfalls falling into clouds, epic scale, concept art",
    "portrait of an old fisherman, weathered face, dramatic side lighting, black and white photography",
    " masterpiece, best quality, 1girl, submissive pose, bdsm dungeon theme, rope bondage, dim lighting, cinematic shadows, detailed skin, nsfw",
    "a bowl of steaming ramen, chopsticks, wooden table, shallow depth of field, food photography",
    "cyberpunk street market, rain, reflections on wet ground, holographic signs, busy crowd",
    "watercolor painting of a small cottage in a meadow, wildflowers, loose brush strokes",
    "2girls, dancing, stage lights, concert, dynamic pose, detailed costumes",
    "an astronaut floating above earth, visor reflection showing the planet, photorealistic",
    "chibi cat warrior with tiny sword, sticker style, simple background",
    " architectural photography, brutalist concrete building, fog, minimalist composition, muted tones",
    "extreme long prompt with many clauses and descriptors, masterpiece, best quality, ultra detailed, intricate details, professional lighting, volumetric light, sharp focus, high contrast, cinematic composition, rule of thirds, film grain, 8k uhd, dslr, high quality, detailed background, bokeh, depth of field",
    "a single red leaf falling",
    "1girl, maid outfit, holding a tray with tea, mansion interior, warm light, victorian",
    "snowy mountain peak at blue hour, alpenglow, stars beginning to appear",
    "cute shiba inu wearing sunglasses, skateboard, summer vibe, vibrant colors",
    "dark fantasy knight in black armor, glowing runes, fog, dramatic low angle",
    "水彩风格, 静物, 一束干花插在玻璃瓶里, 柔和的光",
    "pixel art scene, cozy room, cat sleeping on retro computer, rain outside window",
    "1girl, kimono, autumn maple leaves, traditional japanese garden, falling petals",
    "underwater coral reef, school of tropical fish, sunbeams penetrating water",
    "brutal close-up of a tarantula face, macro photography, high detail texture",
]

print(f"校准 prompt: {len(CAL_PROMPTS)} 条")
tok_, enc_bf, lm_bf, drop = load_encoder()

stats = {}   # name -> [K] fp32 absmax (跨样本 max)
hooks = []
import torch.nn as nn

def make_hook(name):
    def hook(mod, inp, out):
        x = inp[0]
        a = x.detach().abs().amax(dim=tuple(range(x.dim() - 1)))  # per-channel [K]
        if name in stats:
            torch.maximum(stats[name], a, out=stats[name])
        else:
            stats[name] = a.clone().float()
    return hook

nlin = 0
for n, m in enc_bf.named_modules():
    if isinstance(m, nn.Linear) and n.startswith("model.language_model"):
        hooks.append(m.register_forward_hook(make_hook(n)))
        nlin += 1
print(f"hook {nlin} Linear")

with torch.no_grad():
    for i, p in enumerate(CAL_PROMPTS):
        ids = tok_(TPL.format(p), return_tensors="pt")["input_ids"]
        enc_bf(ids, output_hidden_states=True, use_cache=False)
        if (i + 1) % 8 == 0:
            print(f"  {i+1}/{len(CAL_PROMPTS)}")

for h in hooks:
    h.remove()

from safetensors.torch import save_file
out = {k: v.contiguous() for k, v in stats.items()}
save_file(out, OUT)
print(f"OK {OUT}: {len(out)} 层统计")
# 离群概况: 打印最严重的几层
worst = sorted(out.items(), key=lambda kv: -kv[1].max().item())[:6]
for k, v in worst:
    print(f"  {k}: max={v.max():.1f} median={v.median():.2f} ratio={v.max()/v.median():.0f}x")
print("CALIB_DONE")
