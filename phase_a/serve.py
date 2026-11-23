#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""qdit 常驻推理 worker: 模型驻留内存显存 + 编译缓存, HTTP API 供 gradio 调用
端点: POST /load {warmup_prompts:[..]}  POST /unload  GET /state
      POST /generate {prompt,neg,seed,steps,cfg,size,sampler}  GET /job/{id}
用法: ~/rocm10/bin/python serve.py [port]   (默认 7861, 仅 127.0.0.1)"""
import json, os, sys, time, threading, queue, datetime
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
import torch._dynamo as _dyn
_dyn.config.cache_size_limit = 128            # 跨提示词长度的编译图全留进程内

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 7861
ENC_DIR = "~/models/qwen21/fp8/qwen3vl8b_fp8"
VAE_CFG = "~/models/qwen21/fp8/vae21_config.json"
VAE_W   = "~/models/qwen21/qwen_image_2.1_vae_diffusers.safetensors"
OUTDIR  = "~/Pictures/qwen"
TPL = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
SYS_TXT = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"

class S:
    lock = threading.Lock()
    tok = enc = lm = vae = model = model_c = None
    drop = mean = std = None
    loaded_at = None
    warm_lens = []
    jobs = {}           # id -> dict(status/step/total/...)
    q = queue.Queue()

# ---------------- 编码 ----------------
@torch.no_grad()
def _encode(text):
    text = text if text else " "
    ids = S.tok(TPL.format(text), return_tensors="pt")["input_ids"]
    out = S.enc(ids, output_hidden_states=True, use_cache=False)
    return out.hidden_states[-1][0].float()[S.drop:]

# ---------------- 加载/卸载 ----------------
def do_load(warmup_prompts):
    if S.model_c is not None:
        return {"ok": True, "note": "已驻留"}
    t0 = time.time()
    _encm = os.environ.get("QDIT_ENC", "sqrot2")
    if _encm == "sqrot2":
        from encoder import load_encoder_sqrot2
        S.tok, S.enc, _lm, S.drop = load_encoder_sqrot2()
    elif _encm == "sqrot":
        from encoder import load_encoder_sqrot
        S.tok, S.enc, _lm, S.drop = load_encoder_sqrot()
    else:
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
        S.tok = AutoTokenizer.from_pretrained(ENC_DIR)
        S.enc = Qwen3VLForConditionalGeneration.from_pretrained(ENC_DIR, torch_dtype=torch.bfloat16).eval()
        S.drop = S.tok(SYS_TXT, return_tensors="pt")["input_ids"].shape[1]
    from loader import load_fp8
    S.model = load_fp8(device="cuda", mode="fp8")
    S.model_c = torch.compile(S.model, dynamic=True)
    cfgd = json.load(open(VAE_CFG))
    for k in ("_class_name", "_diffusers_version"): cfgd.pop(k, None)
    from diffusers.models import AutoencoderKLQwenImage21
    S.vae = AutoencoderKLQwenImage21(**cfgd).to("cuda", torch.bfloat16).eval()
    _ts = int(os.environ.get("QDIT_VAE_TILE", "256"))
    _tst = int(_ts * 0.75) // 32 * 32
    S.vae.enable_tiling(tile_sample_min_height=_ts, tile_sample_min_width=_ts,
                        tile_sample_stride_height=_tst, tile_sample_stride_width=_tst)
    from safetensors.torch import load_file
    raw = load_file(VAE_W)
    sd = S.vae.state_dict()
    remap = {k: v for k, v in raw.items() if k in sd and sd[k].shape == v.shape}
    assert len(remap) == len(sd), f"VAE 键映射不全 {len(remap)}/{len(sd)}"
    S.vae.load_state_dict({k: v.to(torch.bfloat16) for k, v in remap.items()}, strict=False)
    S.mean = torch.tensor(cfgd["latents_mean"], device="cuda").view(1, 64, 1, 1, 1)
    S.std  = torch.tensor(cfgd["latents_std"],  device="cuda").view(1, 64, 1, 1, 1)
    S.loaded_at = time.time()
    load_s = time.time() - t0
    # 编译预热(每个预热提示词 + 负向词" "各一次真实形状前向; CFG 两个前向都要热)
    warm_s = 0.0
    for p in [" "] + list(warmup_prompts or ["a lovely cat holding a wooden sign that says 'qwen fp8'"]):
        tw = time.time()
        try:
            h = _encode(p)
            x = torch.randn(1, 64, 84, 84, device="cuda", dtype=torch.bfloat16)
            t = torch.tensor([500.0], device="cuda")
            with torch.no_grad():
                S.model_c(x, h.unsqueeze(0).to("cuda", torch.bfloat16), t, None, h.shape[0])
            torch.cuda.synchronize()
            S.warm_lens.append(int(h.shape[0]))
        except Exception as e:
            S.warm_lens.append(f"ERR:{type(e).__name__}")
        warm_s += time.time() - tw
    return {"ok": True, "load_s": round(load_s, 1), "warm_s": round(warm_s, 1), "warm_lens": S.warm_lens}

def do_unload():
    for a in ("model_c", "model", "vae", "enc", "lm"): setattr(S, a, None)
    S.tok = None; S.warm_lens = []; S.loaded_at = None
    import gc; gc.collect(); torch.cuda.empty_cache()
    return {"ok": True}

# ---------------- 生成 ----------------
@torch.no_grad()
def do_generate(jid, req):
    prompt = req.get("prompt", " ")
    neg = req.get("neg", " ") or " "
    seed = int(req.get("seed", 42)); steps = int(req.get("steps", 20))
    cfg = float(req.get("cfg", 4.0)); W = H = int(req.get("size", 1344))
    sampler = req.get("sampler", "er_sde")
    t0 = time.time()
    ctx = _encode(prompt).unsqueeze(0).to("cuda", torch.bfloat16)
    ng  = _encode(neg).unsqueeze(0).to("cuda", torch.bfloat16)
    S.jobs[jid]["enc_s"] = round(time.time() - t0, 1)

    from sample import sample, sample_er_sde
    from sigmas import flux_sigmas
    Hl = Wl = H // 16
    sig = flux_sigmas(steps, Hl * Wl).tolist()
    def model_fn(x, c, t, pe, prefix, **kw):
        if pe is None:   # rope 在编译区外构建 (dict 缓存+LT 偏移会特化)
            pe = S.model._pe(c.shape[1], x.shape[2], x.shape[3], x.device)
        return S.model_c(x, c, t, pe, prefix)
    cb = lambda i, n: S.jobs[jid].update(step=i + 1, total=n)
    t1 = time.time()
    if sampler == "euler":
        x = sample(model_fn, ctx, ng, (64, Hl, Wl), sig, cfg=cfg, seed=seed, device="cuda", callback=cb)
    else:
        x = sample_er_sde(model_fn, ctx, ng, (64, Hl, Wl), sig, cfg=cfg, seed=seed, device="cuda", callback=cb)
    S.jobs[jid]["sample_s"] = round(time.time() - t1, 1)

    t2 = time.time()
    lat = (x.unsqueeze(2).to("cuda", torch.bfloat16) * S.std + S.mean).to(torch.bfloat16)
    td = time.time()
    _ev0, _ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    _ev0.record()
    img = S.vae.decode(lat, return_dict=False)[0][:, :, 0]
    _ev1.record()
    torch.cuda.synchronize()
    tp = time.time()
    S.jobs[jid].update(dec_gpu_s=round(_ev0.elapsed_time(_ev1) / 1000, 1))
    img = ((img.float() / 2 + 0.5).clamp(0, 1) * 255).byte()[0].cpu()
    from PIL import Image
    os.makedirs(OUTDIR, exist_ok=True)
    out = f"{OUTDIR}/qwen_{datetime.datetime.now():%m%d_%H%M%S}_qdit-{sampler}_s{seed}.png"
    Image.fromarray(img.permute(1, 2, 0).numpy()).save(out)
    S.jobs[jid].update(vae_s=round(time.time() - t2, 1),
                      dec_s=round(tp - td, 1), post_s=round(time.time() - tp, 1),
                      total_s=round(time.time() - t0, 1), image=out)
    return out

# ---------------- 任务线程 + HTTP ----------------
def worker():
    while True:
        jid, kind, req = S.q.get()
        j = S.jobs.setdefault(jid, {"status": "running"})
        try:
            if kind == "load":
                j.update(S.jobs[jid].get("phase", "") or {}, **do_load(req.get("warmup_prompts")))
            elif kind == "unload":
                j.update(do_unload())
            elif kind == "selftest":
                import time as _t
                with torch.no_grad():
                    _lat = (torch.randn(1, 64, 1, 11, 11, device="cuda", dtype=torch.bfloat16) * 3)
                    _lat = (_lat * S.std.view(1, 64, 1, 1, 1) + S.mean.view(1, 64, 1, 1, 1)).to(torch.bfloat16)
                    def _bench(fn, n=10, warm=3):
                        for _ in range(warm): fn()
                        torch.cuda.synchronize(); t0 = _t.perf_counter()
                        for _ in range(n): fn()
                        torch.cuda.synchronize()
                        return (_t.perf_counter() - t0) / n
                    S.vae.disable_tiling()
                    _tb = _bench(lambda: S.vae.decode(_lat, return_dict=False))
                    S.vae.enable_tiling(tile_sample_min_height=64, tile_sample_min_width=64,
                                        tile_sample_stride_height=48, tile_sample_stride_width=48)
                    _tt = _bench(lambda: S.vae.decode(_lat, return_dict=False))
                    if req.get("big"):
                        from torch.profiler import profile, ProfilerActivity
                        _g = torch.Generator(device="cuda").manual_seed(3)
                        _big = (torch.randn(1, 64, 1, 84, 84, device="cuda", dtype=torch.bfloat16, generator=_g) * 3)
                        _big = (_big * S.std.view(1, 64, 1, 1, 1) + S.mean.view(1, 64, 1, 1, 1)).to(torch.bfloat16)
                        with torch.no_grad():
                            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as _prof:
                                S.vae.decode(_big, return_dict=False)
                        _rows = sorted(_prof.key_averages(), key=lambda e: -max(e.self_device_time_total, e.self_cpu_time_total))[:12]
                        j.update(prof="\n".join(
                            f"{e.key[:46]:<46} gpu={e.self_device_time_total/1000:7.1f}ms cpu={e.self_cpu_time_total/1000:7.1f}ms n={e.count}"
                            for e in _rows))
                    _ts = int(os.environ.get("QDIT_VAE_TILE", "256"))
                    _tst = int(_ts * 0.75) // 32 * 32
                    S.vae.enable_tiling(tile_sample_min_height=_ts, tile_sample_min_width=_ts,
                                        tile_sample_stride_height=_tst, tile_sample_stride_width=_tst)
                    j.update(base_ms=round(_tb * 1000, 1), tile16_ms=round(_tt * 1000, 1), status="done")
            elif kind == "generate":
                if S.model_c is None:
                    j.update(status="error", error="引擎未加载, 先 POST /load")
                    continue
                do_generate(jid, req)
            j["status"] = "done"
        except Exception as e:
            import traceback
            j.update(status="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-800:])

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/state":
            self._json({"loaded": S.model_c is not None,
                        "loaded_at": S.loaded_at, "warm_lens": S.warm_lens,
                        "vram_alloc_gb": round(torch.cuda.memory_allocated() / 2**30, 1) if torch.cuda.is_available() else 0,
                        "busy": S.q.unfinished_tasks > 0})
        elif self.path.startswith("/job/"):
            jid = self.path.split("/")[-1]
            self._json(S.jobs.get(jid, {"status": "unknown"}))
        else:
            self._json({"ok": True})
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        kind = self.path.strip("/")
        jid = f"{kind}-{int(time.time()*1000)%10**9}"
        S.jobs[jid] = {"status": "queued", "kind": kind}
        S.q.put((jid, kind, req))
        self._json({"job": jid}, 202)

if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    print(f"qdit worker 监听 127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
