#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen Draw · qdit 极简面板: 预热 + 提示词 + 进度 + 历史 + 刷新重连
固定配置: 1344² · 20步 er_sde · CFG4 · 负向" " · NSFW 三件套 LoRA 内置(worker 侧)
旧双引擎版在 app_sdcpp.py"""
import os, time, glob, json, random
import subprocess
import gradio as gr
import urllib.request as _ur

HOME = os.path.expanduser("~")
OUTDIR = f"{HOME}/Pictures/qwen"
QDIT = "http://127.0.0.1:7861"
STEPS, CFG, SIZE, SAMPLER = 20, 4.0, 1344, "er_sde"
ACTIVE = "/tmp/qdit_web_active.json"   # 活任务登记: 页面刷新/换设备后自动重连

def _ts_ip():
    try:
        return subprocess.check_output(["tailscale", "ip", "-4"], text=True).strip().splitlines()[0]
    except Exception:
        return "0.0.0.0"

def _post(path, payload=None, timeout=15):
    req = _ur.Request(QDIT + path, data=json.dumps(payload or {}).encode(),
                      headers={"Content-Type": "application/json"})
    with _ur.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def _get(path, timeout=15):
    with _ur.urlopen(QDIT + path, timeout=timeout) as r:
        return json.loads(r.read())

def _active_save(d):
    try:
        json.dump(d, open(ACTIVE, "w"))
    except Exception:
        pass

def _active_load():
    try:
        return json.load(open(ACTIVE))
    except Exception:
        return None

def _active_clear():
    try:
        os.remove(ACTIVE)
    except FileNotFoundError:
        pass
    except Exception:
        pass

def qdit_state():
    try:
        return _get("/state", timeout=5)
    except Exception:
        return {"down": True, "loaded": False}

def status_text():
    st = qdit_state()
    if st.get("down"):
        return "⚪ qdit worker 未运行（82: `cd ~/项目/qwen21dit/phase_a && ~/rocm10/bin/python serve.py`）"
    if not st.get("loaded"):
        return "⚪ 未驻留（点预热，或直接生成会自动加载）"
    return (f"🟢 已驻留 · 热长度 {st.get('warm_lens')} · VRAM {st.get('vram_alloc_gb')}G · "
            f"LoRA 三件套内置 · 1344²/20步/er_sde/CFG4")

def _poll(jid, progress, a, b):
    while True:
        time.sleep(2)
        try:
            j = _get(f"/job/{jid}", timeout=20)
        except Exception:
            continue
        if j.get("status") == "error":
            raise gr.Error("qdit 失败: " + str(j.get("error", ""))[:300])
        if j.get("status") == "done":
            return j
        if j.get("total"):
            progress(a + (b - a) * min(1, j.get("step", 0) / j["total"]),
                     desc=f"采样 {j.get('step', 0)}/{j['total']} 步")
        else:
            progress(a, desc="qdit 加载+编译预热中…（首次 1-3 分钟）")

def warm(prompt, progress=gr.Progress()):
    st = qdit_state()
    if st.get("down"):
        raise gr.Error("qdit worker 未运行")
    progress(0.0, desc="预热（用当前提示词编译）…")
    j = _post("/load", {"warmup_prompts": [prompt.strip() or " "]})["job"]
    _poll(j, progress, 0.0, 1.0)
    progress(1.0, desc="预热完成")
    return status_text()

def _fmt_info(tag, seed, job, t0):
    return (f"{tag} seed `{seed}` · 编码 {job.get('enc_s')}s · 采样 {job.get('sample_s')}s · "
            f"VAE {job.get('vae_s')}s · ⏱ {time.time()-t0:.0f}s\n\n已存: `{job.get('image','')}`")

def generate(prompt, progress=gr.Progress()):
    if not prompt or not prompt.strip():
        raise gr.Error("提示词不能为空")
    st = qdit_state()
    if st.get("down"):
        raise gr.Error("qdit worker 未运行")
    if not st.get("loaded"):
        progress(0.02, desc="首次加载+预热…")
        j = _post("/load", {"warmup_prompts": [prompt.strip()]})["job"]
        _poll(j, progress, 0.02, 0.15)
    seed = random.randint(0, 2**31 - 1)
    t0 = time.time()
    j = _post("/generate", {"prompt": prompt.strip(), "neg": " ", "seed": seed,
                            "steps": STEPS, "cfg": CFG, "size": SIZE, "sampler": SAMPLER})["job"]
    _active_save({"jid": j, "prompt": prompt.strip(), "seed": seed, "t0": t0})
    job = _poll(j, progress, 0.15, 0.95)
    progress(1.0, desc="完成")
    # 不清 active: 留作"最近任务"记录, 页面刷新后由 resume 弹结果/清文件
    return job.get("image"), _fmt_info("", seed, job, t0)

def resume(progress=gr.Progress()):
    """页面(重)加载时: 若有活任务则自动重连, 后台已完成的直接弹结果"""
    a = _active_load()
    if not a or not a.get("jid"):
        return gr.update(), gr.update(), refresh_gallery()
    try:
        j = _get(f"/job/{a['jid']}", timeout=10)
    except Exception:
        return gr.update(), gr.update(), refresh_gallery()
    if j.get("status") == "unknown":       # worker 重启过, 任务没了
        _active_clear()
        return gr.update(), gr.update(), refresh_gallery()
    if j.get("status") == "done":
        _active_clear()
        return (j.get("image"), _fmt_info("(后台已完成)", a.get("seed"), j, a.get("t0", time.time())),
                refresh_gallery())
    if j.get("status") == "error":
        _active_clear()
        return (gr.update(), "*(后台任务失败)* " + str(j.get("error", ""))[:200], gr.update())
    progress(0.15, desc="重连运行中的任务…")
    job = _poll(a["jid"], progress, 0.15, 0.95)
    _active_clear()
    progress(1.0, desc="重连完成")
    return job.get("image"), _fmt_info("(重连完成)", a.get("seed"), job, a.get("t0", time.time())), refresh_gallery()

def refresh_gallery():
    files = sorted(glob.glob(f"{OUTDIR}/qwen_*.png"), key=os.path.getmtime, reverse=True)[:60]
    return [(f, os.path.basename(f)) for f in files]

with gr.Blocks(title="Qwen Draw · qdit") as demo:
    gr.Markdown("# 🎨 Qwen-Image-2.1 · qdit 自研引擎\n"
                "1344² · 20步 er_sde · CFG4 · NSFW 三件套 LoRA 内置（Alpacas v2 + Vaginus + Genitalia修复）")
    status = gr.Markdown()
    prompt = gr.Textbox(label="提示词", lines=5,
                        placeholder="完整句子：主体 → 场景 → 光线 → 机位 → 材质")
    with gr.Row():
        warm_btn = gr.Button("🔥 预热（用当前提示词编译，之后生成零等待）")
        btn = gr.Button("生成", variant="primary")
    image = gr.Image(label="结果", type="filepath", height=520)
    info = gr.Markdown()
    with gr.Accordion("📁 历史画廊（~/Pictures/qwen）", open=True):
        g = gr.Gallery(columns=6, height=320)
        rf = gr.Button("🔄 刷新历史")

    warm_btn.click(warm, [prompt], status)
    btn.click(generate, [prompt], [image, info])
    rf.click(refresh_gallery, None, g)
    demo.load(resume, None, [image, info, g])
    demo.load(refresh_gallery, None, g)
    demo.load(status_text, None, status)

ip = _ts_ip()
demo.queue(default_concurrency_limit=1, max_size=8).launch(
    server_name=ip, server_port=7860, share=False, show_error=True,
    allowed_paths=[OUTDIR])
print(f"qdit-web listening on http://{ip}:7860")
