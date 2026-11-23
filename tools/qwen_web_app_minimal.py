#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen Draw · qdit 极简面板: 预热 + 提示词 + 进度 + 历史 + 刷新重连 + 显存卸载
固定配置: 1344² · 20步 er_sde · CFG4 · 负向" " · NSFW 三件套 LoRA 内置(worker 侧)
旧双引擎版在 app_sdcpp.py"""
import os, time, glob, json, random, threading
import subprocess
import gradio as gr
import urllib.request as _ur
from urllib.error import HTTPError

OUTDIR = os.path.expanduser(os.environ.get("QDIT_OUTDIR", "~/Pictures/qwen"))
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

active_lock = threading.RLock()


def _active_save(d):
    with active_lock:
        try:
            with open(ACTIVE + ".tmp", "w") as f:
                json.dump(d, f)
            os.replace(ACTIVE + ".tmp", ACTIVE)
        except OSError:
            pass


def _active_load():
    with active_lock:
        try:
            with open(ACTIVE) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None


def _active_clear(jid):
    with active_lock:
        if (_active_load() or {}).get("jid") == jid:
            try:
                os.remove(ACTIVE)
            except OSError:
                pass

def qdit_state():
    try:
        return _get("/state", timeout=5)
    except Exception:
        return {"down": True, "loaded": False}

def status_text():
    st = qdit_state()
    if st.get("down"):
        return "⚪ qdit worker 未运行（运行 `python phase_a/serve.py`）"
    if st.get("busy"):
        phase = {"load": "加载 / 编译预热", "generate": "生成中", "unload": "卸载中",
                 "selftest": "自检中", "queued": "任务排队中"}
        return "🟡 " + phase.get(st.get("phase"), "忙碌")
    if not st.get("loaded"):
        return "⚪ 未驻留（点预热，或直接生成会自动加载）"
    idle, lim = st.get("idle_s"), st.get("idle_limit_s")
    if lim and idle is not None:
        left = max(0, lim - idle)
        tail = f" · 空闲 {idle:.0f}s，{left:.0f}s 后自动卸载" if left > 0 else " · 即将自动卸载"
    else:
        tail = " · 自动卸载已关闭"
    return (f"🟢 已驻留 · 热长度 {st.get('warm_lens')} · VRAM {st.get('vram_alloc_gb')}G"
            f"{tail} · LoRA 三件套内置 · 1344²/20步/er_sde/CFG4")

def _poll(jid, progress, a, b):
    failures = 0
    deadline = time.monotonic() + 900
    while True:
        time.sleep(2)
        if time.monotonic() >= deadline:
            raise gr.Error("网页等待超时，任务没有被取消；刷新页面可尝试重连。")
        try:
            j = _get(f"/job/{jid}", timeout=5)
        except (OSError, ValueError) as e:
            failures += 1
            if failures >= 3:
                raise gr.Error("暂时连不上 qdit，刷新页面可尝试重连。") from e
            continue
        failures = 0
        if j.get("status") == "unknown":
            _active_clear(jid)
            raise gr.Error("任务记录已失效（worker 重启或记录已过期）。已保存的图可在历史里查看。")
        if j.get("status") == "error":
            _active_clear(jid)
            raise gr.Error("qdit 失败: " + str(j.get("error", ""))[:300])
        if j.get("status") == "done":
            return j
        if j.get("status") == "queued":
            progress(a, desc="等待 worker…")
        elif j.get("total"):
            progress(a + (b - a) * min(1, j.get("step", 0) / j["total"]),
                     desc=f"采样 {j.get('step', 0)}/{j['total']} 步")
        else:
            phase = {"load": "加载和编译预热…", "generate": "准备生成…", "unload": "释放权重…"}
            progress(a, desc=phase.get(j.get("phase"), "qdit 处理中…"))

def warm(prompt, progress=gr.Progress()):
    st = qdit_state()
    if st.get("down"):
        raise gr.Error("qdit worker 未运行")
    progress(0.0, desc="预热（用当前提示词编译）…")
    j = _post("/load", {"warmup_prompts": [(prompt or "").strip() or " "]})["job"]
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
    seed = random.randint(0, 2**31 - 1)
    t0 = time.time()
    j = _post("/generate", {"prompt": prompt.strip(), "neg": " ", "seed": seed,
                            "steps": STEPS, "cfg": CFG, "size": SIZE, "sampler": SAMPLER})["job"]
    _active_save({"jid": j, "prompt": prompt.strip(), "seed": seed, "t0": t0})
    job = _poll(j, progress, 0.15, 0.95)
    progress(1.0, desc="完成")
    # 不清 active: 留作"最近任务"记录, 页面刷新后由 resume 弹结果/清文件
    return job.get("image"), _fmt_info("", seed, job, t0), refresh_gallery()

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
        _active_clear(a["jid"])
        return gr.update(), "任务记录已失效，已保存的图可在历史里查看。", refresh_gallery()
    if j.get("status") == "done":
        _active_clear(a["jid"])
        return (j.get("image"), _fmt_info("(后台已完成)", a.get("seed"), j, a.get("t0", time.time())),
                refresh_gallery())
    if j.get("status") == "error":
        _active_clear(a["jid"])
        return gr.update(), "*(后台任务失败)* " + str(j.get("error", ""))[:200], refresh_gallery()
    progress(0.15, desc="重连运行中的任务…")
    try:
        job = _poll(a["jid"], progress, 0.15, 0.95)
    except gr.Error as e:
        return gr.update(), str(e), refresh_gallery()
    _active_clear(a["jid"])
    progress(1.0, desc="重连完成")
    return job.get("image"), _fmt_info("(重连完成)", a.get("seed"), job, a.get("t0", time.time())), refresh_gallery()

def unload_now(progress=gr.Progress()):
    try:
        j = _post("/unload")["job"]
    except HTTPError as e:
        if e.code == 409:
            raise gr.Error("有任务在执行或排队，等它结束再卸载。") from e
        raise
    except OSError as e:
        raise gr.Error("连不上 qdit worker") from e
    r = _poll(j, progress, 0.0, 1.0)
    return (f"🔻 已卸载 · 引擎显存 {r.get('vram_before_gb')} → {r.get('vram_after_gb')} GiB\n\n"
            "下次生成会自动重新加载。", status_text())

def refresh_gallery():
    files = sorted(glob.glob(f"{OUTDIR}/qwen_*.png"), key=os.path.getmtime, reverse=True)[:60]
    return [(f, os.path.basename(f)) for f in files]

with gr.Blocks(title="Qwen Draw · qdit") as demo:
    gr.Markdown("# 🎨 Qwen-Image-2.1 · qdit 自研引擎\n"
                "1344² · 20步 er_sde · CFG4 · NSFW 三件套 LoRA 内置（Alpacas v2 + Vaginus + Genitalia修复）")
    status = gr.Markdown()
    status_timer = gr.Timer(5)
    prompt = gr.Textbox(label="提示词", lines=5,
                        placeholder="完整句子：主体 → 场景 → 光线 → 机位 → 材质")
    with gr.Row():
        warm_btn = gr.Button("🔥 预热（加载权重并编译）")
        btn = gr.Button("生成", variant="primary")
        unload_btn = gr.Button("🔻 卸载（释放内存和显存）", variant="stop")
    image = gr.Image(label="结果", type="filepath", height=520)
    info = gr.Markdown()
    with gr.Accordion("📁 历史画廊", open=True):
        g = gr.Gallery(columns=6, height=320)
        rf = gr.Button("🔄 刷新历史")

    warm_btn.click(warm, [prompt], status)
    btn.click(generate, [prompt], [image, info, g])
    unload_btn.click(unload_now, None, [info, status])
    rf.click(refresh_gallery, None, g)
    demo.load(resume, None, [image, info, g])
    demo.load(status_text, None, status, queue=False)
    status_timer.tick(status_text, None, status, queue=False)

if __name__ == "__main__":
    ip = _ts_ip()
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=ip, server_port=7860, share=False, show_error=True,
        allowed_paths=[OUTDIR])
