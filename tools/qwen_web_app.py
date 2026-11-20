#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""qwen-web — Qwen-Image-2.1 Omni(uncensored) × Lightning 的 Gradio 前端
跑在 82 (RX 9070 GRE · HIP)，只监听 tailscale IP，tailnet 内访问。
- 无参考图: 文生图
- 有参考图: Qwen-Image-2.1 原生编辑模式 (mmproj 视觉投影)
NSFW LoRA 多选叠加；🔞档按社区配方 er_sde + CFG 4（这类 LoRA 在 CFG 1 + euler 下效果差）。
"""
import os, re, random, shutil, subprocess, time, glob
from datetime import datetime
from PIL import Image
import gradio as gr

HOME = os.path.expanduser("~")
D = f"{HOME}/models/qwen21"
CLI = f"{HOME}/build/qwen21/stable-diffusion.cpp/build-hip/bin/sd-cli"
MMPROJ = f"{D}/mmproj-Qwen3VL-8B-Instruct-F16.gguf"
OUTDIR = f"{HOME}/Pictures/qwen"

# ---- qdit 自研引擎 (常驻 worker, 127.0.0.1:7861) ----
QDIT_URL = "http://127.0.0.1:7861"
import json as _json, urllib.request as _ur
def _qdit_post(path, payload=None, timeout=15):
    req = _ur.Request(QDIT_URL + path, data=_json.dumps(payload or {}).encode(),
                      headers={"Content-Type": "application/json"})
    with _ur.urlopen(req, timeout=timeout) as r:
        return _json.loads(r.read())
def _qdit_get(path, timeout=15):
    with _ur.urlopen(QDIT_URL + path, timeout=timeout) as r:
        return _json.loads(r.read())
def qdit_state():
    try:
        return _qdit_get("/state", timeout=5)
    except Exception:
        return {"down": True, "loaded": False}
def _qdit_poll(jid, progress=None, a=0.0, b=1.0):
    import time as _t
    while True:
        _t.sleep(2)
        try:
            j = _qdit_get(f"/job/{jid}", timeout=20)
        except Exception:
            continue
        if j.get("status") == "error":
            raise gr.Error("qdit 失败: " + str(j.get("error", ""))[:300])
        if j.get("status") == "done":
            return j
        if progress and j.get("total"):
            progress(a + (b - a) * min(1, j.get("step", 0) / j["total"]),
                     desc=f"qdit 采样 {j.get('step', 0)}/{j['total']} 步")
        elif progress:
            progress(a, desc="qdit 加载+编译预热中…（首次 1-3 分钟）")
os.makedirs(OUTDIR, exist_ok=True)

# 档位 -> (steps, sigmas, cfg, 蒸馏LoRA文件名或None, 采样器)
PRESETS = {
    "⚡ Lightning 4步 (最快·1024²约45s)": (4, "1.0,0.75,0.5,0.25,0.0", 1.0, "lightning_4step", "euler"),
    "⚡ Viggle 6步 (快·NSFW LoRA更兼容)": (6, "1.0,0.9375,0.875,0.75,0.5,0.25,0.0", 1.0, "viggle_6step", "euler"),
    "🔞 NSFW标准 20步 (er_sde·CFG 4)": (20, None, 4.0, None, "er_sde"),
    "🎯 姿势精修 50步 (er_sde·CFG 2.5·最稳)": (50, None, 2.5, None, "er_sde"),
    "🖼 高质量 25步 (euler·CFG 4)": (25, None, 4.0, None, "euler"),
}
DEFAULT_PRESET_T2I = "⚡ Lightning 4步 (最快·1024²约45s)"
DEFAULT_PRESET_EDIT = "⚡ Viggle 6步 (快·NSFW LoRA更兼容)"

SIZE_PRESETS = [
    "1024×1024  1MP 快",
    "1344×1344  1.8MP 推荐",
    "1536×1024  1.5MP 横",
    "1024×1536  1.5MP 竖",
    "1728×1152  2MP 横",
    "1152×1728  2MP 竖人像",
    "1664×2432  4MP 大画幅",
]

# NSFW LoRA（每行一个开关+独立强度；可叠加）
NSFW_LORAS = {  # 显示名 -> (文件名, 默认强度)
    "TheseAlpacas v2 (社区最热)": ("nsfw_alpacas_v2", 0.8),
    "FemaledVaginus v1.3b (兼容CFG1/编辑)": ("nsfw_vaginus_v13b", 0.9),
    "Genitalia解剖修复 (可叠加)": ("nsfw_genitalia_fix", 0.6),
    "f23gg v1 (旧默认)": ("nsfw_qwen", 1.0),
}
DEFAULT_NSFW_ON = ["TheseAlpacas v2 (社区最热)"]

STEP_RE = re.compile(r"step\s+(\d+)\s*(?:of|/)\s*(\d+)", re.I)


def _ts_ip():
    """拿 tailscale0 的 100.x 地址，没有就退回 0.0.0.0 并警告。"""
    try:
        out = subprocess.run(["ip", "-4", "addr", "show", "tailscale0"],
                             capture_output=True, text=True, timeout=5).stdout
        m = re.search(r"inet\s+(100\.\d+\.\d+\.\d+)", out)
        if m:
            return m.group(1)
    except Exception:
        pass
    print("WARNING: 未找到 tailscale0 接口，回退监听 0.0.0.0（将暴露到局域网）")
    return "0.0.0.0"


def _snap32(v, lo=256, hi=1536):
    v = max(lo, min(hi, int(v)))
    return max(32, round(v / 32) * 32)


def preset_text(preset):
    steps, _, cfg, distill, sampler = PRESETS.get(preset, (0, None, 0, None, "?"))
    return f"{steps}步/{sampler}/CFG {cfg}" + (f" + {distill}" if distill else "")


def generate_qdit(prompt, negative, refs, preset, width, height, seed, progress=gr.Progress()):
    if refs:
        raise gr.Error("qdit 引擎暂不支持参考图编辑, 请切回 sdcpp")
    steps, _sig, cfg, _distill, sampler = PRESETS.get(preset, PRESETS[DEFAULT_PRESET_T2I])
    st = qdit_state()
    if st.get("down"):
        raise gr.Error("qdit worker 未运行 (82: cd ~/项目/qwen21dit/phase_a && ~/rocm10/bin/python serve.py)")
    if not st["loaded"]:
        progress(0.02, desc="qdit 首次加载+预热…")
        j = _qdit_post("/load", {"warmup_prompts": [prompt.strip()]})["job"]
        _qdit_poll(j, progress, 0.02, 0.15)
    w = _snap32(width or 1344); h = _snap32(height or 1344)
    if w != h:
        raise gr.Error("qdit 引擎当前只支持方形输出 (宽=高)")
    if seed is None or seed < 0:
        seed = random.randint(0, 2**31 - 1)
    t0 = time.time()
    j = _qdit_post("/generate", {"prompt": prompt.strip(), "neg": (negative or " ").strip() or " ",
                                 "seed": int(seed), "steps": int(steps), "cfg": cfg,
                                 "size": int(w), "sampler": "er_sde" if "er_sde" in str(sampler) else "euler"})["job"]
    job = _qdit_poll(j, progress, 0.15, 0.95)
    out = job.get("image")
    dt = time.time() - t0
    info = (f"**qdit 自研引擎** · {w}×{h} · seed `{seed}` · 编码 {job.get('enc_s')}s · "
            f"采样 {job.get('sample_s')}s · VAE {job.get('vae_s')}s · ⏱ 全程 {dt:.0f}s\n\n"
            f"已存: `{out}`\n\n🔥 内置固定 LoRA 套装: Alpacas v2@0.8 · Vaginus@0.9 · Genitalia修复@0.6（无需勾选）")
    progress(1.0, desc="完成")
    return out, info, seed


def generate(engine, prompt, negative, refs, preset, width, height, seed, follow, *nsfw_args,
             progress=gr.Progress()):
    t0 = time.time()
    if not prompt or not prompt.strip():
        raise gr.Error("提示词不能为空")
    if str(engine).startswith("qdit"):
        return generate_qdit(prompt, negative, refs, preset, width, height, seed, progress)
    refs = [r for r in (refs or []) if r]
    if refs and not os.path.exists(MMPROJ):
        raise gr.Error(f"编辑模式缺 mmproj 文件: {MMPROJ}")

    # nsfw_args = [on_1, str_1, on_2, str_2, ...] 按 NSFW_LORAS 顺序
    picks = []
    for i, (name, (fname, _)) in enumerate(NSFW_LORAS.items()):
        on, s = nsfw_args[2 * i], nsfw_args[2 * i + 1]
        if on and s and s > 0:
            picks.append((fname, float(s)))

    steps, sigmas, cfg, distill, sampler = PRESETS.get(preset, PRESETS[DEFAULT_PRESET_T2I])
    p = prompt.strip()
    if distill:
        p += f" <lora:{distill}:1>"
    for fname, s in picks:
        p += f" <lora:{fname}:{s:.2f}>"

    w, h = _snap32(width or 1024), _snap32(height or 1024)
    if refs and follow_ref:
        with Image.open(refs[0]) as im:
            w, h = _snap32(im.width), _snap32(im.height)

    if seed is None or seed < 0:
        seed = random.randint(0, 2**31 - 1)

    stamp = datetime.now().strftime("%m%d_%H%M%S")
    out = f"/tmp/qwen_web_{stamp}_s{seed}.png"
    mode = "编辑" if refs else "文生图"

    st = qdit_state()      # 显存互斥: sdcpp 跑之前卸掉常驻的 qdit
    if st.get("loaded"):
        try:
            _qdit_post("/unload")
        except Exception:
            pass
    cmd = [CLI,
           "--diffusion-model", f"{D}/omni-int8-convrot-comfy.safetensors",
           "--vae", f"{D}/qwen_image_2.1_vae_bf16.safetensors",
           "--llm", f"{D}/Qwen3VL-8B-Instruct-Q8_0.gguf",
           "--lora-model-dir", f"{D}/loras",
           "--params-backend", "te=cpu",
           "-W", str(w), "-H", str(h),
           "--sampling-method", sampler, "--cfg-scale", str(cfg),
           "-s", str(seed), "--steps", str(steps), "-v"]
    if sigmas:
        cmd += ["--sigmas", sigmas]
    if refs:
        cmd += ["--llm_vision", MMPROJ]
        for r in refs:
            cmd += ["-r", r]
    if negative and cfg > 1:
        cmd += ["-n", negative]
    cmd += ["-p", p, "-o", out]

    # 用 bash 先 source rocm 环境再 exec，参数原样传给 sd-cli
    run = ["bash", "-c", 'source "$HOME/rocm10/bin/rocm-env.sh" && exec "$@"', "sd-cli"] + cmd
    progress(0.0, desc=f"[{mode}] 加载模型中…")

    log_tail = []
    try:
        proc = subprocess.Popen(run, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)
        for line in proc.stdout:
            m = STEP_RE.search(line)
            if m:
                cur, tot = int(m.group(1)), int(m.group(2))
                progress(0.1 + 0.85 * min(1.0, cur / max(tot, 1)),
                         desc=f"[{mode}] 采样 {cur}/{tot} 步")
            log_tail.append(line.rstrip())
            del log_tail[:-15]
        rc = proc.wait(timeout=900)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise gr.Error("生成超时（15 分钟），已终止")
    if rc != 0 or not os.path.exists(out):
        raise gr.Error("sd-cli 失败：\n" + "\n".join(log_tail[-8:]))

    progress(0.98, desc="保存…")
    saved = f"{OUTDIR}/qwen_{stamp}_s{seed}.png"
    shutil.copy(out, saved)
    dt = time.time() - t0
    loras_txt = "+".join(f"{f}@{s:.2f}" for f, s in picks) or "无"
    info = (f"**{mode}** · {w}×{h} · seed `{seed}` · {preset_text(preset)} · "
            f"NSFW `{loras_txt}` · ⏱ {dt:.0f}s\n\n"
            f"已存: `{saved}`")
    progress(1.0, desc="完成")
    return out, info, seed


def refresh_gallery():
    files = sorted(glob.glob(f"{OUTDIR}/qwen_*.png"), key=os.path.getmtime, reverse=True)[:60]
    return [(f, os.path.basename(f)) for f in files]


with gr.Blocks(title="Qwen Draw · 82") as demo:
    gr.Markdown("# 🎨 Qwen-Image-2.1 · Omni × Lightning\n"
                "RX 9070 GRE · HIP · 仅 tailnet 内可访问 · 同一时间跑一张（排队）\n"
                "💡 质量三板斧：**尺寸上 1.8MP+** · **完整句子提示词**（主体→场景→光线→机位→材质）· "
                "**NSFW 配 🔞/Viggle 档**。姿势用「双图编辑」见下方折叠说明")
    with gr.Row():
        with gr.Column(scale=5):
            prompt = gr.Textbox(label="提示词", lines=4, max_lines=10,
                                placeholder="完整描述画面；上传参考图后请描述想要的改动，如 \"把招牌上的文字改成 sd.cpp\"")
            negative = gr.Textbox(label="负向提示词（仅非 CFG1 档生效）", lines=1,
                                  value="extra limbs, deformed hands, fused fingers, bad anatomy")
            refs = gr.File(label="参考图（0~N 张；≥1 张即进入编辑模式，多张按顺序传入）",
                           file_count="multiple", type="filepath", file_types=["image"])
            preset = gr.Radio(list(PRESETS.keys()), value=DEFAULT_PRESET_T2I, label="采样档位")
            nsfw_widgets = []
            with gr.Accordion("🔞 NSFW LoRA（可叠加，独立强度）", open=True):
                for name, (fname, s) in NSFW_LORAS.items():
                    with gr.Row():
                        on = gr.Checkbox(value=(name in DEFAULT_NSFW_ON), label=name, scale=3)
                        st = gr.Slider(0.0, 1.5, value=s, step=0.05, label="强度", scale=2)
                        nsfw_widgets += [on, st]
            size_dd = gr.Dropdown(SIZE_PRESETS, value="1344×1344  1.8MP 推荐",
                                  label="尺寸预设（甜点区 1.8~2MP，1024 会糊）")
            with gr.Row():
                width = gr.Number(value=1344, precision=0, label="宽（32 倍数）")
                height = gr.Number(value=1344, precision=0, label="高（32 倍数）")
                seed = gr.Number(value=-1, precision=0, label="seed（-1 随机）")
            follow = gr.Checkbox(value=True, label="有参考图时，输出尺寸自动跟随第一张参考图")
            with gr.Accordion("💡 姿势控制玩法（双图编辑）", open=False):
                gr.Markdown("1️⃣ 图1 = 姿势图（3D 木偶截图 / 任何摆好姿势的图）\n\n"
                            "2️⃣ 图2 = 人物照（想要的那个人/角色）\n\n"
                            "3️⃣ 提示词：`draw the character from image 2 in the pose from image 1` "
                            "再补场景与光线\n\n"
                            "4️⃣ 档位选 ⚡Viggle 或 🎯姿势精修。提示词写完整句子（主体→场景→光线方向→机位→材质），"
                            "CFG 1 档下负向词无效，限制写进正向。")
            engine = gr.Radio(["sdcpp", "qdit(自研·常驻)"], value="sdcpp", label="推理引擎")
            with gr.Accordion("🚀 qdit 引擎管理（预热后生成零等待）", open=False):
                warm_btn = gr.Button("🔥 预热/驻留（用当前提示词编译缓存）")
                qdit_status = gr.Markdown()
            btn = gr.Button("生成", variant="primary")
        with gr.Column(scale=4):
            image = gr.Image(label="结果", type="filepath", height=520)
            info = gr.Markdown()
    with gr.Accordion("📁 历史画廊（~/Pictures/qwen）", open=False):
        gallery = gr.Gallery(columns=6, height=300)
        refresh = gr.Button("🔄 刷新历史")

    def on_refs_change(files):
        return gr.update(value=DEFAULT_PRESET_EDIT if files else DEFAULT_PRESET_T2I)
    refs.change(on_refs_change, refs, preset)

    def on_size(sel):
        try:
            w, h = sel.split()[0].split("×")
            return int(w), int(h)
        except Exception:
            return gr.update(), gr.update()
    size_dd.change(on_size, size_dd, [width, height])

    # 任一 NSFW LoRA 打开 → 自动切 🔞 NSFW 档（这类 LoRA 在 CFG1 下效果差）
    preset_after_nsfw = "🔞 NSFW标准 20步 (er_sde·CFG 4)"
    for i in range(0, len(nsfw_widgets), 2):
        def _mk():
            def fn(on_val):
                if on_val:
                    return gr.update(value=preset_after_nsfw)
                return gr.update()
            return fn
        nsfw_widgets[i].change(_mk(), [nsfw_widgets[i]], preset)

    def qdit_refresh_status():
        st = qdit_state()
        if st.get("down"):
            return "⚪ qdit worker 未运行"
        if not st.get("loaded"):
            return "⚪ 已连接 · 未驻留（生成时自动加载, 或点预热）"
        return f"🟢 已驻留 · 热长度 {st.get('warm_lens')} · VRAM {st.get('vram_alloc_gb')}G"

    def qdit_warm(p):
        st = qdit_state()
        if st.get("down"):
            raise gr.Error("qdit worker 未运行")
        j = _qdit_post("/load", {"warmup_prompts": [p.strip() or " "]})["job"]
        _qdit_poll(j)
        return qdit_refresh_status()

    warm_btn.click(qdit_warm, [prompt], qdit_status)
    btn.click(generate,
              inputs=[engine, prompt, negative, refs, preset, width, height, seed, follow] + nsfw_widgets,
              outputs=[image, info, seed])
    refresh.click(refresh_gallery, None, gallery)
    demo.load(refresh_gallery, None, gallery)
    demo.load(qdit_refresh_status, None, qdit_status)

ip = _ts_ip()
demo.queue(default_concurrency_limit=1, max_size=8).launch(
    server_name=ip, server_port=7860, share=False, show_error=True,
    allowed_paths=[OUTDIR])
print(f"qwen-web listening on http://{ip}:7860")
