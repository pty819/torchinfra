#!/bin/bash
# qwen-draw — Qwen-Image-2.1 Omni(uncensored, INT8) + Lightning 4步 + NSFW LoRA
# 用法: qwen-draw "提示词" [-s seed] [-w 宽] [-h 高] [-n nsfw强度|0关] [-o 输出.png]
#       提示词传 "-" 则从 stdin 读
D=~/models/qwen21
CLI=$HOME/build/qwen21/stable-diffusion.cpp/build-hip/bin/sd-cli
source ~/rocm10/bin/rocm-env.sh

PROMPT=""; SEED=""; W=1024; H=1024; NSFW=1.0; OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    -s) SEED="$2"; shift 2;;
    -w) W="$2"; shift 2;;
    -h) H="$2"; shift 2;;
    -n) NSFW="$2"; shift 2;;
    --no-nsfw) NSFW="0"; shift;;
    -o) OUT="$2"; shift 2;;
    *) PROMPT="$1"; shift;;
  esac
done
[ "$PROMPT" = "-" ] && PROMPT=$(cat)
if [ -z "$PROMPT" ]; then
  echo "用法: qwen-draw \"提示词\" [-s seed] [-w 1024] [-h 1024] [-n nsfw强度,0=关] [-o out.png]"; exit 1
fi
[ -z "$SEED" ] && SEED=$((RANDOM * 32768 + RANDOM))
mkdir -p ~/Pictures/qwen
OUT="${OUT:-$HOME/Pictures/qwen/qwen_$(date +%m%d_%H%M%S)_s${SEED}.png}"

P="$PROMPT <lora:lightning_4step:1>"
if [ "$NSFW" != "0" ] && [ -n "$NSFW" ]; then P="$P <lora:nsfw_qwen:$NSFW>"; fi

t0=$SECONDS
$CLI --diffusion-model $D/omni-int8-convrot-comfy.safetensors \
     --vae $D/qwen_image_2.1_vae_bf16.safetensors \
     --llm $D/Qwen3VL-8B-Instruct-Q8_0.gguf \
     --lora-model-dir $D/loras --params-backend te=cpu \
     -W $W -H $H --sampling-method euler --seed $SEED \
     --flash-attn off,diffusion=on \
     -p "$P" --sigmas 1.0,0.75,0.5,0.25,0.0 --steps 4 --cfg-scale 1.0 \
     -v -o "$OUT" 2>&1 | grep -E "\[E\]|LoRA tensors have been applied|sampling completed" | tail -4
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then echo "生成失败 rc=$rc"; exit $rc; fi
echo "→ $OUT  ($((SECONDS-t0))s, seed=$SEED)"
