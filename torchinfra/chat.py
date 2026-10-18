"""交互对话 CLI：PYTHONPATH=. ~/rocm10/bin/python -m torchinfra.chat"""
import argparse
import time
from pathlib import Path
import torch
from torchinfra.model import Qwen3ForCausalLM
from torchinfra.tokenizer import TokenizerBox
from torchinfra.generate import generate

SYSTEM = "你是一个乐于助人的中文助手，回答简洁准确。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "models" / "Qwen3-0.6B"))
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    print("加载模型中……")
    model = Qwen3ForCausalLM.from_pretrained(args.model)
    tok = TokenizerBox(args.model)
    messages = [{"role": "system", "content": SYSTEM}]
    print("模型就绪，输入文字开始对话（Ctrl-D 退出）")
    while True:
        try:
            user = input("\n用户> ").strip()
        except EOFError:
            break
        if not user:
            continue
        messages.append({"role": "user", "content": user})
        ids = tok.apply_chat(messages)
        t0 = time.perf_counter()
        n_out = 0
        print("助手> ", end="", flush=True)
        turn_ids = []
        for t, delta in generate(model, tok, ids, args.max_new,
                                 args.temperature, top_p=args.top_p,
                                 seed=args.seed):
            print(delta, end="", flush=True)
            turn_ids.append(t)
            n_out += 1
        dt = time.perf_counter() - t0
        print(f"\n[{n_out} tokens / {dt:.1f}s = {n_out / dt:.1f} tok/s]")
        messages.append({"role": "assistant", "content": tok.decode(turn_ids)})


if __name__ == "__main__":
    main()
