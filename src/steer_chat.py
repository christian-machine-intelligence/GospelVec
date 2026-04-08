"""
Gospel Steering Chat Interface.

Loads Qwen 3.5 9B with steering hooks that add Gospel direction vectors
to the residual stream at a specific layer during generation.

Usage:
  python3 src/steer_chat.py
  python3 src/steer_chat.py --alpha 2.0    # stronger steering
  python3 src/steer_chat.py --layer 30     # override layer

Commands during chat:
  /matthew 2.0    — set Matthew steering strength
  /mark -1.0      — set Mark steering (negative = suppress)
  /luke 3.0       — set Luke steering
  /john 0.0       — disable John steering
  /alpha 2.0      — set all Gospels to same strength
  /reset           — zero all steering
  /status          — show current steering config
  /temp 0.5        — change temperature
  /tokens 512      — change max tokens
"""

import argparse
import json
import logging
import os
import sys
import textwrap
import time
from pathlib import Path

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import GOSPELS, HIDDEN_DIM, MODEL_ID, VECTOR_DIR, USE_QUANTIZATION

if USE_QUANTIZATION:
    from transformers import BitsAndBytesConfig

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
COLORS = {
    "matthew": "\033[91m",
    "mark":    "\033[93m",
    "luke":    "\033[92m",
    "john":    "\033[94m",
}
HRULE = "─" * 72


class GospelSteerer:
    """Hooks into one or more layers to add Gospel direction vectors."""

    def __init__(self, model, layer_indices: list[int],
                 all_layer_vectors: torch.Tensor, device):
        """
        Args:
            model: the transformer model
            layer_indices: list of decoder layer indices to intervene at
            all_layer_vectors: [num_layers, 4, hidden_dim] vectors for all layers
            device: target device
        """
        self.layer_indices = layer_indices
        self.device = device

        # Store per-layer vectors only for active layers
        self.layer_vectors = {}
        for li in layer_indices:
            self.layer_vectors[li] = all_layer_vectors[li].to(device)  # [4, hidden_dim]

        # Steering strengths (one per Gospel)
        self.alphas = {gospel: 0.0 for gospel in GOSPELS}

        # Register hooks on all active layers
        self.hooks = []
        for li in layer_indices:
            hook = model.model.layers[li].register_forward_hook(
                self._make_hook(li)
            )
            self.hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            vectors = self.layer_vectors[layer_idx]

            steering = torch.zeros(vectors.shape[1], device=self.device,
                                   dtype=hidden.dtype)
            for gi, gospel in enumerate(GOSPELS):
                alpha = self.alphas[gospel]
                if alpha != 0.0:
                    steering += alpha * vectors[gi]

            hidden = hidden + steering.unsqueeze(0).unsqueeze(0)

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return hook_fn

    def _hook_fn(self, module, input, output):
        """Add weighted Gospel direction vectors to residual stream."""
        hidden = output[0] if isinstance(output, tuple) else output

        # Compute combined steering vector
        steering = torch.zeros(self.vectors.shape[1], device=self.device,
                               dtype=hidden.dtype)
        for gi, gospel in enumerate(GOSPELS):
            alpha = self.alphas[gospel]
            if alpha != 0.0:
                steering += alpha * self.vectors[gi]

        # Add to all token positions
        hidden = hidden + steering.unsqueeze(0).unsqueeze(0)

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    def set_alpha(self, gospel: str, value: float):
        self.alphas[gospel] = value

    def set_all(self, value: float):
        for g in GOSPELS:
            self.alphas[g] = value

    def reset(self):
        self.set_all(0.0)

    def status(self) -> str:
        parts = []
        for g in GOSPELS:
            a = self.alphas[g]
            color = COLORS.get(g, "")
            if a != 0.0:
                parts.append(f"{color}{g}={a:+.1f}{RESET}")
            else:
                parts.append(f"{DIM}{g}=0{RESET}")
        return "  ".join(parts)

    def remove(self):
        for h in self.hooks:
            h.remove()


def generate_response(model, tokenizer, prompt: str,
                      max_new_tokens: int = 256,
                      temperature: float = 0.7) -> str:
    """Generate with steering active via hooks."""
    # Base model: use raw completion with a simple prompt format
    # Instruct model: use chat template
    is_base = "Base" in MODEL_ID or "base" in MODEL_ID
    if is_base:
        input_text = f"Question: {prompt}\nAnswer:"
    else:
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            top_p=0.9 if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Gospel Steering Chat")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Initial steering strength for all Gospels")
    parser.add_argument("--layer", type=int, default=None,
                        help="Override center steering layer (default: best)")
    parser.add_argument("--spread", type=int, default=3,
                        help="Number of layers on each side of center to steer "
                             "(default: 3, so 7 layers total)")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # Load vectors
    meta_path = VECTOR_DIR / "meta.json"
    if not meta_path.exists():
        print("ERROR: No vectors found. Run extract.py and compute_vectors.py first.")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    center_layer = args.layer if args.layer is not None else meta["best_layer"]
    num_layers = meta["num_layers"]

    # Build layer range: center ± spread, clamped to valid range
    layer_indices = list(range(
        max(0, center_layer - args.spread),
        min(num_layers, center_layer + args.spread + 1)
    ))

    all_vectors = torch.load(VECTOR_DIR / "gospel_vectors_all_layers.pt",
                             weights_only=True)

    print(f"\n{BOLD}{'═' * 72}{RESET}")
    print(f"{BOLD}  GOSPEL STEERING CHAT{RESET}")
    print(f"{BOLD}  Model: {MODEL_ID}{RESET}")
    print(f"{BOLD}  Steering layers: {layer_indices[0]}-{layer_indices[-1]} "
          f"(center={center_layer}, {len(layer_indices)} layers){RESET}")
    print(f"{BOLD}  Best layer accuracy: "
          f"{meta['layer_accuracies'][center_layer]:.3f}{RESET}")
    print(f"{BOLD}{'═' * 72}{RESET}\n")

    # Load model
    device = torch.device(f"cuda:{args.gpu}")
    kwargs = dict(
        dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    if USE_QUANTIZATION:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=False,
        )

    print("Loading model ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Set up multi-layer steering
    steerer = GospelSteerer(model, layer_indices, all_vectors, device)
    if args.alpha != 0.0:
        steerer.set_all(args.alpha)

    print(f"\n{BOLD}Model loaded. Steering: [{steerer.status()}]{RESET}")
    print(f"{DIM}Commands: /matthew 2.0, /mark -1.0, /reset, /status, /temp, /tokens{RESET}")
    print(f"{DIM}Type 'quit' or Ctrl+C to exit.{RESET}\n")

    temperature = 0.7
    max_tokens = 256

    while True:
        try:
            # Show steering state in prompt
            active = [f"{g[0].upper()}={steerer.alphas[g]:+.1f}"
                      for g in GOSPELS if steerer.alphas[g] != 0.0]
            prompt_suffix = f" [{', '.join(active)}]" if active else ""
            user_input = input(f"{BOLD}You{prompt_suffix} > {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}Goodbye.{RESET}")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        # ── Slash commands ───────────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd == "/reset":
                steerer.reset()
                print(f"{DIM}Steering reset to zero.{RESET}")
            elif cmd == "/status":
                print(f"  Steering: {steerer.status()}")
                print(f"  Layer: {layer_idx}, Temp: {temperature}, "
                      f"Max tokens: {max_tokens}")
            elif cmd == "/temp" and len(parts) == 2:
                temperature = float(parts[1])
                print(f"{DIM}Temperature: {temperature}{RESET}")
            elif cmd == "/tokens" and len(parts) == 2:
                max_tokens = int(parts[1])
                print(f"{DIM}Max tokens: {max_tokens}{RESET}")
            elif cmd == "/alpha" and len(parts) == 2:
                steerer.set_all(float(parts[1]))
                print(f"  All Gospels: {steerer.status()}")
            elif cmd[1:] in GOSPELS and len(parts) == 2:
                gospel = cmd[1:]
                steerer.set_alpha(gospel, float(parts[1]))
                print(f"  {steerer.status()}")
            else:
                print(f"{DIM}Unknown command. Try: /matthew 2.0, /reset, "
                      f"/status, /alpha 1.5{RESET}")
            continue

        # ── Generate ─────────────────────────────────────────────────────
        t0 = time.time()
        response = generate_response(model, tokenizer, user_input,
                                     max_tokens, temperature)
        elapsed = time.time() - t0

        # Color output based on dominant steering
        dominant = max(GOSPELS, key=lambda g: abs(steerer.alphas[g]))
        color = COLORS.get(dominant, "") if steerer.alphas[dominant] != 0.0 else ""

        wrapped = textwrap.fill(response, width=72)
        print(f"\n{color}{HRULE}{RESET}")
        print(f"{color}{wrapped}{RESET}")
        print(f"{BOLD}{color}{HRULE}{RESET}")
        print(f"{DIM}{elapsed:.1f}s | {steerer.status()}{RESET}\n")


if __name__ == "__main__":
    main()
