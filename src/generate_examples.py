"""
Generate side-by-side steering examples for the GospelVec paper.

Runs each prompt through multiple steering configurations and saves
the results as JSON for inclusion in the paper.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import GOSPELS, MODEL_ID, VECTOR_DIR

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


class GospelSteerer:
    """Multi-layer steering for example generation."""

    def __init__(self, model, layer_indices, all_vectors, device):
        self.layer_indices = layer_indices
        self.device = device
        self.layer_vectors = {}
        for li in layer_indices:
            self.layer_vectors[li] = all_vectors[li].to(device)
        self.alphas = {g: 0.0 for g in GOSPELS}
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
                if self.alphas[gospel] != 0.0:
                    steering += self.alphas[gospel] * vectors[gi]
            hidden = hidden + steering.unsqueeze(0).unsqueeze(0)
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return hook_fn

    def set_config(self, config: dict):
        self.alphas = {g: 0.0 for g in GOSPELS}
        for g, a in config.items():
            self.alphas[g] = a

    def reset(self):
        self.alphas = {g: 0.0 for g in GOSPELS}


def generate(model, tokenizer, prompt, max_new_tokens=300, temperature=0.7):
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
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ── Example definitions ──────────────────────────────────────────────────

EXAMPLES = [
    {
        "id": "narrative",
        "prompt": "Tell me a story about a dog",
        "description": "Creative fiction — tests how steering shifts narrative archetypes",
        "configs": [
            {"label": "Baseline (no steering)", "alphas": {}},
            {"label": "Matthew α=+3.0", "alphas": {"matthew": 3.0}},
            {"label": "John α=+3.0", "alphas": {"john": 3.0}},
            {"label": "John α=+5.0, Luke α=-3.0", "alphas": {"john": 5.0, "luke": -3.0}},
        ],
    },
    {
        "id": "theological",
        "prompt": "Who is Jesus and what did he come to do?",
        "description": "Direct theological question — tests Gospel-specific Christology",
        "configs": [
            {"label": "Baseline (no steering)", "alphas": {}},
            {"label": "Mark α=+4.0", "alphas": {"mark": 4.0}},
            {"label": "John α=+4.0", "alphas": {"john": 4.0}},
            {"label": "Luke α=+4.0", "alphas": {"luke": 4.0}},
        ],
    },
    {
        "id": "encouragement",
        "prompt": "Write a letter of encouragement to someone going through a difficult time",
        "description": "Personal/emotional — tests how Gospel perspectives shape pastoral voice",
        "configs": [
            {"label": "Baseline (no steering)", "alphas": {}},
            {"label": "Matthew α=+4.0", "alphas": {"matthew": 4.0}},
            {"label": "Mark α=+4.0", "alphas": {"mark": 4.0}},
            {"label": "Luke α=+4.0", "alphas": {"luke": 4.0}},
        ],
    },
    {
        "id": "community",
        "prompt": "Describe what a perfect community would look like",
        "description": "Worldbuilding — tests how Gospel perspectives shape social vision",
        "configs": [
            {"label": "Baseline (no steering)", "alphas": {}},
            {"label": "Matthew α=+4.0", "alphas": {"matthew": 4.0}},
            {"label": "Luke α=+4.0", "alphas": {"luke": 4.0}},
            {"label": "John α=+4.0", "alphas": {"john": 4.0}},
        ],
    },
]


def main():
    # Load meta
    with open(VECTOR_DIR / "meta.json") as f:
        meta = json.load(f)

    center = meta["best_layer"]
    spread = 3
    num_layers = meta["num_layers"]
    layer_indices = list(range(max(0, center - spread),
                               min(num_layers, center + spread + 1)))

    all_vectors = torch.load(VECTOR_DIR / "gospel_vectors_all_layers.pt",
                             weights_only=True)

    print(f"{BOLD}GospelVec Example Generator{RESET}")
    print(f"  Model: {MODEL_ID}")
    print(f"  Steering layers: {layer_indices[0]}-{layer_indices[-1]} "
          f"({len(layer_indices)} layers)")
    print()

    # Load model
    device = torch.device("cuda:0")
    print("Loading model ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    steerer = GospelSteerer(model, layer_indices, all_vectors, device)

    # Generate all examples
    results = []
    for ex in EXAMPLES:
        print(f"\n{'='*60}")
        print(f"Prompt: {ex['prompt']}")
        print(f"{'='*60}")

        ex_result = {
            "id": ex["id"],
            "prompt": ex["prompt"],
            "description": ex["description"],
            "responses": [],
        }

        for cfg in ex["configs"]:
            steerer.set_config(cfg["alphas"])
            label = cfg["label"]
            print(f"  [{label}] generating ...", end=" ", flush=True)

            t0 = time.time()
            response = generate(model, tokenizer, ex["prompt"])
            elapsed = time.time() - t0
            print(f"{elapsed:.1f}s")

            ex_result["responses"].append({
                "label": label,
                "alphas": cfg["alphas"],
                "response": response,
                "generation_time": elapsed,
            })

        results.append(ex_result)

    # Save
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "steering_examples.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": MODEL_ID,
            "steering_layers": layer_indices,
            "best_layer": center,
            "best_accuracy": meta["best_accuracy"],
            "gospel_geometry": meta,
            "examples": results,
        }, f, indent=2)

    print(f"\n{BOLD}Saved to {out_path}{RESET}")
    print(f"Total examples: {sum(len(e['responses']) for e in results)}")


if __name__ == "__main__":
    main()
