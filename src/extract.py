"""
Extract activation vectors from Gospel texts using Qwen 3.5 9B.

Runs each Gospel on a separate GPU for parallel extraction.
Adapted from emotion-vectors/src/extraction.py for TinyBox Green.

Usage:
  python3 src/extract.py                    # All gospels in parallel
  python3 src/extract.py --gospel matthew   # Single gospel (subprocess mode)
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    ACTIVATION_DIR, CHUNK_MAX_TOKENS, GOSPEL_DATA_DIR, GOSPELS,
    HIDDEN_DIM, MEAN_POOL_SKIP_TOKENS, MODEL_ID, NUM_LAYERS,
    NEUTRAL_TEXTS, USE_QUANTIZATION,
)


# ── Activation Hooks ─────────────────────────────────────────────────────

class ActivationCollector:
    """Registers hooks on all decoder layers to capture residual stream."""

    def __init__(self, model):
        self.activations = {}
        self.hooks = []
        # Hook into each decoder layer's output
        for i, layer in enumerate(model.model.layers):
            hook = layer.register_forward_hook(self._make_hook(i))
            self.hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # output is a tuple; first element is the hidden states
            hidden = output[0] if isinstance(output, tuple) else output
            self.activations[layer_idx] = hidden.detach()
        return hook_fn

    def clear(self):
        self.activations = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor,
              skip_tokens: int = MEAN_POOL_SKIP_TOKENS) -> torch.Tensor:
    """Mean-pool across token positions, skipping BOS/special tokens."""
    mask = attention_mask.clone().float()
    mask[:, :skip_tokens] = 0.0
    mask = mask.unsqueeze(-1)  # [batch, seq, 1]
    pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
    return pooled  # [batch, hidden_dim]


# ── Text Chunking ────────────────────────────────────────────────────────

def chunk_text(text: str, tokenizer, max_tokens: int = CHUNK_MAX_TOKENS) -> list[str]:
    """Split text into chunks of ~max_tokens on sentence boundaries."""
    sentences = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        # Split on sentence-ending punctuation
        import re
        for sent in re.split(r'(?<=[.!?])\s+', para):
            if sent.strip():
                sentences.append(sent.strip())

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = len(tokenizer.encode(sent, add_special_tokens=False))
        if current_tokens + sent_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_tokens = sent_tokens
        else:
            current_chunk.append(sent)
            current_tokens += sent_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ── Extraction ───────────────────────────────────────────────────────────

def load_model(gpu_id: int):
    """Load Qwen 3.5 9B on specified GPU."""
    device = torch.device(f"cuda:{gpu_id}")

    kwargs = dict(
        dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )

    if USE_QUANTIZATION:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=False,
        )

    print(f"  Loading {MODEL_ID} on GPU {gpu_id} "
          f"({'8-bit' if USE_QUANTIZATION else 'bf16'}) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    return model, tokenizer


def extract_activations(model, tokenizer, texts: list[str],
                        gpu_id: int) -> torch.Tensor:
    """Extract mean-pooled activations at all layers for a list of texts.

    Returns: [num_layers, num_texts, hidden_dim]
    """
    device = torch.device(f"cuda:{gpu_id}")
    collector = ActivationCollector(model)
    all_reps = []  # Will be [num_texts, num_layers, hidden_dim]

    for i, text in enumerate(texts):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"    Chunk {i+1}/{len(texts)} ...", flush=True)

        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=CHUNK_MAX_TOKENS + 32).to(device)

        collector.clear()
        with torch.no_grad():
            model(**inputs)

        # Mean-pool each layer's activations
        text_reps = []
        for layer_idx in range(len(collector.activations)):
            hidden = collector.activations[layer_idx]
            pooled = mean_pool(hidden, inputs["attention_mask"])
            text_reps.append(pooled.squeeze(0).cpu())  # [hidden_dim]

        all_reps.append(torch.stack(text_reps))  # [num_layers, hidden_dim]

    collector.remove_hooks()

    # Stack: [num_texts, num_layers, hidden_dim] → transpose → [num_layers, num_texts, hidden_dim]
    result = torch.stack(all_reps).permute(1, 0, 2)
    return result


def extract_single_gospel(gospel: str, gpu_id: int):
    """Extract activations for one Gospel + neutral texts."""
    print(f"\n{'='*60}")
    print(f"Extracting: {gospel.upper()} on GPU {gpu_id}")
    print(f"{'='*60}")

    # Load model
    model, tokenizer = load_model(gpu_id)

    # Load Gospel text
    raw_path = GOSPEL_DATA_DIR / f"{gospel}_raw.txt"
    if not raw_path.exists():
        raise FileNotFoundError(f"Gospel text not found: {raw_path}")
    text = raw_path.read_text()
    print(f"  Loaded {gospel}: {len(text.split())} words")

    # Chunk Gospel text
    chunks = chunk_text(text, tokenizer)
    print(f"  Chunked into {len(chunks)} segments (~{CHUNK_MAX_TOKENS} tokens each)")

    # Extract Gospel activations
    print(f"  Extracting Gospel activations ...")
    t0 = time.time()
    gospel_reps = extract_activations(model, tokenizer, chunks, gpu_id)
    print(f"  Gospel activations: {gospel_reps.shape} in {time.time()-t0:.1f}s")

    # Extract neutral activations (same for all gospels, but each GPU computes its own)
    print(f"  Extracting neutral activations ...")
    neutral_reps = extract_activations(model, tokenizer, NEUTRAL_TEXTS, gpu_id)
    print(f"  Neutral activations: {neutral_reps.shape}")

    # Save
    ACTIVATION_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(gospel_reps, ACTIVATION_DIR / f"{gospel}_reps.pt")
    torch.save(neutral_reps, ACTIVATION_DIR / f"neutral_reps_{gospel}.pt")

    # Save chunk metadata
    meta = {
        "gospel": gospel,
        "num_chunks": len(chunks),
        "chunk_texts": chunks,  # For interpretability later
    }
    with open(ACTIVATION_DIR / f"{gospel}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved to {ACTIVATION_DIR}/")

    # Cleanup
    del model
    torch.cuda.empty_cache()


def run_all_sequential(gpu_id: int = 0):
    """Run all 4 Gospels sequentially on one GPU (for DGX Spark)."""
    print(f"\nExtracting all Gospels on GPU {gpu_id} ...")

    # Load model once, reuse for all gospels
    model, tokenizer = load_model(gpu_id)

    for gospel in GOSPELS:
        out_path = ACTIVATION_DIR / f"{gospel}_reps.pt"
        if out_path.exists():
            print(f"  Skipping {gospel} (already extracted)")
            continue

        print(f"\n{'='*60}")
        print(f"Extracting: {gospel.upper()}")
        print(f"{'='*60}")

        # Load Gospel text
        raw_path = GOSPEL_DATA_DIR / f"{gospel}_raw.txt"
        if not raw_path.exists():
            raise FileNotFoundError(f"Gospel text not found: {raw_path}")
        text = raw_path.read_text()
        print(f"  Loaded {gospel}: {len(text.split())} words")

        # Chunk
        chunks = chunk_text(text, tokenizer)
        print(f"  Chunked into {len(chunks)} segments")

        # Extract
        t0 = time.time()
        gospel_reps = extract_activations(model, tokenizer, chunks, gpu_id)
        print(f"  Gospel activations: {gospel_reps.shape} in {time.time()-t0:.1f}s")

        # Save
        ACTIVATION_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(gospel_reps, ACTIVATION_DIR / f"{gospel}_reps.pt")

        meta = {"gospel": gospel, "num_chunks": len(chunks), "chunk_texts": chunks}
        with open(ACTIVATION_DIR / f"{gospel}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    # Neutral activations (once)
    neutral_path = ACTIVATION_DIR / f"neutral_reps_{GOSPELS[0]}.pt"
    if not neutral_path.exists():
        print(f"\nExtracting neutral activations ...")
        neutral_reps = extract_activations(model, tokenizer, NEUTRAL_TEXTS, gpu_id)
        torch.save(neutral_reps, neutral_path)
        print(f"  Neutral: {neutral_reps.shape}")
    else:
        print("  Neutral activations already extracted.")

    del model
    torch.cuda.empty_cache()
    print("\nAll extractions complete.")


def main():
    parser = argparse.ArgumentParser(description="Extract Gospel activations")
    parser.add_argument("--gospel", type=str, choices=GOSPELS,
                        help="Single gospel (subprocess mode)")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if args.gospel:
        extract_single_gospel(args.gospel, args.gpu)
    else:
        run_all_sequential(args.gpu)


if __name__ == "__main__":
    main()
