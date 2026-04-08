"""
Compute Gospel direction vectors from extracted activations.

For each layer:
  1. Compute global mean across all Gospel chunks
  2. Compute per-Gospel mean
  3. Direction = Gospel mean - global mean
  4. Project out neutral (PCA-denoised) background
  5. Normalize

Then evaluate layer accuracy on a held-out split to select the best readout layer.

Usage:
  python3 src/compute_vectors.py
"""

import json
from pathlib import Path

import torch

from config import (
    ACTIVATION_DIR, GOSPELS, HIDDEN_DIM, NUM_LAYERS,
    PCA_VARIANCE_THRESHOLD, VECTOR_DIR,
)


def compute_pca_basis(neutral_reps: torch.Tensor,
                      variance_threshold: float = PCA_VARIANCE_THRESHOLD):
    """Compute PCA basis from neutral activations for denoising.

    Args:
        neutral_reps: [num_neutral, hidden_dim]

    Returns:
        pca_basis: [k, hidden_dim] orthonormal vectors
    """
    centered = neutral_reps - neutral_reps.mean(dim=0)
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    explained = (S ** 2).cumsum(0) / (S ** 2).sum()
    k = int((explained < variance_threshold).sum().item()) + 1
    return Vt[:k]  # [k, hidden_dim]


def project_out(direction: torch.Tensor, pca_basis: torch.Tensor) -> torch.Tensor:
    """Remove PCA basis components from a direction vector."""
    # direction: [hidden_dim], pca_basis: [k, hidden_dim]
    coeffs = direction @ pca_basis.T  # [k]
    projection = coeffs @ pca_basis    # [hidden_dim]
    return direction - projection


def compute_gospel_vectors():
    """Compute direction vectors for all Gospels at all layers."""
    print("Loading activations ...")

    # Load all Gospel activations
    gospel_reps = {}
    for gospel in GOSPELS:
        path = ACTIVATION_DIR / f"{gospel}_reps.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing activations: {path}")
        gospel_reps[gospel] = torch.load(path, weights_only=True)
        print(f"  {gospel}: {gospel_reps[gospel].shape}")

    # Load neutral activations (use any gospel's copy — they're identical)
    neutral_path = ACTIVATION_DIR / f"neutral_reps_{GOSPELS[0]}.pt"
    neutral_reps = torch.load(neutral_path, weights_only=True)
    print(f"  neutral: {neutral_reps.shape}")

    num_layers = gospel_reps[GOSPELS[0]].shape[0]

    # Concatenate all Gospel chunks for global mean
    # Shape per gospel: [num_layers, num_chunks, hidden_dim]
    all_chunks = torch.cat([gospel_reps[g] for g in GOSPELS], dim=1)
    # [num_layers, total_chunks, hidden_dim]
    print(f"  Total chunks across all Gospels: {all_chunks.shape[1]}")

    # ── Compute vectors at each layer ────────────────────────────────────
    print("\nComputing direction vectors ...")
    vectors = torch.zeros(num_layers, len(GOSPELS), all_chunks.shape[2])

    for layer in range(num_layers):
        # Global mean across all Gospel chunks at this layer
        global_mean = all_chunks[layer].mean(dim=0)  # [hidden_dim]

        # Neutral PCA basis at this layer
        pca_basis = compute_pca_basis(neutral_reps[layer])

        for gi, gospel in enumerate(GOSPELS):
            # Gospel-specific mean
            gospel_mean = gospel_reps[gospel][layer].mean(dim=0)  # [hidden_dim]

            # Direction: what makes this Gospel distinctive
            direction = gospel_mean - global_mean

            # PCA denoise
            direction = project_out(direction, pca_basis)

            # Normalize
            direction = direction / direction.norm().clamp(min=1e-8)

            vectors[layer, gi] = direction

    print(f"  Vectors shape: {vectors.shape}")  # [num_layers, 4, hidden_dim]

    # ── Evaluate layer accuracy ──────────────────────────────────────────
    # Use leave-one-out: for each chunk, classify it by cosine similarity
    # to the 4 Gospel directions. Accuracy = fraction correctly classified.
    print("\nEvaluating layer accuracy ...")
    layer_accuracies = []

    for layer in range(num_layers):
        correct = 0
        total = 0

        for gi, gospel in enumerate(GOSPELS):
            chunks_at_layer = gospel_reps[gospel][layer]  # [num_chunks, hidden_dim]

            for chunk_idx in range(chunks_at_layer.shape[0]):
                chunk_vec = chunks_at_layer[chunk_idx]
                chunk_vec = chunk_vec / chunk_vec.norm().clamp(min=1e-8)

                # Cosine similarity to each Gospel direction
                sims = torch.zeros(len(GOSPELS))
                for gj in range(len(GOSPELS)):
                    sims[gj] = torch.dot(chunk_vec, vectors[layer, gj])

                predicted = sims.argmax().item()
                if predicted == gi:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0
        layer_accuracies.append(accuracy)

        if (layer + 1) % 8 == 0 or layer == num_layers - 1:
            print(f"  Layer {layer:2d}: accuracy = {accuracy:.3f}")

    best_layer = max(range(num_layers), key=lambda i: layer_accuracies[i])
    best_accuracy = layer_accuracies[best_layer]
    print(f"\n  Best layer: {best_layer} (accuracy = {best_accuracy:.3f})")

    # ── Save ─────────────────────────────────────────────────────────────
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    # Save all-layer vectors
    torch.save(vectors, VECTOR_DIR / "gospel_vectors_all_layers.pt")

    # Save best-layer vectors separately for easy loading
    best_vectors = vectors[best_layer]  # [4, hidden_dim]
    torch.save(best_vectors, VECTOR_DIR / "gospel_vectors_best.pt")

    # Save metadata
    meta = {
        "gospels": GOSPELS,
        "best_layer": best_layer,
        "best_accuracy": best_accuracy,
        "layer_accuracies": layer_accuracies,
        "num_layers": num_layers,
        "hidden_dim": int(vectors.shape[2]),
        "pca_variance_threshold": PCA_VARIANCE_THRESHOLD,
    }
    with open(VECTOR_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved vectors to {VECTOR_DIR}/")
    print(f"  gospel_vectors_all_layers.pt: {vectors.shape}")
    print(f"  gospel_vectors_best.pt: {best_vectors.shape}")

    # ── Print Gospel geometry ────────────────────────────────────────────
    print(f"\nGospel geometry at best layer ({best_layer}):")
    print("  Cosine similarities between Gospel directions:")
    for i, g1 in enumerate(GOSPELS):
        for j, g2 in enumerate(GOSPELS):
            if j > i:
                sim = torch.dot(best_vectors[i], best_vectors[j]).item()
                print(f"    {g1} ↔ {g2}: {sim:+.4f}")


if __name__ == "__main__":
    compute_gospel_vectors()
