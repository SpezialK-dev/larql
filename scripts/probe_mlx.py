#!/usr/bin/env python3
"""Probe entities with MLX — full forward pass with real attention.

Runs prompted templates through the model via MLX, captures the residual
stream at each knowledge layer, projects through gate vectors to find
which features ACTUALLY fire, and matches against Wikidata triples.

This is ground truth: real attention, real residuals, real gate activations.

Usage:
    pip install mlx mlx-lm
    python3 scripts/probe_mlx.py

Requires:
    - MLX installed (Apple Silicon)
    - Model: google/gemma-3-4b-it (downloaded by mlx-lm)
    - Vindex: output/gemma3-4b-full.vindex (for gate vectors + down_meta)
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load as mlx_load
except ImportError as e:
    print(f"Install MLX: pip install mlx mlx-lm ({e})", file=sys.stderr)
    sys.exit(1)


TEMPLATES = {
    "capital": "The capital of {X} is",
    "language": "The official language of {X} is",
    "continent": "{X} is a country in",
    "borders": "{X} shares a border with",
    "occupation": "{X} was a",
    "birthplace": "{X} was born in",
    "currency": "The currency of {X} is",
    "located in": "{X} is located in",
    "author": "{X} was written by",
    "director": "{X} was directed by",
    "genre": "The genre of {X} is",
    "founder": "{X} was founded by",
    "nationality": "{X} has the nationality of",
}

VINDEX = "output/gemma3-4b-full.vindex"
MODEL_ID = "google/gemma-3-4b-it"


def load_vindex_gates_and_meta(vindex_dir):
    """Load gate vectors and down_meta from vindex."""
    vindex_dir = Path(vindex_dir)

    with open(vindex_dir / "index.json") as f:
        config = json.load(f)

    hidden_size = config["hidden_size"]
    gate_raw = np.fromfile(vindex_dir / "gate_vectors.bin", dtype=np.float32)
    gates = {}
    for layer_info in config["layers"]:
        layer = layer_info["layer"]
        nf = layer_info["num_features"]
        offset = layer_info["offset"] // 4
        gates[layer] = gate_raw[offset:offset + nf * hidden_size].reshape(nf, hidden_size)

    down_meta = {}
    with open(vindex_dir / "down_meta.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            down_meta[(obj.get("l", 0), obj.get("f", 0))] = obj.get("t", "")

    return config, gates, down_meta


def get_residuals_mlx(model, tokenizer, prompt, num_layers):
    """Run a forward pass and capture the residual stream at each layer."""
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    # Hook into the model to capture residuals
    residuals = {}

    # Run the model's embedding
    h = model.model.embed_tokens(input_ids)

    # Apply embed scale if Gemma
    if hasattr(model.model, 'embed_scale'):
        h = h * model.model.embed_scale
    elif hasattr(model.config, 'hidden_size'):
        # Gemma scales by sqrt(hidden_size)
        import math
        h = h * math.sqrt(model.config.hidden_size)

    # Run through each layer, capturing residuals
    cache = None
    mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
    mask = mask.astype(h.dtype)

    for i, layer in enumerate(model.model.layers):
        h = layer(h, mask=mask, cache=cache)
        # Capture the residual after each layer (last token position)
        residuals[i] = np.array(h[0, -1, :].astype(mx.float32))

    return residuals


def get_residuals_simple(model, tokenizer, prompt):
    """Run forward pass through Gemma3 MLX model, capture per-layer residuals."""
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    try:
        # Gemma3 MLX: model has .layers directly, and embedding via __call__
        # We need to replicate the forward pass manually to capture residuals

        # Gemma3 MLX structure: model['language_model']['model'] has embed_tokens, layers, norm
        inner = model['language_model']['model']
        embed_fn = inner.embed_tokens
        layers = inner.layers

        h = embed_fn(input_ids)

        # Gemma scaling: multiply by sqrt(hidden_size)
        import math
        hidden_size = h.shape[-1]
        h = h * math.sqrt(hidden_size)

        # Create causal mask
        seq_len = h.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        residuals = {}
        for i, layer in enumerate(layers):
            h = layer(h, mask=mask)
            # Capture last token residual (convert bf16→f32 for numpy/gate projection)
            mx.eval(h)
            residuals[i] = np.array(h[0, -1, :].astype(mx.float32))

        return residuals
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("Loading vindex gates and metadata...")
    config, gates, down_meta = load_vindex_gates_and_meta(VINDEX)
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]
    print(f"  {num_layers} layers, {hidden_size} hidden, {len(down_meta)} features")

    print("Loading triples...")
    with open("data/wikidata_triples.json") as f:
        triples = json.load(f)

    pair_to_relation = {}
    for rel_name, rel_data in triples.items():
        for pair in rel_data.get("pairs", []):
            if len(pair) >= 2:
                pair_to_relation[(pair[0].lower(), pair[1].lower())] = rel_name

    print(f"Loading MLX model: {MODEL_ID}...")
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"  # Use cached model, don't hit HF
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model, tokenizer = mlx_load(MODEL_ID)
    print("  Model loaded")

    # Quick test
    print("\nTest: 'The capital of France is'")
    residuals = get_residuals_simple(model, tokenizer, "The capital of France is")
    if residuals is None:
        print("ERROR: Could not capture residuals from MLX model")
        sys.exit(1)

    print(f"  Captured residuals at {len(residuals)} layers")

    # Check L27 gate activations
    if 27 in residuals and 27 in gates:
        r = residuals[27]
        scores = gates[27] @ r
        top5 = np.argsort(-np.abs(scores))[:5]
        print(f"  L27 top features:")
        for idx in top5:
            target = down_meta.get((27, int(idx)), "?")
            print(f"    F{idx} gate={scores[idx]:+.1f} → {target}")

    # Full probe
    print(f"\nProbing with {len(TEMPLATES)} templates...")
    feature_labels = {}
    relation_counts = defaultdict(int)
    total_probes = 0
    start_time = time.time()

    for rel_name, template in TEMPLATES.items():
        if rel_name not in triples:
            continue

        # Use ALL subjects, prioritizing short/well-known names
        all_subjects = list(set(
            pair[0] for pair in triples[rel_name].get("pairs", [])
            if len(pair) >= 2 and 2 <= len(pair[0]) <= 30
        ))
        # Sort: single-word short names first, then multi-word
        all_subjects.sort(key=lambda s: (len(s.split()), len(s)))
        subjects = all_subjects

        if not subjects:
            continue

        matched = 0
        rel_start = time.time()
        for si, subject in enumerate(subjects):
            prompt = template.replace("{X}", subject)
            residuals = get_residuals_simple(model, tokenizer, prompt)
            if residuals is None:
                continue

            total_probes += 1
            if (si + 1) % 50 == 0:
                elapsed_rel = time.time() - rel_start
                rate = (si + 1) / elapsed_rel if elapsed_rel > 0 else 0
                eta = (len(subjects) - si - 1) / rate if rate > 0 else 0
                sys.stdout.write(f"\r  {rel_name:<20s} {si+1}/{len(subjects)} ({matched} labels, {rate:.0f}/s, ETA {eta:.0f}s)")
                sys.stdout.flush()

            for layer in range(14, 28):
                if layer not in residuals or layer not in gates:
                    continue
                r = residuals[layer]
                scores = gates[layer] @ r
                top_indices = np.argsort(-np.abs(scores))[:50]

                for feat_idx in top_indices:
                    score = float(scores[feat_idx])
                    if abs(score) < 5.0:
                        continue
                    target = down_meta.get((layer, int(feat_idx)), "")
                    if len(target) < 2:
                        continue
                    key = (subject.lower(), target.lower())
                    if key in pair_to_relation and pair_to_relation[key] == rel_name:
                        feat_key = f"L{layer}_F{feat_idx}"
                        if feat_key not in feature_labels:
                            feature_labels[feat_key] = rel_name
                            relation_counts[rel_name] += 1
                            matched += 1

        elapsed = time.time() - start_time
        rate = total_probes / elapsed if elapsed > 0 else 0
        print(f"  {rel_name:<20s} {len(subjects):3d} entities → {matched:3d} features  ({rate:.1f} probes/s)")

    elapsed = time.time() - start_time
    print(f"\nTotal: {total_probes} probes in {elapsed:.0f}s ({total_probes/elapsed:.1f}/s)")
    print(f"Labeled {len(feature_labels)} features")

    if relation_counts:
        print(f"\nRelation distribution:")
        for rel, count in sorted(relation_counts.items(), key=lambda x: -x[1]):
            print(f"  {rel:<25s} {count:4d}")

    # Merge with existing
    existing_path = Path(VINDEX) / "feature_labels.json"
    existing = {}
    if existing_path.exists():
        with open(existing_path) as f:
            existing = json.load(f)

    new_count = 0
    for key, rel in feature_labels.items():
        if key not in existing:
            existing[key] = rel
            new_count += 1

    with open(existing_path, "w") as f:
        json.dump(existing, f, indent=2)

    data_path = Path("data/feature_labels.json")
    with open(data_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\nMerged: {new_count} new + {len(existing) - new_count} existing = {len(existing)} total")
    print(f"Saved to {existing_path}")


if __name__ == "__main__":
    main()
