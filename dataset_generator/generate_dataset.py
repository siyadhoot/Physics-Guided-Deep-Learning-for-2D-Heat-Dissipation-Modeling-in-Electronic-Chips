"""
generate_dataset.py
====================
Step 2 of the execution mandate: generate the full dataset (default 8000
samples: 6400 train / 800 val / 800 test) using disjoint per-split random
seed blocks (train/val/test never share seeds -> no duplicate scenarios,
and test draws from an entirely separate seed range for OOD evaluation).

Resumable: progress + valid samples are checkpointed to disk in shards
after every BATCH_SIZE valid samples. If the process is interrupted, simply
re-run this script and it will continue from the last completed shard.
"""

import os
import sys
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import config as C
from sample_gen import generate_sample

FIELDS = ["Q", "material", "kx", "ky", "ax", "ay", "theta", "T"]


def progress_path(split):
    return os.path.join(C.CHECKPOINT_DIR, f"{split}_progress.json")


def shard_path(split, shard_idx):
    return os.path.join(C.CHECKPOINT_DIR, f"{split}_shard_{shard_idx:04d}.npz")


def shard_meta_path(split, shard_idx):
    return os.path.join(C.CHECKPOINT_DIR, f"{split}_shard_{shard_idx:04d}_meta.json")


def load_progress(split):
    p = progress_path(split)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"next_attempt": 0, "valid_count": 0, "n_shards": 0, "rejections": {}}


def save_progress(split, prog):
    tmp = progress_path(split) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(prog, f, indent=2)
    os.replace(tmp, progress_path(split))


def save_shard(split, shard_idx, samples, metas):
    arrs = {f: np.stack([s[f] for s in samples]) for f in FIELDS}
    np.savez_compressed(shard_path(split, shard_idx), **arrs)
    with open(shard_meta_path(split, shard_idx), "w") as f:
        json.dump(metas, f)


def generate_split(split, target, workers=None):
    base_seed = C.SEED_BLOCK[split]
    prog = load_progress(split)

    if prog["valid_count"] >= target:
        print(f"[{split}] already complete: {prog['valid_count']}/{target} (skipping)")
        return

    print(f"[{split}] resuming: attempt={prog['next_attempt']} "
          f"valid={prog['valid_count']}/{target} shards={prog['n_shards']}")

    attempt = prog["next_attempt"]
    valid_count = prog["valid_count"]
    shard_idx = prog["n_shards"]
    rejections = prog.get("rejections", {})
    max_attempts = target * C.MAX_ATTEMPT_MULTIPLIER

    pending_samples = []
    pending_metas = []
    t0 = time.time()
    chunk = 300
    workers = workers or max(1, os.cpu_count() - 1)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        while valid_count < target and attempt < max_attempts:
            n_chunk = min(chunk, max_attempts - attempt)
            seeds = list(range(base_seed + attempt, base_seed + attempt + n_chunk))
            ids = list(range(attempt, attempt + n_chunk))
            splits = [split] * n_chunk

            for r, log in ex.map(generate_sample, seeds, ids, splits, chunksize=10):
                attempt += 1
                if r is not None and valid_count < target:
                    r["meta"]["sample_id"] = valid_count
                    r["meta"]["seed"] = int(log["seed"])
                    pending_samples.append(r["sample"])
                    pending_metas.append(r["meta"])
                    valid_count += 1
                elif r is None:
                    reason = log["reason"]
                    rejections[reason] = rejections.get(reason, 0) + 1

                if len(pending_samples) >= C.BATCH_SIZE or (
                    valid_count >= target and pending_samples
                ):
                    save_shard(split, shard_idx, pending_samples, pending_metas)
                    shard_idx += 1
                    pending_samples, pending_metas = [], []
                    prog = {
                        "next_attempt": attempt, "valid_count": valid_count,
                        "n_shards": shard_idx, "rejections": rejections,
                    }
                    save_progress(split, prog)
                    elapsed = time.time() - t0
                    rate = valid_count / max(elapsed, 1e-6)
                    eta = (target - valid_count) / max(rate, 1e-6)
                    print(f"[{split}] valid={valid_count}/{target} attempts={attempt} "
                          f"elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={eta:.1f}s")

                if valid_count >= target:
                    break

    if pending_samples:
        save_shard(split, shard_idx, pending_samples, pending_metas)
        shard_idx += 1
        prog = {"next_attempt": attempt, "valid_count": valid_count,
                "n_shards": shard_idx, "rejections": rejections}
        save_progress(split, prog)

    total_elapsed = time.time() - t0
    print(f"[{split}] DONE: {valid_count}/{target} valid, {attempt} attempts, "
          f"{total_elapsed:.1f}s total. Rejections: {rejections}")

    if valid_count < target:
        print(f"[{split}] WARNING: only reached {valid_count}/{target} "
              f"within max_attempts={max_attempts}")


def main():
    print("=" * 70)
    print(f"FULL DATASET GENERATION  |  TOTAL_SAMPLES={C.TOTAL_SAMPLES}  "
          f"GRID={C.GRID_SIZE}x{C.GRID_SIZE}")
    print(f"  train={C.N_TRAIN}  val={C.N_VAL}  test={C.N_TEST}")
    print("=" * 70)

    generate_split("train", C.N_TRAIN)
    generate_split("val", C.N_VAL)
    generate_split("test", C.N_TEST)

    print("\nAll splits generated (or resumed to completion).")


if __name__ == "__main__":
    main()
