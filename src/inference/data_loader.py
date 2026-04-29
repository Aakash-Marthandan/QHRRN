import json
import urllib.request
import urllib.error
import jax.numpy as jnp
from typing import Tuple, List

def load_arc_task(task_url: str = "https://raw.githubusercontent.com/fchollet/ARC/master/data/training/007bbfb7.json") -> Tuple[List, List]:
    """
    Downloads an authentic ARC task JSON dynamically for Test-Time execution.
    Defaults to 007bbfb7 (a simple grid projection pattern) for 'First Light'.

    If the task is not found in data/training/, automatically retries
    against data/evaluation/ before raising an error.
    """
    print(f"Downloading ARC task from: {task_url}")
    try:
        req = urllib.request.urlopen(task_url)
    except urllib.error.HTTPError as e:
        if e.code == 404 and '/data/training/' in task_url:
            fallback_url = task_url.replace('/data/training/', '/data/evaluation/')
            print(f"  404 on training/ — retrying: {fallback_url}")
            req = urllib.request.urlopen(fallback_url)
        else:
            raise
    data = json.loads(req.read().decode('utf-8'))
    
    train_pairs = []
    for pair in data['train']:
        # For MERA abstraction, we represent the 10 categorical indices.
        # Input/outputs are initially natively 2D maps. We expand dims to simulate
        # the required `chi` dimension functionally initialized as 1-hot or identity scale.
        
        # We will utilize a standard sparse representation for the pipeline evaluation
        x = jnp.array(pair['input'], dtype=jnp.int32)
        y = jnp.array(pair['output'], dtype=jnp.int32)
        train_pairs.append((x, y))
        
    test_pairs = []
    for pair in data['test']:
        x = jnp.array(pair['input'], dtype=jnp.int32)
        # The test cases in ARC json occasionally omit outputs for strict evaluations;
        # if evaluating locally, public sets contain 'output'.
        y = jnp.array(pair.get('output', jnp.zeros_like(x)), dtype=jnp.int32)
        test_pairs.append((x, y))
        
    return train_pairs, test_pairs


# ── Task Discovery Utilities ──────────────────────────────────────────────────

ARC_GITHUB_API = "https://api.github.com/repos/fchollet/ARC-AGI/contents/data"

def fetch_valid_task_ids(split: str = "evaluation") -> List[str]:
    """
    Fetches the official index of valid ARC task IDs directly from the
    GitHub repository tree.  No local files or hardcoded lists required.

    Args:
        split: 'training' or 'evaluation' (default 'evaluation').

    Returns:
        Sorted list of 8-character hex task IDs (e.g. ['00576224', ...]).
    """
    api_url = f"{ARC_GITHUB_API}/{split}"
    print(f"Fetching valid {split} task IDs from: {api_url}")

    req = urllib.request.Request(api_url)
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("User-Agent", "QHRRN-DataLoader/1.0")

    resp = urllib.request.urlopen(req)
    entries = json.loads(resp.read().decode('utf-8'))

    task_ids = []
    for entry in entries:
        name = entry.get("name", "")
        if name.endswith(".json"):
            task_ids.append(name.replace(".json", ""))

    task_ids.sort()
    print(f"  Found {len(task_ids)} valid {split} tasks.")
    return task_ids


def generate_eval_task_file(
    output_path: str = "eval_tasks.txt",
    count: int = 10,
    exclude_file: str = "pretrain_tasks.txt",
    split: str = "evaluation",
    seed: int = 42,
) -> List[str]:
    """
    Generates a guaranteed-valid eval_tasks.txt by:
      1. Fetching the full official task index from GitHub.
      2. Excluding any IDs found in the pretrain task file.
      3. Deterministically sampling `count` non-overlapping tasks.

    Args:
        output_path: Where to write the output file.
        count: Number of evaluation tasks to select.
        exclude_file: Path to pretrain_tasks.txt (IDs to exclude).
        split: ARC split to draw from ('training' or 'evaluation').
        seed: Random seed for reproducible selection.

    Returns:
        List of selected task IDs.
    """
    import random

    # 1. Fetch valid IDs from GitHub
    valid_ids = fetch_valid_task_ids(split=split)

    # 2. Load exclusion set from pretrain file
    exclude_ids = set()
    try:
        with open(exclude_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    exclude_ids.add(line)
        print(f"  Excluding {len(exclude_ids)} pretrain IDs from {exclude_file}")
    except FileNotFoundError:
        print(f"  No exclusion file found at {exclude_file} — skipping.")

    # 3. Filter and sample
    candidates = [tid for tid in valid_ids if tid not in exclude_ids]
    print(f"  {len(candidates)} candidates after exclusion.")

    rng = random.Random(seed)
    selected = rng.sample(candidates, min(count, len(candidates)))
    selected.sort()

    # 4. Write output
    with open(output_path, 'w') as f:
        for tid in selected:
            f.write(tid + '\n')

    print(f"  Wrote {len(selected)} tasks to {output_path}")
    return selected
