import json
import urllib.request
import jax.numpy as jnp
from typing import Tuple, List

def load_arc_task(task_url: str = "https://raw.githubusercontent.com/fchollet/ARC/master/data/training/007bbfb7.json") -> Tuple[List, List]:
    """
    Downloads an authentic ARC task JSON dynamically for Test-Time execution.
    Defaults to 007bbfb7 (a simple grid projection pattern) for 'First Light'.
    """
    print(f"Downloading ARC task from: {task_url}")
    req = urllib.request.urlopen(task_url)
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
