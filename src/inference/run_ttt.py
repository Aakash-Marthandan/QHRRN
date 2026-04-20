import jax
import jax.numpy as jnp
import numpy as np
import optax
import math
import argparse
import pickle
import os

from src.inference.data_loader import load_arc_task
from src.inference.visualize import print_side_by_side
from src.mera.cayley import cayley_transform
from src.mera.hypernetwork import init_hyper_params, modulate_weights
from src.tensor.peps import init_peps_params, embed_grid_to_peps, decode_from_peps
from src.ttt.loss import holographic_entropy_proxy
from src.ttt.langevin import langevin_dynamics_optimizer
from src.ttt.constraints import enforce_parameter_constraints

# ── Constants ──────────────────────────────────────────────────────────────────
PHYSICAL_DIM = 10       # ARC categorical colors — universally fixed
CHI = 4                 # Bond dimension — strict information bottleneck
RANK = 2                # Low-rank delta factorization rank
MAX_GRID_SIZE = 32      # Covers all ARC tasks (max 30×30). XLA compiles exactly once.
MAX_TTT_STEPS = 1000    # Maximum TTT steps per task
NAN_STRIKE_LIMIT = 10   # Abort task after this many consecutive NaN events

# Learning rates: pre-training uses a lower LR for broader generalization
LR_PRETRAIN = 0.01
LR_TTT = 0.05

DEFAULT_TASK_URL = (
    "https://raw.githubusercontent.com/fchollet/ARC/master/data/training/007bbfb7.json"
)
ARC_BASE_URL = (
    "https://raw.githubusercontent.com/fchollet/ARC/master/data/training/"
)

# ── Checkpoint I/O ─────────────────────────────────────────────────────────

def save_checkpoint(params, path):
    """
    Saves the JAX parameter pytree to disk via pickle.
    All leaves are converted to numpy arrays for portability —
    no JAX dependency needed to read the file.
    """
    np_params = jax.tree.map(lambda x: np.array(x), params)
    with open(path, 'wb') as f:
        pickle.dump(np_params, f)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Checkpoint saved: {path} ({size_kb:.1f} KB)")


def load_checkpoint(path):
    """
    Loads a checkpoint and converts all numpy arrays back to JAX arrays.
    """
    with open(path, 'rb') as f:
        np_params = pickle.load(f)
    params = jax.tree.map(lambda x: jnp.array(x), np_params)
    print(f"  Checkpoint loaded: {path}")
    return params

# ── Parameter helpers ──────────────────────────────────────────────────────────

def init_all_params(key, grid_size, chi):
    """
    Initialize the complete learnable parameter set:
      - PEPS embedding (5-index site tensor + boundary projector + decoder)
      - Modulation Hypernetwork (shared base + tiny MLP for scale-dependent delta)
    """
    k_peps, k_hyper = jax.random.split(key)

    peps_params = init_peps_params(k_peps, physical_dim=PHYSICAL_DIM, chi=chi)

    block_dim = 4 * chi
    hyper_params = init_hyper_params(k_hyper, block_dim=block_dim, rank=RANK)

    return {
        'peps': peps_params,
        'hyper': hyper_params,
    }

# ── Pure-functional MERA passes (hypernetwork-modulated) ───────────────────

def functional_forward(params, grid_size, chi):
    """
    PEPS → MERA encode: (H, W, 10) → embed → (H, W, chi) → (1, 1, chi).
    """
    def _encode(embedded_input):
        state = embedded_input
        current_size = grid_size
        num_layers = int(math.log2(grid_size))
        block_dim = 4 * chi
        layer_idx = 0

        while current_size > 1:
            H, W, C = state.shape
            state_blocks = state.reshape(H // 2, 2, W // 2, 2, C).transpose(0, 2, 1, 3, 4)
            state_flat = state_blocks.reshape(H // 2, W // 2, 4 * C)

            d_weights, i_weights = modulate_weights(
                params['hyper'], layer_idx, num_layers, block_dim, rank=RANK
            )

            U = cayley_transform(d_weights)
            disentangled = jnp.dot(state_flat, U.T)

            W_full = cayley_transform(i_weights)
            state = jnp.dot(disentangled, W_full[:, :chi])

            current_size //= 2
            layer_idx += 1

        return state
    return _encode


def functional_reverse(params, grid_size, chi):
    """
    MERA decode → PEPS: (1, 1, chi) → (H, W, chi) → decode → (H, W, 10).
    """
    def _decode(latent):
        state = latent
        num_layers = int(math.log2(grid_size))
        block_dim = 4 * chi

        for layer_idx in reversed(range(num_layers)):
            d_weights, i_weights = modulate_weights(
                params['hyper'], layer_idx, num_layers, block_dim, rank=RANK
            )

            W_full = cayley_transform(i_weights)
            expanded = jnp.dot(state, W_full[:, :chi].T)

            U = cayley_transform(d_weights)
            entangled = jnp.dot(expanded, U)

            H, W, C_flat = entangled.shape
            C = C_flat // 4
            blocks = entangled.reshape(H, W, 2, 2, C)
            state = blocks.transpose(0, 2, 1, 3, 4).reshape(H * 2, W * 2, C)

        return state
    return _decode

# ── Loss function (jointly across ALL support pairs) ───────────────────────

def ttt_loss_fn(params, x_batch, y_batch, grid_size, chi, lambda_entropy=0.01):
    """
    Holographic loss averaged across every demonstration pair.
    Uses jax.vmap so the invariant rule emerges from the full support set.
    """
    encode = functional_forward(params, grid_size, chi)
    decode = functional_reverse(params, grid_size, chi)

    def single_pair_loss(x_onehot, y_onehot):
        embedded = embed_grid_to_peps(x_onehot, params['peps'], chi)
        latent = encode(embedded)
        predicted = decode(latent)
        logits = decode_from_peps(predicted, params['peps'])
        probs = jax.nn.softmax(logits, axis=-1)
        return -jnp.mean(jnp.sum(y_onehot * jnp.log(probs + 1e-8), axis=-1))

    ce_losses = jax.vmap(single_pair_loss)(x_batch, y_batch)
    loss_ce = jnp.mean(ce_losses)

    learnable = {'peps': params['peps'], 'hyper': params['hyper']}
    loss_entropy = holographic_entropy_proxy(learnable)

    return loss_ce + lambda_entropy * loss_entropy, (loss_ce, loss_entropy)

# ── JIT-compiled update (optimizer created ONCE, closed over) ──────────────

def make_update_step(optimizer):
    """
    Returns a JIT-compiled training step that closes over a single
    optimizer instance — never re-instantiated inside the traced function.
    """
    def _step(params, opt_state, x_batch, y_batch, grid_size, chi):
        (loss_val, aux), grads = jax.value_and_grad(ttt_loss_fn, has_aux=True)(
            params, x_batch, y_batch, grid_size, chi
        )
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val, aux

    return jax.jit(_step, static_argnames=['grid_size', 'chi'])

# ── Helper: load and pad a task ────────────────────────────────────────────

def _load_and_pad_task(task_url, grid_size):
    """
    Downloads a task, pads all support pairs to grid_size × grid_size,
    and returns (x_batch, y_batch, train_pairs, test_pairs).
    """
    train_pairs, test_pairs = load_arc_task(task_url)

    # Pre-flight dimension guard
    all_dims = []
    for x, y in train_pairs + test_pairs:
        all_dims.extend([x.shape[0], x.shape[1], y.shape[0], y.shape[1]])
    max_dim = max(all_dims)
    assert max_dim <= grid_size, (
        f"Task dimension {max_dim} exceeds MAX_GRID_SIZE={grid_size}. "
        f"Increase MAX_GRID_SIZE to the next power of 2."
    )

    x_list, y_list = [], []
    for x, y in train_pairs:
        x_pad = jnp.pad(x, ((0, grid_size - x.shape[0]), (0, grid_size - x.shape[1])))
        y_pad = jnp.pad(y, ((0, grid_size - y.shape[0]), (0, grid_size - y.shape[1])))
        x_list.append(jax.nn.one_hot(x_pad, PHYSICAL_DIM))
        y_list.append(jax.nn.one_hot(y_pad, PHYSICAL_DIM))

    x_batch = jnp.stack(x_list)
    y_batch = jnp.stack(y_list)

    return x_batch, y_batch, train_pairs, test_pairs

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Pre-training Mode
# ══════════════════════════════════════════════════════════════════════════════

def run_pretrain(task_ids, num_epochs=5, checkpoint_path="core_checkpoint.pkl"):
    """
    Stage 1: Pre-train θ_base + hypernetwork MLP + PEPS embedding across
    a representative set of ARC tasks.

    Uses a reduced learning rate (LR_PRETRAIN = 0.01) for broader
    generalization. Cycles through all tasks for num_epochs, saving the
    converged params to a portable checkpoint file.
    """
    grid_size = MAX_GRID_SIZE
    chi = CHI
    steps_per_task = 50

    print(f"\n{'='*70}")
    print(f"  STAGE 1: PRE-TRAINING  |  {len(task_ids)} tasks  |  {num_epochs} epochs")
    print(f"  LR={LR_PRETRAIN}  |  {steps_per_task} steps/task/epoch")
    print(f"{'='*70}\n")

    # Initialize params
    key = jax.random.PRNGKey(0)
    params = init_all_params(key, grid_size, chi)
    enforce_parameter_constraints(params)

    # Create optimizer with pre-training LR
    langevin = langevin_dynamics_optimizer(learning_rate=LR_PRETRAIN, initial_T=1.0)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), langevin)
    opt_state = optimizer.init(params)
    update_step = make_update_step(optimizer)

    # Pre-load and pad ALL tasks
    print("  Loading tasks...")
    all_task_data = []
    for i, task_id in enumerate(task_ids):
        url = f"{ARC_BASE_URL}{task_id}.json"
        try:
            x_batch, y_batch, _, _ = _load_and_pad_task(url, grid_size)
            all_task_data.append((task_id, x_batch, y_batch))
            print(f"    [{i+1}/{len(task_ids)}] {task_id} — {x_batch.shape[0]} pairs")
        except Exception as e:
            print(f"    [{i+1}/{len(task_ids)}] {task_id} — SKIP ({e})")

    print(f"\n  Loaded {len(all_task_data)}/{len(task_ids)} tasks successfully.\n")

    # Training loop
    global_step = 0
    nan_count = 0

    for epoch in range(num_epochs):
        epoch_ce = []

        for task_idx, (task_id, x_batch, y_batch) in enumerate(all_task_data):
            for _ in range(steps_per_task):
                prev_params, prev_opt_state = params, opt_state

                params, opt_state, loss_val, aux = update_step(
                    params, opt_state, x_batch, y_batch, grid_size, chi
                )

                if jnp.isnan(loss_val):
                    nan_count += 1
                    params, opt_state = prev_params, prev_opt_state
                    if nan_count >= NAN_STRIKE_LIMIT:
                        print(f"  FATAL: NaN limit reached at epoch {epoch+1}, task {task_id}")
                        save_checkpoint(params, checkpoint_path)
                        return params
                    continue
                else:
                    nan_count = 0

                global_step += 1

            ce_loss = float(aux[0])
            epoch_ce.append(ce_loss)

            if (task_idx + 1) % 5 == 0 or task_idx == len(all_task_data) - 1:
                langevin_state = opt_state[1]
                temp = float(langevin_state.temperature)
                print(f"  Epoch {epoch+1}/{num_epochs} | Task {task_idx+1}/{len(all_task_data)} "
                      f"({task_id}) | CE={ce_loss:.4f} | T={temp:.5f} | step={global_step}")

        avg_ce = np.mean(epoch_ce)
        print(f"\n  --- Epoch {epoch+1} complete: avg CE = {avg_ce:.4f} ---\n")

    # Save final checkpoint
    save_checkpoint(params, checkpoint_path)

    print(f"\n{'='*70}")
    print(f"  STAGE 1 COMPLETE — {global_step} total steps")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*70}\n")

    return params

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Per-Task TTT Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def solve_single_task(task_url, grid_size, chi, checkpoint_path=None):
    """
    Loads one ARC task, runs TTT with NaN recovery, and returns the
    prediction as a numpy array (severing the JAX autograd graph).

    If checkpoint_path is provided, loads pre-trained params as the
    warm start instead of random initialization.

    Returns:
        dict with 'prediction', 'target', 'ce_loss', 'steps', 'frozen'
    """
    x_batch, y_batch, train_pairs, test_pairs = _load_and_pad_task(task_url, grid_size)

    print(f"  Loaded {len(train_pairs)} support pairs  |  grid_size={grid_size}  |  chi={chi}")

    # Initialize: checkpoint injection or fresh random
    if checkpoint_path:
        print(f"  Warm start from checkpoint: {checkpoint_path}")
        params = load_checkpoint(checkpoint_path)
    else:
        key = jax.random.PRNGKey(0)
        params = init_all_params(key, grid_size, chi)

    enforce_parameter_constraints(params)

    # Create optimizer with TTT learning rate
    langevin = langevin_dynamics_optimizer(learning_rate=LR_TTT, initial_T=1.0)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), langevin)
    opt_state = optimizer.init(params)
    update_step = make_update_step(optimizer)

    # TTT loop with NaN-poison recovery
    print(f"  {'Step':<6} | {'Temp (T)':<10} | {'Grad Var':<12} | {'CE Loss':<10} | {'SVD Entropy':<15}")
    print(f"  {'-' * 63}")

    step = 0
    nan_count = 0
    final_ce = float('inf')
    frozen = False

    while step <= MAX_TTT_STEPS:
        prev_params, prev_opt_state = params, opt_state

        params, opt_state, loss_val, aux = update_step(
            params, opt_state, x_batch, y_batch, grid_size, chi
        )
        ce_loss, svd_loss = aux

        # NaN-poison check
        if jnp.isnan(loss_val):
            nan_count += 1
            print(f"  Warning: NaN at step {step} — reverting ({nan_count}/{NAN_STRIKE_LIMIT})")
            params, opt_state = prev_params, prev_opt_state
            if nan_count >= NAN_STRIKE_LIMIT:
                print(f"  FATAL: {NAN_STRIKE_LIMIT} consecutive NaN events. Aborting task.")
                break
            step += 1
            continue
        else:
            nan_count = 0

        langevin_state = opt_state[1]
        temp = float(langevin_state.temperature)
        var  = float(langevin_state.grad_ema)
        final_ce = float(ce_loss)

        if step % 20 == 0 or temp < 1e-4:
            print(f"  {step:<6} | {temp:10.5f} | {var:12.7f} | {final_ce:10.5f} | {float(svd_loss):15.5f}")

        step += 1

        if temp < 1e-4:
            print(f"\n  *** FROZEN: Invariant rule extracted at step {step} ***")
            frozen = True
            break

    # Predict on test input
    x_test, y_test = test_pairs[0]
    x_test_pad = jnp.pad(x_test, ((0, grid_size - x_test.shape[0]),
                                    (0, grid_size - x_test.shape[1])))
    x_test_onehot = jax.nn.one_hot(x_test_pad, PHYSICAL_DIM)

    encode = functional_forward(params, grid_size, chi)
    decode = functional_reverse(params, grid_size, chi)

    embedded  = embed_grid_to_peps(x_test_onehot, params['peps'], chi)
    latent    = encode(embedded)
    predicted = decode(latent)
    logits    = decode_from_peps(predicted, params['peps'])

    predicted_colors = jnp.argmax(logits, axis=-1)
    predicted_crop   = predicted_colors[:y_test.shape[0], :y_test.shape[1]]

    # Convert to numpy to sever the JAX computation graph
    prediction_np = np.array(predicted_crop)
    target_np     = np.array(y_test)

    print_side_by_side(x_test, predicted_crop, y_test)

    return {
        'prediction': prediction_np,
        'target': target_np,
        'ce_loss': final_ce,
        'steps': step,
        'frozen': frozen,
    }

# ── Multi-task benchmark loop ──────────────────────────────────────────────

def run_benchmark(task_ids, base_url=ARC_BASE_URL, subset=None, checkpoint_path=None):
    """
    Iterates over a list of ARC task IDs, solving each independently.
    Parameters are freshly initialized per task (or loaded from checkpoint).
    JAX caches are cleared between tasks to prevent HBM accumulation.

    Args:
        task_ids: List of ARC task ID strings.
        base_url: Base URL for fetching task JSON files.
        subset: If provided, evaluate only the first N tasks.
        checkpoint_path: If provided, warm-start each task from this checkpoint.
    """
    if subset is not None:
        original_count = len(task_ids)
        task_ids = task_ids[:subset]
        print(f"  Subset mode: evaluating {len(task_ids)}/{original_count} tasks")

    total = len(task_ids)
    results = []

    print(f"\n{'='*70}")
    print(f"  QHRRN BENCHMARK — {total} tasks  |  MAX_GRID_SIZE={MAX_GRID_SIZE}  |  chi={CHI}")
    if checkpoint_path:
        print(f"  Warm start: {checkpoint_path}")
    print(f"{'='*70}\n")

    for i, task_id in enumerate(task_ids):
        task_url = f"{base_url}{task_id}.json"
        print(f"\n--- Task {i+1}/{total}: {task_id} ---")

        try:
            result = solve_single_task(task_url, MAX_GRID_SIZE, CHI,
                                       checkpoint_path=checkpoint_path)
            result['task_id'] = task_id
            results.append(result)

            match = np.array_equal(result['prediction'], result['target'])
            status = "CORRECT" if match else "INCORRECT"
            print(f"  Result: {status}  |  CE={result['ce_loss']:.4f}  |  steps={result['steps']}")

        except Exception as e:
            print(f"  ERROR: Task {task_id} failed -- {e}")
            results.append({
                'task_id': task_id,
                'prediction': None,
                'target': None,
                'ce_loss': float('inf'),
                'steps': 0,
                'frozen': False,
                'error': str(e),
            })

        # Free HBM
        jax.clear_caches()
        print(f"  [Memory] JAX caches cleared.")

    # Summary report
    print(f"\n{'='*70}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'='*70}")

    correct = sum(
        1 for r in results
        if r.get('prediction') is not None
        and r.get('target') is not None
        and np.array_equal(r['prediction'], r['target'])
    )
    failed = sum(1 for r in results if r.get('error'))
    total_run = total - failed

    print(f"  Tasks run:    {total_run}/{total}")
    print(f"  Correct:      {correct}/{total_run}")
    print(f"  Failed:       {failed}/{total}")
    if total_run > 0:
        valid_results = [r for r in results if r['ce_loss'] < float('inf')]
        if valid_results:
            avg_ce = np.mean([r['ce_loss'] for r in valid_results])
            avg_steps = np.mean([r['steps'] for r in valid_results])
            frozen_count = sum(1 for r in valid_results if r.get('frozen'))
            print(f"  Avg CE Loss:  {avg_ce:.4f}")
            print(f"  Avg Steps:    {avg_steps:.0f}")
            print(f"  Frozen (SSB): {frozen_count}/{total_run}")
    print(f"{'='*70}\n")

    return results

# ── Main entry point ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="QHRRN Test-Time Training Pipeline — pre-train, single task, or full benchmark."
    )
    # Stage 1: Pre-training
    parser.add_argument("--pretrain", default=None,
                        help="Path to file with task IDs for pre-training. "
                             "Saves core_checkpoint.pkl on completion.")
    parser.add_argument("--pretrain-epochs", type=int, default=5,
                        help="Number of epochs for pre-training (default: 5).")

    # Stage 2: Evaluation
    parser.add_argument("--benchmark", default=None,
                        help="Path to file with ARC task IDs (one per line).")
    parser.add_argument("--eval-subset", type=int, default=None,
                        help="Evaluate only the first N tasks from the benchmark file.")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to core_checkpoint.pkl for warm-start TTT.")

    # Single task
    parser.add_argument("--task-url", default=DEFAULT_TASK_URL,
                        help="URL to a single ARC task JSON (default: 007bbfb7).")

    args = parser.parse_args()

    if args.pretrain:
        # Stage 1: Pre-training
        with open(args.pretrain) as f:
            task_ids = [line.strip() for line in f
                        if line.strip() and not line.strip().startswith('#')]
        run_pretrain(task_ids, num_epochs=args.pretrain_epochs)

    elif args.benchmark:
        # Stage 2: Benchmark evaluation
        with open(args.benchmark) as f:
            task_ids = [line.strip() for line in f
                        if line.strip() and not line.strip().startswith('#')]
        run_benchmark(task_ids, subset=args.eval_subset,
                      checkpoint_path=args.checkpoint)

    else:
        # Single-task mode
        print("Initializing QHRRN Single-Task TTT\n")
        solve_single_task(args.task_url, MAX_GRID_SIZE, CHI,
                          checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
