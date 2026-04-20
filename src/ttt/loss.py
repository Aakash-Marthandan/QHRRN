import jax
import jax.numpy as jnp

def holographic_entropy_proxy(params):
    """
    Computes a topological bound approximating the global Von Neumann entropy
    by evaluating the local SVD spectra of the parameters mapped across boundaries.
    Minimizing this value strictly enforces Area Law causation, keeping logic simple.
    
    Uses jax.tree.map + jnp.stack for fully JAX-native aggregation — no Python
    for-loop accumulator.
    """
    def leaf_entropy(w):
        """Compute Von Neumann entropy proxy for a single parameter leaf."""
        if w.ndim < 2:
            return jnp.float32(0.0)
        # Flatten high-rank tensors (e.g. PEPS site_tensor with ndim=5) to 2D
        if w.ndim > 2:
            w = w.reshape(w.shape[0], -1)
        # Perturbation: prevents degenerate singular values → NaN in SVD backward pass.
        # jnp.eye(rows, cols) handles non-square matrices correctly.
        rows, cols = w.shape
        w_safe = w + 1e-6 * jnp.eye(rows, cols)
        # Extract localized singular value spectrum reflecting structural complexity
        s = jnp.linalg.svd(w_safe, compute_uv=False)
        # Normalize the distribution (probability proxy)
        p = s / (jnp.sum(s) + 1e-8)
        # Apply standard Von Neumann localized entropy boundary (-p log p)
        return -jnp.sum(p * jnp.log(p + 1e-8))

    entropy_tree = jax.tree.map(leaf_entropy, params)
    entropy_leaves = jax.tree.leaves(entropy_tree)
    return jnp.sum(jnp.stack(entropy_leaves))

def compute_holographic_loss(params, reverse_logits, targets, lambda_entropy=0.01):
    """
    The Hamiltonian evaluating energy boundaries defined by geometric truth vs topological constraints.
    
    Args:
        params: Extracted network state dictionary for dimensionality analysis.
        reverse_logits: The generated grid matrix output natively expanding out of the Latent phase.
        targets: One-hot encoded ARC target grid structures.
        lambda_entropy: Parameter weighting the bounds restriction.
    """
    # 1. Categorical Cross Entropy reflecting actual dimensional recreation logic.
    probabilities = jax.nn.softmax(reverse_logits, axis=-1)
    
    # Standard Negative Log-Likelihood over the discrete categories
    loss_ce = -jnp.mean(jnp.sum(targets * jnp.log(probabilities + 1e-8), axis=-1))
    
    # 2. Local Von Neumann approximation restricting Volume Law parameter storage
    loss_complexity = holographic_entropy_proxy(params)
    
    total_loss = loss_ce + (lambda_entropy * loss_complexity)
    
    return total_loss, {"ce_loss": loss_ce, "entropy_complexity": loss_complexity}
