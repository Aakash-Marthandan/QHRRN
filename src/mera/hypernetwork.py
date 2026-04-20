import jax
import jax.numpy as jnp
from typing import Dict

def init_hyper_params(
    key: jax.Array,
    block_dim: int,
    hidden_dim: int = 8,
    rank: int = 2,
    scale: float = 1e-4
) -> Dict:
    """
    Initialize the Modulation Hypernetwork parameters.
    
    Implements θ(s) = θ_base + Δ(s) where Δ(s) = MLP_small(s).
    The delta is low-rank factorized: Δ = u @ v.T with u,v ∈ R^{block_dim × rank}.
    This keeps the MLP output compact while providing full-matrix modulation.
    
    Args:
        key: JAX PRNG key.
        block_dim: Dimension of disentangler/isometry weight matrices (= 4 * chi).
        hidden_dim: MLP hidden layer width.
        rank: Rank of the low-rank delta factorization.
        scale: Initialization scale (tight around identity for localized causality).
    """
    k_bd, k_bi, k_w1, k_b1, k_w2d, k_w2i = jax.random.split(key, 6)

    # Shared base weights — initialized near identity via tight Gaussians
    base_d = jax.random.normal(k_bd, (block_dim, block_dim)) * scale
    base_i = jax.random.normal(k_bi, (block_dim, block_dim)) * scale

    # MLP: scalar scale index → low-rank factors for both Δ_d and Δ_i
    # Output per type: u (block_dim * rank) + v (block_dim * rank) = 2 * block_dim * rank
    output_per_type = 2 * block_dim * rank
    mlp_output_dim = 2 * output_per_type  # for d and i combined

    return {
        'base_d': base_d,
        'base_i': base_i,
        'mlp_w1': jax.random.normal(k_w1, (1, hidden_dim)) * 0.1,
        'mlp_b1': jnp.zeros(hidden_dim),
        'mlp_w2': jax.random.normal(k_w2d, (hidden_dim, mlp_output_dim)) * scale,
        'mlp_b2': jnp.zeros(mlp_output_dim),
    }


def modulate_weights(
    hyper_params: Dict,
    scale_idx: int,
    num_layers: int,
    block_dim: int,
    rank: int = 2
) -> tuple:
    """
    Compute scale-dependent effective weights: θ(s) = θ_base + Δ(s).
    
    The MLP maps a normalized scale index s ∈ [0,1] through a single hidden
    layer, then outputs low-rank factors (u, v) for both disentangler and
    isometry deltas.  The full delta is reconstructed as Δ = u @ v.T.
    
    Args:
        hyper_params: Dict from init_hyper_params.
        scale_idx: Integer layer index (0 = finest / UV, L-1 = coarsest / IR).
        num_layers: Total number of MERA layers.
        block_dim: Weight matrix dimension (= 4 * chi).
        rank: Low-rank factorization rank (must match init).
    
    Returns:
        (d_weights, i_weights): Effective weight matrices for this scale.
    """
    # Normalize scale to [0, 1]
    s_norm = scale_idx / max(num_layers - 1, 1)
    s_input = jnp.array([s_norm], dtype=jnp.float32)  # (1,)

    # Forward through tiny MLP
    h = jnp.tanh(jnp.dot(s_input, hyper_params['mlp_w1']) + hyper_params['mlp_b1'])  # (hidden,)
    raw_output = jnp.dot(h, hyper_params['mlp_w2']) + hyper_params['mlp_b2']  # (mlp_output_dim,)

    # Split output into low-rank factors for Δ_d and Δ_i
    factors_per_type = 2 * block_dim * rank
    d_factors = raw_output[:factors_per_type]
    i_factors = raw_output[factors_per_type:]

    # Reconstruct Δ_d = u_d @ v_d.T  (low-rank outer product)
    u_d = d_factors[:block_dim * rank].reshape(block_dim, rank)
    v_d = d_factors[block_dim * rank:].reshape(block_dim, rank)
    delta_d = jnp.dot(u_d, v_d.T)

    # Reconstruct Δ_i = u_i @ v_i.T
    u_i = i_factors[:block_dim * rank].reshape(block_dim, rank)
    v_i = i_factors[block_dim * rank:].reshape(block_dim, rank)
    delta_i = jnp.dot(u_i, v_i.T)

    # θ(s) = θ_base + Δ(s)
    d_weights = hyper_params['base_d'] + delta_d
    i_weights = hyper_params['base_i'] + delta_i

    return d_weights, i_weights
