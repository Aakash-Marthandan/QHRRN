import jax
import jax.numpy as jnp
from src.mera.cayley import cayley_transform

class Isometry:
    """
    Coarse-graining Isometry (W) projecting higher dimensional block vectors
    into condensed physical representations while maintaining W^T @ W = I
    over the Stiefel manifold properties.
    """
    def __init__(self, key: jax.Array, input_dim: int, reduced_dim: int, scale: float = 1e-4):
        """
        Constructs the weights tightly mapping identity space.
        """
        assert input_dim >= reduced_dim, "Isometry must purely compress or maintain rank."
        self.input_dim = input_dim
        self.reduced_dim = reduced_dim

        # Evaluate Stiefel behavior natively via a full Cayley transform on the input space
        self.weights = jax.random.normal(key, (input_dim, input_dim)) * scale

    def get_isometry(self) -> jax.Array:
        """
        Computes a strictly orthogonal matrix via the canonical Cayley transform
        and extracts the leading subspace columns for dimensionality reduction.
        """
        W_full = cayley_transform(self.weights)
        # W_isometry: Truncates to isolated mapping. (input_dim, reduced_dim)
        return W_full[:, :self.reduced_dim]

    def forward(self, x: jax.Array) -> jax.Array:
        """
        Pulls abstract local details sequentially downward.
        x shape: (..., input_dim) -> (..., reduced_dim)
        """
        W = self.get_isometry()
        return jnp.dot(x, W)

    def reverse(self, z: jax.Array) -> jax.Array:
        """
        Inflates the state vector geometrically.
        z shape: (..., reduced_dim).
        Pad the missing "vacuum" dimensions functionally with implicit 0-states
        by evaluating against W transpose.
        """
        W = self.get_isometry()
        return jnp.dot(z, W.T)

    def get_params(self):
        return self.weights

    def update_params(self, new_weights: jax.Array):
        self.weights = new_weights
