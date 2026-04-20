import jax
import jax.numpy as jnp
from src.mera.cayley import cayley_transform

class Disentangler:
    """
    Unitary Disentangler (U) parameterized strictly via an anti-symmetric matrix
    transformed by the Cayley Mapping. This preserves U^T @ U = I perfectly across
    arbitrary Test-Time Training gradient steps.
    """
    def __init__(self, key: jax.Array, dim: int, scale: float = 1e-4):
        """
        Initializes the abstract weights. To force initial RG flow to mimic identity
        (preventing artificial noise injection), we use a deeply localized Gaussian
        centered at 0 scaled tightly.
        """
        self.dim = dim
        self.weights = jax.random.normal(key, (dim, dim)) * scale

    def get_unitary(self) -> jax.Array:
        """Returns the orthogonal matrix via the canonical Cayley transform."""
        return cayley_transform(self.weights)

    def forward(self, x: jax.Array) -> jax.Array:
        """
        Rotates geometric features in basis space.
        Applies U explicitly to block x.
        Assuming x has shape (..., dim).
        """
        U = self.get_unitary()
        return jnp.dot(x, U.T)

    def reverse(self, y: jax.Array) -> jax.Array:
        """
        Decodes the geometric state backwards.
        Reverses the unitary transformation via transpose inverse: U^-1 = U^T.
        y = x @ U^T => x = y @ U
        """
        U = self.get_unitary()
        return jnp.dot(y, U)

    def get_params(self):
        return self.weights

    def update_params(self, new_weights: jax.Array):
        self.weights = new_weights
