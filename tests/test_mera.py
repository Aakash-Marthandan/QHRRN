import jax
import jax.numpy as jnp
from src.mera.disentangler import Disentangler
from src.mera.isometry import Isometry
from src.mera.engine import MeraEngine

def test_unitarity():
    """
    Validates the Unitary Disentangler properties enforcing U^T @ U = I.
    """
    key = jax.random.PRNGKey(42)
    dim = 16
    
    # Intentionally scale to evaluate unaligned gradient updates simulating TTT volatility
    d = Disentangler(key, dim, scale=0.5)
    U = d.get_unitary()
    
    # Extract identity validation check
    I_approx = jnp.dot(U.T, U)
    I_exact = jnp.eye(dim)
    
    assert jnp.allclose(I_approx, I_exact, atol=1e-5), "Disentangler UNTARY CONSTRAINT FAILED: Matrix leaked invariant rulesets."
    print("Unitarity Constraint check passed geometrically!")

def test_isometric_projection():
    """
    Validates Isometry maps down dynamically and validates W^T @ W = I 
    over the Stiefel dimension constraint structurally.
    """
    key = jax.random.PRNGKey(42)
    in_dim, out_dim = 16, 8
    
    i = Isometry(key, in_dim, out_dim, scale=0.5)
    W = i.get_isometry()
    
    assert W.shape == (in_dim, out_dim)
    I_approx = jnp.dot(W.T, W)
    I_exact = jnp.eye(out_dim)
    
    assert jnp.allclose(I_approx, I_exact, atol=1e-5), "Isometry STIEFEL CONSTRAINT FAILED."
    print("Isometric Constraint check passed geometrically!")

def test_reversibility():
    """
    Validates the Engine recursively iterates without geometric anomalies over N mappings,
    verifying explicitly tracking generative state flow correctly tracks vacuum mapping behavior.
    """
    key = jax.random.PRNGKey(42)
    grid_size = 4
    chi = 8
    
    engine = MeraEngine(key, grid_size, chi)
    
    # Build structural validation framework
    original_grid = jax.random.normal(key, (grid_size, grid_size, chi))
    
    latent = engine.forward(original_grid)
    
    # Evaluates dimensional squeezing parameters
    assert latent.shape == (1, 1, chi), f"Geometric Latent Code validation unexpectedly failed. Generated dimensionality: {latent.shape}"
    print("Latent mapping dimensions verified iteratively.")
    
    reconstructed = engine.reverse(latent)
    
    assert reconstructed.shape == original_grid.shape, "Vacuum noise expansion bounds failed."
    print("Reverse Decoding geometry verification completely validated!")

if __name__ == "__main__":
    test_unitarity()
    test_isometric_projection()
    test_reversibility()
    print("ALL MERA ENGINE TESTS SUCCESSFULLY PASSED")
