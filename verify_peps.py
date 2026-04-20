import sys
import os

# Add the root project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

try:
    import jax
    import jax.numpy as jnp
    from src.tensor.peps import PEPSTensor, create_lattice

    key = jax.random.PRNGKey(42)
    
    print("Testing Normal Initialization...")
    tensor_normal = PEPSTensor(chi=8, key=key, init_method='normal')
    print("Normal Shape:", tensor_normal.dimensions)
    assert tensor_normal.dimensions == (10, 8, 8, 8, 8)

    print("Testing Haar Initialization...")
    tensor_haar = PEPSTensor(chi=4, key=key, init_method='haar')
    print("Haar Shape:", tensor_haar.dimensions)
    assert tensor_haar.dimensions == (10, 4, 4, 4, 4)

    print("Testing Lattice Creation...")
    lattice = create_lattice((3, 3), chi=4, key=key)
    print(f"Lattice Shape: {lattice.shape}")
    assert lattice.shape == (3, 3)
    
    print("ALL TESTS PASSED")

except ImportError as e:
    print(f"Import Error (JAX might not be installed yet): {e}")
except Exception as e:
    print(f"Validation Error: {e}")
    sys.exit(1)
