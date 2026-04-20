import jax
import jax.numpy as jnp
from src.pallas_kernels.sparse_contract import contract_sparse

def test_sparse_contract_equivalence():
    """
    Verifies Phase 2 Mathematical Equivalence + Zero-MXU wastage assertion.
    Constructs densely padded geometric arrays, multiplies them, and computes
    them using the sparse kernel proxy to assert accuracy.
    """
    block_size = 4
    grid_h = 4
    grid_w = 4
    
    H = grid_h * block_size
    W = grid_w * block_size
    
    # Create two matrices that are mostly zero.
    # Specifically, we inject values only at block coordinate (1,1) and (2,3)
    a_dense = jnp.zeros((H, W))
    b_dense = jnp.zeros((H, W))
    
    # Block 1,1
    a_dense = a_dense.at[4:8, 4:8].set(jax.random.normal(jax.random.PRNGKey(0), (4,4)))
    b_dense = b_dense.at[4:8, 4:8].set(jax.random.normal(jax.random.PRNGKey(1), (4,4)))
    
    # Block 2,3
    a_dense = a_dense.at[8:12, 12:16].set(jax.random.normal(jax.random.PRNGKey(2), (4,4)))
    b_dense = b_dense.at[8:12, 12:16].set(jax.random.normal(jax.random.PRNGKey(3), (4,4)))
    
    # Native dense dot proxy for symmetric element-wise block testing:
    # A standard element-wise matrix multiply handles the exact localized rule representation
    # standard dense equivalent:
    c_dense_expected = jnp.zeros((H, W))
    c_dense_expected = c_dense_expected.at[4:8, 4:8].set(jnp.dot(a_dense[4:8, 4:8], b_dense[4:8, 4:8]))
    c_dense_expected = c_dense_expected.at[8:12, 12:16].set(jnp.dot(a_dense[8:12, 12:16], b_dense[8:12, 12:16]))

    # Run sparse routine
    c_dense_actual, non_zero_count = contract_sparse(a_dense, b_dense, block_size)
    
    # 1. Test Mathematical Form Equivalence
    assert jnp.allclose(c_dense_expected, c_dense_actual, atol=1e-5), "Sparse matrix elements did not reconstruct dense representation successfully."
    
    # 2. Test MXU Zero-Wastage Proxy
    # Out of a 4x4 grid (16 blocks total), only 2 blocks contain values.
    # Therefore, the execution length grid on TPU MXU *must* strictly be 2.
    assert non_zero_count == 2, f"MXU Cycle test failed! Expected exactly 2 block computations, but evaluated {non_zero_count}."
    print("ALL TESTS PASSED: Contract Sparse Equivalence AND MXU Block-Wastage Check evaluated correctly.")

if __name__ == '__main__':
    test_sparse_contract_equivalence()
