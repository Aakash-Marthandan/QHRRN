import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from typing import Tuple

try:
    from jax.experimental.pallas import tpu as pltpu
except ImportError:
    pltpu = None

def pack_sparse_blocks(matrix: jax.Array, block_size: int) -> Tuple[jax.Array, jax.Array]:
    """
    Reshapes a dense matrix into a fixed-size array of blocks plus a boolean
    mask identifying which blocks are non-zero.
    
    JIT-safe: all output shapes are statically known at trace time.
    
    Returns:
        all_blocks: shape (total_blocks, block_size, block_size) — FIXED size
        mask: shape (total_blocks,) — True where block is non-zero
    """
    H, W = matrix.shape
    assert H % block_size == 0 and W % block_size == 0
    
    grid_h = H // block_size
    grid_w = W // block_size
    
    # Reshape into grid of blocks: (grid_h, grid_w, block_size, block_size)
    blocks = matrix.reshape(grid_h, block_size, grid_w, block_size).transpose(0, 2, 1, 3)
    
    # Boolean mask: True for non-zero blocks (fixed-size, no dynamic indexing)
    norms = jnp.linalg.norm(blocks, axis=(2, 3))
    mask = norms > 1e-6
    
    # Flatten spatial dims — shape is statically (grid_h * grid_w, block_size, block_size)
    all_blocks = blocks.reshape(grid_h * grid_w, block_size, block_size)
    mask_flat = mask.reshape(grid_h * grid_w)
    
    return all_blocks, mask_flat


def pallas_sparse_dot(a_blocks: jax.Array, b_blocks: jax.Array) -> jax.Array:
    """
    Pallas kernel compiled specifically for Cloud TPU architectures.
    Leverages pltpu.PrefetchScalarGridSpec to pipeline mapping.
    """
    num_blocks = a_blocks.shape[0]
    
    def dot_kernel(a_ref, b_ref, c_ref):
        c_ref[...] = jnp.dot(a_ref[...], b_ref[...])
        
    grid = (num_blocks,)
    
    compiler_params = {}
    if pltpu is not None:
        compiler_params['mosaic'] = dict(
            grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=1)
        )
        
    c_blocks = pl.pallas_call(
        dot_kernel,
        out_shape=jax.ShapeDtypeStruct(a_blocks.shape, a_blocks.dtype),
        in_specs=[
            pl.BlockSpec(lambda i: (i, 0, 0), (1, a_blocks.shape[1], a_blocks.shape[2])),
            pl.BlockSpec(lambda i: (i, 0, 0), (1, b_blocks.shape[1], b_blocks.shape[2]))
        ],
        out_specs=pl.BlockSpec(lambda i: (i, 0, 0), (1, a_blocks.shape[1], a_blocks.shape[2])),
        grid=grid,
        compiler_params=compiler_params
    )(a_blocks, b_blocks)
    
    return c_blocks


def contract_sparse(tensor_a: jax.Array, tensor_b: jax.Array, block_size: int) -> Tuple[jax.Array, jax.Array]:
    """
    JIT-safe block-sparse contraction.
    
    Uses fixed-size arrays and boolean masking throughout — no dynamic
    indexing, no variable-length intermediates.  The 3-argument jnp.where
    zeroes out results for empty blocks after computation.
    
    Returns:
        reconstructed_dense (jax.Array): The result matrix.
        non_zero_count (jax.Array): Number of non-zero blocks (proxy metric).
    """
    H, W = tensor_a.shape
    grid_h = H // block_size
    grid_w = W // block_size
    
    a_blocks, a_mask = pack_sparse_blocks(tensor_a, block_size)
    b_blocks, b_mask = pack_sparse_blocks(tensor_b, block_size)
    
    # Combined mask: only blocks where BOTH inputs are non-zero matter
    mask = a_mask & b_mask
    non_zero_count = jnp.sum(mask)
    
    backend = jax.lib.xla_bridge.get_backend()
    
    if backend.platform == 'tpu':
        c_blocks = pallas_sparse_dot(a_blocks, b_blocks)
    else:
        # CPU/GPU fallback: compute ALL block dot products via vmap
        # Zeros naturally produce zero results; mask safety below.
        c_blocks = jax.vmap(jnp.dot)(a_blocks, b_blocks)
    
    # Zero out results for empty blocks (3-argument jnp.where is JIT-safe)
    c_blocks = jnp.where(mask[:, None, None], c_blocks, 0.0)
    
    # Reconstruct dense matrix
    c_dense = c_blocks.reshape(grid_h, grid_w, block_size, block_size)
    c_dense = c_dense.transpose(0, 2, 1, 3).reshape(H, W)
    
    return c_dense, non_zero_count
