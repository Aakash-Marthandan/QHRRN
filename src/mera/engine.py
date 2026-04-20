import jax
import jax.numpy as jnp
from src.mera.disentangler import Disentangler
from src.mera.isometry import Isometry

class MeraEngine:
    """
    Coordinates topological flow iterating geometric abstraction 
    levels recursively N -> N/2 ... -> 1 while ensuring bidirectional 
    reversibility mappings.
    """
    def __init__(self, key: jax.Array, grid_size: int, chi: int):
        self.chi = chi
        self.grid_size = grid_size
        self.layers = []
        
        current_size = grid_size
        idx = 0
        
        # Recursively fold geometric scale downward until the logic maps to 1 causal latent region
        while current_size > 1:
            # We enforce symmetrical isolation of subkeys targeting isolated block layers
            k1, k2 = jax.random.split(jax.random.fold_in(key, idx), 2)
            
            # Each local interaction step parses a 2x2 grid representing physically
            # 4 interacting chi bonds. Resulting localized dimensionality: 4*chi
            block_dim = 4 * chi
            
            # Establish the unitary Disentangler rotating feature noise internally
            d = Disentangler(key=k1, dim=block_dim)
            
            # Isometrically squeeze the invariant states to localized scale representation
            i = Isometry(key=k2, input_dim=block_dim, reduced_dim=chi)
            
            self.layers.append((d, i))
            current_size //= 2
            idx += 1

    def forward(self, grid: jax.Array) -> jax.Array:
        """
        Propagates spatial abstraction iteratively.
        Input Grid Expects: (H, W, chi)
        """
        state = grid
        for d, i in self.layers:
            H, W, C = state.shape
            
            # Map structural components out into distinct 2x2 matrices decoupled natively
            state_blocks = state.reshape(H // 2, 2, W // 2, 2, C).transpose(0, 2, 1, 3, 4)
            
            # Condense the 2x2 boundaries functionally representing causal constraints
            state_flat = state_blocks.reshape(H // 2, W // 2, 4 * C)
            
            # Pass 1: Disentangle spatial entanglement to align structure into isolated channels
            disentangled = d.forward(state_flat)
            
            # Pass 2: Isometry dimensionally projects away strictly isolated vacuum blocks
            state = i.forward(disentangled)
            
        return state

    def reverse(self, latent: jax.Array) -> jax.Array:
        """
        Inversion engine natively injecting 0-noise vacuum to reconstitute
        geometry dynamically backward toward scale-1 resolutions.
        """
        state = latent
        # Ensure operations act iteratively upward natively
        for d, i in reversed(self.layers):
            # Decompress structural scale expanding via padding
            expanded = i.reverse(state)
            
            # Reconstruct rotated parameters across standard bases natively
            entangled = d.reverse(expanded)
            
            # Fold states spatially reflecting initial PEPS symmetries
            H, W, C_flat = entangled.shape
            C = C_flat // 4
            
            blocks = entangled.reshape(H, W, 2, 2, C)
            state = blocks.transpose(0, 2, 1, 3, 4).reshape(H * 2, W * 2, C)
            
        return state
