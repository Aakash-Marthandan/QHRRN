import jax
import jax.numpy as jnp
from typing import Literal, Tuple, Dict

class PEPSTensor:
    """
    Represents a fundamental 5-index tensor in the 2D PEPS Lattice.
    Indices: (physical, left, right, up, down)
    """
    def __init__(
        self,
        chi: int,
        key: jax.Array,
        init_method: Literal['normal', 'haar'] = 'normal',
        physical_dim: int = 10
    ):
        """
        Initializes the PEPS tensor.

        Args:
            chi (int): The configurable Bond Dimension for the 4 virtual indices.
            key (jax.Array): JAX PRNG key for random initialization.
            init_method (str): 'normal' for Gaussian, 'haar' for random orthogonal/unitary bounds.
            physical_dim (int): Physical index dimension (default 10 for ARC categorical colors).
        """
        self.chi = chi
        self.physical_dim = physical_dim
        self.shape = (physical_dim, chi, chi, chi, chi)
        self.tensor = self._initialize_tensor(key, init_method)

    def _initialize_tensor(self, key: jax.Array, init_method: str) -> jax.Array:
        if init_method == 'normal':
            # Standard random normal initialization scaled by 1/sqrt(chi)
            return jax.random.normal(key, self.shape) / jnp.sqrt(self.chi)
        elif init_method == 'haar':
            # For Haar-like initialization of a highly structured tensor without blowing up memory,
            # we construct the 5-index tensor by locally entangling orthogonal matrices for each bond.
            # In a strict physics context, the PEPS core would be contracted with random unitaries on the virtual legs.
            key, *subkeys = jax.random.split(key, 6)
            
            # Generate random orthogonal matrices for each of the 4 virtual indices
            orth_mats = []
            for i in range(4):
                random_matrix = jax.random.normal(subkeys[i], (self.chi, self.chi))
                q, _ = jnp.linalg.qr(random_matrix)
                orth_mats.append(q)
                
            # Initialize a core physical embedding
            core = jax.random.normal(subkeys[4], (self.physical_dim, self.chi, self.chi, self.chi, self.chi))
            
            # Apply orthogonal transformations along each leg (simulating random unitary bonding)
            # Using jnp.einsum for tensor contraction
            # core: p, i, j, k, l. orth_mats: (i, L), (j, R), (k, U), (l, D)
            t = core
            t = jnp.einsum('pijkl,ia->pajkl', t, orth_mats[0])
            t = jnp.einsum('pajkl,jb->pabkl', t, orth_mats[1])
            t = jnp.einsum('pabkl,kc->pabcl', t, orth_mats[2])
            t = jnp.einsum('pabcl,ld->pabcd', t, orth_mats[3])
            
            return t / jnp.sqrt(self.chi)
        else:
            raise ValueError(f"Unknown init_method: {init_method}")

    def get_tensor(self) -> jax.Array:
        """Returns the instantiated JAX tensor."""
        return self.tensor
    
    @property
    def dimensions(self) -> Tuple[int, int, int, int, int]:
        return self.shape

def create_lattice(grid_size: Tuple[int, int], chi: int, key: jax.Array) -> jax.Array:
    """
    Helper function to scaffold an N x N lattice of PEPSTensors.
    For Phase 1, we return a grid object or a combined JAX array.
    """
    import numpy as np # For building the grid references before pure JAX compilation
    rows, cols = grid_size
    lattice = np.empty((rows, cols), dtype=object)
    
    keys = jax.random.split(key, rows * cols)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            lattice[r, c] = PEPSTensor(chi=chi, key=keys[idx], init_method='normal')
            idx += 1
            
    return lattice

# ---------------------------------------------------------------------------
# PEPS Embedding Pipeline (Phase 7 — connects PEPS substrate to MERA engine)
# ---------------------------------------------------------------------------

def init_peps_params(key: jax.Array, physical_dim: int = 10, chi: int = 4) -> Dict:
    """
    Initialize learnable PEPS embedding parameters.
    
    The site_tensor is a shared 5-index PEPS tensor:
        (physical_dim, chi, chi, chi, chi)
    This is the core object the specification mandates — one physical index
    (dim 10 for ARC colors) and four virtual bond indices (dim chi each)
    enforcing the Area Law.
    
    The bond_projector performs approximate local contraction of the 4
    virtual legs into a single chi-dimensional feature vector.  On the
    local RTX 3070 this replaces the intractable O(chi^7) full PEPS
    contraction; on Cloud TPU the Pallas sparse kernel handles it.
    
    The decoder maps chi-dimensional MERA output back to 10 color logits.
    """
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        'site_tensor': jax.random.normal(k1, (physical_dim, chi, chi, chi, chi)) / jnp.sqrt(chi),
        'bond_projector': jax.random.normal(k2, (chi**4, chi)) / jnp.sqrt(chi**2),
        'decoder': jax.random.normal(k3, (chi, physical_dim)) / jnp.sqrt(chi),
    }

def embed_grid_to_peps(grid_onehot: jax.Array, peps_params: Dict, chi: int) -> jax.Array:
    """
    Embeds a one-hot encoded ARC grid into the PEPS lattice.
    
    Step 1: Contracts the physical index (dim 10) of the shared 5-index
            PEPS site tensor with the one-hot pixel value at each grid site.
            Result: 4-leg virtual bond tensor (chi, chi, chi, chi) per site.
    
    Step 2: Approximate local contraction — flattens the 4 virtual legs
            and projects via a learned boundary contraction matrix to produce
            a chi-dimensional feature vector per site.
    
    Args:
        grid_onehot: (H, W, 10) — one-hot encoded pixel grid
        peps_params: dict with 'site_tensor', 'bond_projector', 'decoder'
        chi: bond dimension (must match peps_params shapes)
    
    Returns:
        (H, W, chi) — PEPS-embedded feature grid ready for the MERA engine
    """
    site_tensor = peps_params['site_tensor']       # (10, chi, chi, chi, chi)
    bond_proj   = peps_params['bond_projector']     # (chi**4, chi)
    
    # Contract physical index with one-hot pixel values.
    # For each site (h,w): B[l,r,u,d] = sum_p one_hot[h,w,p] * T[p,l,r,u,d]
    B = jnp.einsum('hwp,plrud->hwlrud', grid_onehot, site_tensor)
    # B: (H, W, chi, chi, chi, chi) — the 4 virtual Area Law bonds per site
    
    H, W = B.shape[:2]
    
    # ── Open Boundary Conditions ──────────────────────────────────────────
    # Edge sites have virtual legs pointing outside the grid.  We project
    # each boundary-facing leg onto the trivial [1, 0, ..., 0] subspace,
    # effectively capping the boundary bond dimension to 1.  This strictly
    # enforces the Area Law: no information leaks through non-existent bonds.
    bv = jnp.zeros(chi).at[0].set(1.0)
    
    # Left boundary (w=0): clamp left virtual leg (axis 2)
    B = B.at[:, 0].set(B[:, 0] * bv[None, :, None, None, None])
    # Right boundary (w=-1): clamp right virtual leg (axis 3)
    B = B.at[:, -1].set(B[:, -1] * bv[None, None, :, None, None])
    # Top boundary (h=0): clamp up virtual leg (axis 4)
    B = B.at[0, :].set(B[0, :] * bv[None, None, None, :, None])
    # Bottom boundary (h=-1): clamp down virtual leg (axis 5)
    B = B.at[-1, :].set(B[-1, :] * bv[None, None, None, None, :])
    
    # Flatten the 4 virtual legs into a single vector per site
    B_flat = B.reshape(H, W, chi**4)
    
    # Learned approximate contraction → (H, W, chi)
    features = jnp.dot(B_flat, bond_proj)
    
    return features

def decode_from_peps(features: jax.Array, peps_params: Dict) -> jax.Array:
    """
    Decodes chi-dimensional MERA output features back to 10-category logits.
    
    Args:
        features: (H, W, chi) — output from MERA reverse pass
        peps_params: dict containing 'decoder' of shape (chi, 10)
    
    Returns:
        (H, W, 10) — logits over the 10 ARC color categories
    """
    return jnp.dot(features, peps_params['decoder'])
