import jax.numpy as jnp

def cayley_transform(weights):
    """
    Single source of truth for the Cayley mapping.
    Maps an unconstrained weight matrix to an orthogonal matrix via:
        U = (I - A)(I + A)^{-1}
    where A is the anti-symmetric part of the weights.
    
    The dimension is inferred dynamically from weights.shape[0],
    so this scales correctly regardless of bond dimension chi.
    """
    dim = weights.shape[0]
    A = 0.5 * (weights - weights.T)
    I = jnp.eye(dim)
    U_T = jnp.linalg.solve(I + A, I - A)
    return U_T.T
