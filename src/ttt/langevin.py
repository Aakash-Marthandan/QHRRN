import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple

class LangevinState(NamedTuple):
    count: jax.Array
    temperature: jax.Array
    grad_ema: jax.Array
    rng_key: jax.Array          # Properly evolving PRNG key

def langevin_dynamics_optimizer(
    learning_rate: float, 
    initial_T: float = 1.0, 
    variance_threshold: float = 1e-3, 
    cooling_factor: float = 0.95,
    rng_key: jax.Array = None
) -> optax.GradientTransformation:
    """
    Constructs a dynamically responding Inference Optimizer implementing physical Spontaneous 
    Symmetry Breaking phases purely reflecting topological energy variances, bypassing blind epochs.
    
    The PRNG key is stored in the optimizer state and properly split on each
    update step, guaranteeing cryptographically distinct noise sequences.
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(42)

    def init_fn(params):
        return LangevinState(
            count=jnp.zeros([], jnp.int32),
            temperature=jnp.array(initial_T, dtype=jnp.float32),
            grad_ema=jnp.zeros([], dtype=jnp.float32),
            rng_key=rng_key
        )

    def update_fn(updates, state, params=None):
        # Evaluate global parameter topology gradient movement
        flat_updates = jnp.concatenate([jnp.ravel(u) for u in jax.tree.leaves(updates)])
        current_norm = jnp.linalg.norm(flat_updates)
        
        # Calculate trailing Exponential Moving Average tracking long term stable bounds
        alpha = 0.9
        new_grad_ema = alpha * state.grad_ema + (1.0 - alpha) * current_norm
        
        # Drive thermal thresholds strictly over plateau detection metrics
        variance = jnp.abs(current_norm - state.grad_ema)
        
        # The Dynamic Core shifting logic:
        # If gradient movement has stabilized (variance < threshold), cool the metal.
        # Otherwise, maintain temperature allowing structural "Melting" discovery.
        new_temperature = jax.lax.cond(
            variance < variance_threshold,
            lambda t: t * cooling_factor,  # Execute continuous Cooling scaling
            lambda t: t,                   # Melting bounds exploration maintained
            state.temperature
        )
        
        # Hard Symmetry-Breaking Phase Lock ensuring convergence strictly limits drift
        new_temperature = jnp.where(new_temperature < 1e-4, 0.0, new_temperature)
        
        # Stochastic Gaussian Parameter Distribution bounds dictated naturally by sqrt(2*lr*T)
        noise_scale = jnp.sqrt(2 * learning_rate * new_temperature)
        
        # Proper key evolution: split the stored key for this step's noise
        step_key, next_key = jax.random.split(state.rng_key)
        
        def add_thermal_noise(g, subkey):
            noise = jax.random.normal(subkey, g.shape) * noise_scale
            return g + noise

        # Dynamically inject thermal brownian noise targeting symmetry phase locations
        treedef = jax.tree.structure(updates)
        num_leaves = len(jax.tree.leaves(updates))
        subkeys = jax.random.split(step_key, num_leaves)
        
        noisy_updates = jax.tree.map(add_thermal_noise, updates, jax.tree.unflatten(treedef, subkeys))
        
        # Execute traditional Gradient step bounds mapped over noisy topologies natively
        final_updates = jax.tree.map(lambda g: -learning_rate * g, noisy_updates)
        
        new_state = LangevinState(
            count=state.count + 1,
            temperature=new_temperature,
            grad_ema=new_grad_ema,
            rng_key=next_key              # Evolving key for next step
        )
        
        return final_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)
