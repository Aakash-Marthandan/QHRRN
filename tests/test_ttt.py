import jax
import jax.numpy as jnp
import optax
from src.ttt.loss import holographic_entropy_proxy, compute_holographic_loss
from src.ttt.langevin import langevin_dynamics_optimizer
from src.ttt.constraints import enforce_parameter_constraints

def test_holographic_proxy():
    print("Testing SVD local dimensionality approximation...")
    # Generate mock weights representing isometric constraints bounds
    key = jax.random.PRNGKey(0)
    mock_params = {
        'layer_1': jax.random.normal(key, (16, 16)),
        'layer_2': jax.random.normal(key, (8, 8))
    }
    
    # Execute proxy mapping dynamically tracking local boundary metrics
    entropy = holographic_entropy_proxy(mock_params)
    assert float(entropy) > 0, "SVD Entropy mapping functionally halted incorrectly."
    print("SVD proxy computed bounded dimensionality dynamically successfully.")

def test_langevin_phase_shift_dynamics():
    print("Testing dynamic phase temperature transitions...")
    optimizer = langevin_dynamics_optimizer(learning_rate=0.1, initial_T=1.0, variance_threshold=100.0)
    
    mock_params = {'weight': jnp.zeros((10, 10))}
    opt_state = optimizer.init(mock_params)
    
    assert opt_state.temperature == 1.0, "Initial Melting Phase configuration bypassed incorrectly!"
    
    # Simulate a stationary gradient representing structural alignment over TTT evaluation
    # Given variance threshold is artificially inflated, any bounded stability immediately
    # initiates cooling boundaries autonomously.
    mock_grad = {'weight': jnp.ones((10, 10))}
    updates, opt_state = optimizer.update(mock_grad, opt_state, mock_params)
    
    assert opt_state.temperature < 1.0, f"Dynamic autonomous plateaus bypassed incorrectly! T={opt_state.temperature}"
    print("Langevin phase shift actively decayed geometry constraints securely dynamically.")

def test_constraints_limit():
    print("Validating constraints capacity hard limits mapping...")
    valid_architecture = jnp.zeros((100, 100))
    count = enforce_parameter_constraints(valid_architecture)
    assert count == 10000, "Structural parameter counting bypassed local matrices incorrectly."
    
    try:
        violation_architecture = jnp.zeros((4000, 4000))  # 16,000,000 params
        enforce_parameter_constraints(violation_architecture)
        assert False, "Constraints validation allowed >10M dimensional boundaries to generate without halting."
    except ValueError as e:
        assert "FATAL REJECTION" in str(e), "Constraints rejection message missing correct format parameters."
        print("Model Size Validation constraints successfully crashed oversize mapping.")

if __name__ == "__main__":
    test_holographic_proxy()
    test_langevin_phase_shift_dynamics()
    test_constraints_limit()
    print("ALL TEST-TIME TRAINING DYNAMICS SUCCESSFULLY PASSED!")
