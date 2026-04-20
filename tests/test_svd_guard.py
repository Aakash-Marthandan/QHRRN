import jax
import jax.numpy as jnp
from src.ttt.loss import holographic_entropy_proxy

# Test 1: All-zero matrix — previously would produce NaN gradients
print("Test 1: SVD perturbation on all-zero matrix")
zero_params = {'w': jnp.zeros((4, 4))}
val = holographic_entropy_proxy(zero_params)
print(f"  Entropy on zeros: {float(val):.6f}")

grad_fn = jax.grad(holographic_entropy_proxy)
g = grad_fn(zero_params)
grad_finite = bool(jnp.all(jnp.isfinite(g['w'])))
grad_norm = float(jnp.linalg.norm(g['w']))
print(f"  Grad finite: {grad_finite}")
print(f"  Grad norm:   {grad_norm:.6f}")
assert grad_finite, "FAIL: SVD gradient is NaN/Inf on zero matrix!"

# Test 2: Near-degenerate matrix (repeated singular values)
print("\nTest 2: SVD perturbation on near-degenerate matrix")
degen_params = {'w': jnp.ones((4, 4)) * 0.5}
val2 = holographic_entropy_proxy(degen_params)
g2 = grad_fn(degen_params)
grad_finite2 = bool(jnp.all(jnp.isfinite(g2['w'])))
print(f"  Entropy: {float(val2):.6f}")
print(f"  Grad finite: {grad_finite2}")
assert grad_finite2, "FAIL: SVD gradient is NaN/Inf on degenerate matrix!"

# Test 3: Non-square matrix
print("\nTest 3: Non-square matrix (8, 256)")
rect_params = {'w': jax.random.normal(jax.random.PRNGKey(0), (8, 256)) * 0.01}
val3 = holographic_entropy_proxy(rect_params)
g3 = grad_fn(rect_params)
grad_finite3 = bool(jnp.all(jnp.isfinite(g3['w'])))
print(f"  Entropy: {float(val3):.6f}")
print(f"  Grad finite: {grad_finite3}")
assert grad_finite3, "FAIL: SVD gradient is NaN/Inf on non-square matrix!"

print("\nALL SVD PERTURBATION TESTS PASSED")
