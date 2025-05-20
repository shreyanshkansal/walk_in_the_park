import jax
import jax.numpy as jnp

print("JAX version:", jax.__version__)
print("Devices:", jax.devices())

x = jnp.array([1.0, 2.0, 3.0])
y = jnp.sin(x)
print("sin(x):", y)
