"""
see kernel.m
"""
import jax.numpy as jnp
from jax import jit
from functools import partial


# @partial(jit, static_argnums=(0,))
def exponential(duration, tau):
    """ kernel = exponential(duration, tau)
        normalized exponential kernel: EXP(-t/tau), width TAU in points."""
    if duration>=1:
        kernel = jnp.exp(-(jnp.arange(0, duration-1)/tau))/tau
    else:
        kernel = jnp.ones((1,))
    return kernel
