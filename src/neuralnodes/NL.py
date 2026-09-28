# import numpy as np
import jax.numpy as np
# from jax import jit


# @jit
def exponential(x, slope: float = 1, offset: float = 0):
    return np.exp(slope * x) + offset

# @jit
def linear(x, slope: float = 1, bias: float = 0):
    return slope * x + bias

# @jit
def identity(x):
    return x

# @jit
def rectifier(x, minval = 0, maxval = None):
    return np.clip(x, minval, maxval)

# @jit
def relu(x, thres: float = 0):
    return np.clip(x, thres, None) - thres

# @jit
def sigmoidal(x, slope=1, midpoint=0, gain=1, offset=0):
    return offset + gain / (1 + np.exp( -slope * (x - midpoint)))

# @jit
def sigmoidal_fast(x, slope=1, midpoint=0, gain=1, offset=0):
    """Elliot's fast approx. to the Boltzmann."""
    x = slope * (x - midpoint)
    x = offset + gain * x / (1 + np.abs(x))
    return x * 0.5 + 0.5

# @jit
def square(x, offset: float = 0):
    return x**2 + offset
