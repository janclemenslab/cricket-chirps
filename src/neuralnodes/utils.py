import numpy as np
import jax.numpy as jnp
import scipy.signal.windows
from itertools import product
import jax.scipy as jsc
from jax import jit
from functools import partial
from scipy.optimize import minimize


@jit
def conv(x, filter):
    """filter...

    Args:
        x ([type]): [description]
        filter (np.array like)

    Returns:
        [type]: [description]
    """
    if filter.ndim==1 and x.ndim==2:
        filter = filter[:, np.newaxis]
    x = jsc.signal.convolve2d(x, filter[:x.shape[0]-1,:], mode='full')
    return x

@jit
def conv1d(x, filter):
    """filter...

    Args:
        x ([type]): [description]
        filter (np.array like)

    Returns:
        [type]: [description]
    # """
    x = jsc.signal.convolve(x, filter, mode='full')
    return x


#@partial(jit, static_argnums=(0,))
def gausswin(N: float, a: float = 2.5):
    N = N-1
    if N<=1:
        w = jnp.ones((1,))
    else:
        n = jnp.arange(-N/2, N/2)
        w = jnp.exp(-(1/2)*(a*n/(N/2))**2)
    return w
