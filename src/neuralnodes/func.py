import jax.numpy as jnp
import jax.scipy as jsc
from jax import jit
from . import utils


@jit
def ln(input, filter):
    try:
        outputLinear = utils.conv(input, filter)
    except:
        outputLinear = input
    return outputLinear[:input.shape[0], :]


@jit
def synapse(input, delay=0, weight=1):
    T = jnp.tile(jnp.arange(input.shape[0])[:, jnp.newaxis], (1, input.shape[1]))
    S = jnp.tile(jnp.arange(input.shape[1]), (input.shape[0], 1))
    output = jsc.ndimage.map_coordinates(input, [T -delay, S], mode='nearest', order=1)
    output = weight * output[:input.shape[0],:]
    return output


@jit
def summate(inputs):
        assert len(inputs) == 2
        assert inputs[0].shape == inputs[1].shape
        outputLinear = inputs[0] + inputs[1]
        return outputLinear


@jit
def divnorm(input, filter, offset=1, weight=1):
        normSignal = jnp.abs(utils.conv(input, filter))
        outputLinear = input / (offset + weight * normSignal[:input.shape[0],:])
        return outputLinear[:input.shape[0], :]
