from neuralnodes import utils, NL
from neuralnodes.utils import gausswin
from neuralnodes import kernel
import neuralnodes.func

from functools import partial

import jax.numpy as np
import jax.numpy as jnp
from .params import Params


def to_int(x):
    return jnp.array(x, dtype=int)


# @partial(jit, static_argnums=(1,))
def run_model(
    stim: np.ndarray, p: Params, verbose_output: bool = False, return_dict: bool = False
):
    # --- STIMULUS -----------------------------------------------------------
    try:
        p = Params.from_list(p)
    except:
        pass

    # --- AN1 -----------------------------------------------------------------
    #     - almost no adaptation and hence weak inh lobe
    #     - no strong smoothing/integration - hence rel. short pos. lobe
    #     - FIXED MINIMAL DELAY TO 5ms
    filt_delay = jnp.zeros((np.round(5 + p.an1_filter_delay).astype(int),))
    filt_pos = utils.gausswin(p.an1_filter_exc_dur, p.an1_filter_exc_sigma)
    filt_neg = -p.an1_filter_inh_gain * utils.gausswin(
        p.an1_filter_inh_dur, p.an1_filter_inh_sigma
    )

    p.an1_filter = np.concatenate([filt_delay, filt_pos, filt_neg], axis=0)

    an1_sub_output = neuralnodes.func.ln(stim, p.an1_filter)
    p.an1_sub_output = an1_sub_output
    an1_sub_output = NL.relu(an1_sub_output)

    an1_output = neuralnodes.func.divnorm(
        an1_sub_output,
        kernel.exponential(1000, p.an1_ada_filter_tau),
        offset=1,
        weight=p.an1_ada_strength,
    )

    nl_sigmoidal = partial(
        NL.sigmoidal,
        slope=p.an1_nonlinearity_slope,
        midpoint=p.an1_nonlinearity_shift,
        gain=p.an1_nonlinearity_gain,
        offset=p.an1_nonlinearity_baseline,
    )
    p.an1_nonlinearity = lambda x=None: NL.relu(nl_sigmoidal(x))
    an1_output = p.an1_nonlinearity(an1_output)

    an1_ln2_output = neuralnodes.func.synapse(
        an1_output, p.an1_ln2_delay, p.an1_ln2_gain
    )

    # --- LN2 -----------------------------------------------------------------
    # --  FILTER & NONLINEARITY:
    #     - stronger adaptation across pulses in a train - hence longer and stronger neg. lobe
    #     - no strong smoothing/integration - hence very short pos. lobe
    #     - nonlinearity thresholds to get rid of neg. resp. components from neg. filter lobe
    POS = p.ln2_filter_exc_gain * gausswin(p.ln2_filter_exc_dur, p.ln2_filter_exc_sigma)
    NEG = -np.flipud(kernel.exponential(1000, p.ln2_filter_inh_tau).T)
    p.ln2_filter = np.flipud(np.concatenate((NEG, POS[2:])))
    p.ln2_nonlinearity = lambda x=None: NL.relu(x) * p.ln2_nonlinearity_gain

    ln2_output = p.ln2_nonlinearity(neuralnodes.func.ln(an1_ln2_output, p.ln2_filter))

    # --- LN5 -----------------------------------------------------------------
    # --  LN2-LN5 SYNAPSE - DELAY
    #     - synaptic delay
    ln2_ln5_output = neuralnodes.func.synapse(
        ln2_output, p.ln2_ln5_delay, p.ln2_ln5_gain
    )

    # --  LN5 INPUT strong adaptation via differentiating filter
    #     - inh. decreases during long pulses - this reduces PIR for long pulses
    p.ln5_ada_filter = np.diff(gausswin(p.ln5_ada_filter_dur, 3.5))
    p.ln5_ada_filter = p.ln5_ada_filter.at[
        int(np.ceil(p.ln5_ada_filter_dur / 2) - 1) :
    ].set(
        p.ln5_ada_filter_exc_gain
        * p.ln5_ada_filter[int(np.ceil(p.ln5_ada_filter_dur / 2) - 1) :]
    )

    p.ln5_ada_nonlinearity = lambda x=None: np.clip(x, -np.inf, 0)
    ln5_ada_output = p.ln5_ada_nonlinearity(
        neuralnodes.func.ln(ln2_ln5_output, p.ln5_ada_filter)
    )

    # --  FILTER:
    #     - inverted biphasic filter to produce post-inhibitory rebound
    #     - reproduces timing and time coues of rebound
    #     - fits resposnes to LONG pause stimuli very well
    #     - does not match pause and duration tuning of PIR amplitude
    #     - maybe this is fixed with adapatation and long inh lobe that integrates PIR across pulses
    p.ln5_filter = np.concatenate(
        [
            kernel.exponential(p.ln5_filter_exc_dur, p.ln5_filter_exc_tau).T
            * p.ln5_filter_exc_gain,
            -kernel.exponential(500, p.ln5_filter_inh_tau).T * p.ln5_filter_inh_gain,
        ]
    )
    p.ln5_filter = utils.conv1d(p.ln5_filter, gausswin(6))

    p.ln5_nonlinearity = lambda x=None: NL.linear(x, p.ln5_nonlinearity_gain)
    ln5_output = p.ln5_nonlinearity(neuralnodes.func.ln(ln5_ada_output, p.ln5_filter))

    # --- LN3 -----------------------------------------------------------------
    # --  AN1-LN3 synapse
    #     - delay is a little long!! <- may just be there to compensate for
    #       overall delays introduced by the various filters upstream of LN3
    # an1_ln3_output = neuralnodes.func.synapse(an1_output, p.an1_ln3_delay, p.an1_ln3_gain)
    an1_ln3_output = neuralnodes.func.synapse(
        an1_output, p.an1_ln3_delay, p.an1_ln3_gain
    )

    # --  LN5-LN3 synapse
    #     - no delay
    p.ln5_output_relu = NL.relu(ln5_output, 0)
    ln5_ln3_output = neuralnodes.func.synapse(
        p.ln5_output_relu, p.ln5_ln3_delay, p.ln5_ln3_gain
    )

    # --  SUBTHRESHOLD - coincidence detector
    #     - responds only if AN1 is active (to first pulse but not to last
    #       rebound
    #     - LN5 mainly modulates response
    #     - summation - not multiplications, since AN1 is sufficient to elicit spikes
    p.ln3_sub_nonlinearity = (
        lambda x=None: NL.relu(x, p.ln3_sub_nonlinearity_thres)
        * p.ln3_sub_nonlinearity_gain
    )
    ln3_sub_output = p.ln3_sub_nonlinearity(
        neuralnodes.func.summate([ln5_ln3_output, an1_ln3_output])
    )

    # --  SPIKING ADAPTATION
    #     - adaptation reduces spiking for long chirps
    #     - this is not apparent in the subthreshold responses so must be SFA
    #     - hence ln3_sub has threshold NL to  mimic driving current on which the divNorm node acts
    p.ln3_nonlinearity = (
        lambda x=None: NL.relu(x, p.ln3_nonlinearity_thres) * p.ln3_nonlinearity_gain
    )

    ln3_output = neuralnodes.func.divnorm(
        ln3_sub_output,
        kernel.exponential(1000, p.ln3_ada_filter_tau),
        offset=1,
        weight=p.ln3_ada_strength,
    )
    ln3_output = p.ln3_nonlinearity(ln3_output)
    # --- LN4 -----------------------------------------------------------------
    # --  LN2-LN4 synapse
    ln2_ln4_output = neuralnodes.func.synapse(
        ln2_output, p.ln2_ln4_delay, p.ln2_ln4_gain
    )

    # --  LN3-LN4 synapse
    ln3_ln4_output = neuralnodes.func.synapse(
        ln3_output, p.ln3_ln4_delay, p.ln3_ln4_gain
    )

    # --  LN4 is feature detector
    p.ln4_nonlinearity = (
        lambda x=None: NL.relu(x, p.ln4_nonlinearity_thres) * p.ln4_nonlinearity_gain
    )
    ln4_output = p.ln4_nonlinearity(
        neuralnodes.func.summate([ln2_ln4_output, ln3_ln4_output])
    )

    if return_dict:
        out = {
            "stim": stim,
            "an1": an1_output,
            "ln2": ln2_output,
            "ln5_ada": ln5_ada_output,
            "ln5": ln5_output,
            "ln3_sub": ln3_sub_output,
            "ln3": ln3_output,
            "ln4": ln4_output,
            "p": p,
        }
    else:
        if verbose_output:
            out = (
                an1_output,
                ln2_output,
                ln5_ada_output,
                ln5_output,
                ln3_sub_output,
                ln3_output,
                ln2_ln4_output,
                ln4_output,
                p,
            )
        else:
            out = (an1_output, ln2_output, ln5_output, ln3_output, ln4_output)
    return out
