import neuralnodes.params


@neuralnodes.params.jax_dataclass
class Params(neuralnodes.params.Params):
    an1_filter_inh_dur: float = 183.803241
    an1_filter_inh_sigma: float = 2.185188
    an1_filter_inh_gain: float = 0.100389
    an1_filter_delay: float = 7.845125
    an1_filter_exc_dur: float = 9.566566
    an1_filter_exc_sigma: float = 0.457490
    an1_ada_filter_tau: float = 3763.272297
    an1_ada_strength: float = 4.0  # 10.0
    an1_nonlinearity_slope: float = 1.500000
    an1_nonlinearity_shift: float = 1.500000
    an1_nonlinearity_gain: float = 5.000000
    an1_nonlinearity_baseline: float = -0.500000
    an1_ln2_delay: float = 0.0
    an1_ln2_gain: float = 1.0
    ln2_filter_inh_tau: float = 5.236091
    ln2_filter_exc_dur: float = 14.230023
    ln2_filter_exc_sigma: float = 0.611233
    ln2_filter_exc_gain: float = 0.259168
    ln2_nonlinearity_gain: float = 1.162401
    ln2_ln5_gain: float = -0.006702
    ln2_ln5_delay: float = 8.668116
    ln5_ada_filter_dur: float = 4.972162
    ln5_ada_filter_exc_gain: float = 1.0  # 1.110563
    ln5_filter_exc_tau: float = 3.305648
    ln5_filter_exc_dur: float = 20.680255
    ln5_filter_exc_gain: float = 914.750505
    ln5_filter_inh_tau: float = 30.308115
    ln5_filter_inh_gain: float = 1718.351292
    ln5_nonlinearity_gain: float = 0.531748
    an1_ln3_delay: float = 7.187678
    an1_ln3_gain: float = 36.000785
    ln5_ln3_delay: float = 2.001966
    ln5_ln3_gain: float = 21.624769
    ln3_sub_nonlinearity_thres: float = 0.081156
    ln3_sub_nonlinearity_gain: float = 0.012875
    ln3_ada_filter_tau: float = 49.342662
    ln3_ada_strength: float = 0.241336
    ln3_nonlinearity_thres: float = 2.485875
    ln3_nonlinearity_gain: float = 211.359865
    ln2_ln4_delay: float = 16.408722
    ln2_ln4_gain: float = 0  #-546.784527  # NO INHIBIBTION FROM LN2 to LN4
    ln3_ln4_delay: float = 4.368713
    ln3_ln4_gain: float = 9.580906
    ln4_nonlinearity_thres: float = 1123.634900
    ln4_nonlinearity_gain: float = 0.002190
