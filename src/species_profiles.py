"""Original species-profile extraction and shared experimental fit targets.

The saved fields and profiles in results_data.npz are one preprocessing snapshot.
Historical fit archives contain targets from a different snapshot; never substitute
those targets for the profiles of the displayed experimental fields.
"""
from collections import defaultdict

import numpy as np
from scipy.stats import binned_statistic


cper_low = defaultdict(lambda: 250, {'BIM': 200, 'FIR': 350, 'VEL': 400, 'G15': 250, 'G13': 500, 'G14': 500, 'OAX': 150, 'PER': 200, 'RUB': 250, 'VOC': 100, 'LIN': 250})
cper_high = defaultdict(lambda: 500, {'BIM': 400, 'FIR': 550, 'VEL': 550, 'G15': 400, 'G13': 1000, 'G14': 1000, 'OAX': 220, 'PER': 350, 'RUB': 800, 'VOC': 250, 'LIN': 450})
cdc_low = defaultdict(lambda: 0.33, {'RUB': 0.9})
cdc_high = defaultdict(lambda: 0.67, {'RUB': 1.0})


def transect_slice(selector, x, lower, upper, field):
    """Original data notebook, cell 5: maxima, then 15 percentile-bin means."""
    keep = (selector > lower) & (selector < upper)
    x, y = x[keep], field[keep]
    unique_x = np.unique(x)
    y = np.array([np.nanmax(y[x == value]) for value in unique_x])
    x = unique_x
    sort_idx = np.argsort(x)
    x, y = x[sort_idx], y[sort_idx]
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    bins = np.percentile(x, np.linspace(0, 100, 16))
    ys, xs, _ = binned_statistic(x, y, bins=bins, range=(0, 1))
    xs = xs - np.diff(xs, prepend=0) / 2
    return x, y, ys, xs


def filled(values):
    """Original fitter: fill missing bins by interpolation over bin indices."""
    values = np.asarray(values, dtype=float).copy()
    missing = np.isnan(values)
    if missing.any():
        known = ~missing
        values[missing] = np.interp(np.flatnonzero(missing), np.flatnonzero(known), values[known])
    return values


def targets_from_data(data, cdc_x, cper_x):
    """Resample this dataset's profiles onto the two fitting grids."""
    targets = {
        species: (
            np.interp(cdc_x, data['cdc_transect_x'][i], filled(data['cdc_transect_y'][i])),
            np.interp(cper_x, data['cper_transect_x'][i], filled(data['cper_transect_y'][i])),
        )
        for i, species in enumerate(data['species'])
    }
    assert all(np.isfinite(curve).all() for pair in targets.values() for curve in pair)
    return targets
