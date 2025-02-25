#!/usr/env python3
"""
analysis.py

Provide utilities to analyse neural data.

sample usage:
```
import NeuronUtils.analyis as nua
import numpy as np

st0 = np.random.normal(loc=10, scale=2, size=10000)
st1 = np.random.normal(loc=20, scale=5, size=15000)

print(f's/n: {nua.get_s2n(st0, st1):.6f}') # analytically about 6.9

```
"""

import numpy as np


def get_s2n(learnt: np.ndarray, novel: np.ndarray):
    """
    Calculate signal-to-noise ratio (s/n) of two data.
    """
    return (
        2 * (novel.mean() - learnt.mean()) ** 2 / (novel.std() ** 2 + learnt.std() ** 2)
    )


def get_first_sbp(
    spiketimes: np.ndarray,
    cutoff=5000,
    epoch_dur=1000,
    pause_threshold=20,
    variable_threshold=False,
):
    """
    Obtain the time of the first spike (f) and length of first burst (b) and pause (p).
    `variable_threshold` is the flag to get `pause_threshold` from spike train (not yet implemented)
    """
    cut = spiketimes[spiketimes >= cutoff]
    isi = np.diff(cut)
    epoch_ids = np.hstack([-1, cut // epoch_dur])
    first = cut[epoch_ids[1:] != epoch_ids[:-1]]

    pause_idx = isi >= pause_threshold
    _, first_pause_idx = np.unique(epoch_ids[1:-1][pause_idx], return_index=True)

    first_burst_end = cut[:-1][pause_idx][first_pause_idx]
    first_pause_end = cut[1:][pause_idx][first_pause_idx]
    burst = first_burst_end - first
    pause = first_pause_end - first_burst_end

    return first, burst, pause
