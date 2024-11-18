# NeuronUtils

Personal utilities to use NEURON simulator.

## ProgressManager.py

Wrapper for nrnmpi and tqdm progress bar.

You don't need to prepare `pc = ParallelContext()` for simulation.
Rather, you can refer `pm.pc` for `ParallelContext()`.

You can also refer `pm.rank` and `pm.size` for MPI.

Usage:
```
from NeuronUtils import ProgressManager
with ProgressManager() as pm:
    # preparation
    pm.execution(tstop)
    # output
```

## update_requirements.sh

Generate `requirements.txt` for pip and `environment.yaml` for conda with current Poetry environment.
Obviously, poetry is required.

Usage: `NeuronUtils/update_requirements.sh` command from the parent repository, or `./update_requrements.sh` from this repository.

## Author

Ohki Katakura
