#!/usr/bin/env python
"""
ProgressManager.py

Wrapper for nrnmpi and tqdm progress bar.
The parent code should be execute with mpirun, then you can immediately abort with ctrl-c.

Usage:
```
from NeuronUtils import ProgressManager
with ProgressManager() as pm:
    # preparation
    pm.execution
    # output
```

"""

import time

import numpy as np
from neuron import h
from tqdm.auto import tqdm


class ProgressManager:
    def __init__(self, pc=None, tstop=1.0, tstep=1.0, pstep=None):
        """
        Initialise ProgressManager object
        """
        if pc is None:
            h.nrnmpi_init()
            pc = h.ParallelContext()
        self.pc = pc
        self.rank = self.pc.id()
        self.size = self.pc.nhost()
        self.tstop = tstop  # ms, total simulation time
        self.tstep = tstep  # ms, simulation time per update
        self.fih = h.FInitializeHandler(2, self.update)
        if pstep is None:
            self.pstep = tstep  # ms, same as time step
        else:
            self.pstep = pstep  # ms, updating time for progress bar
        self.cvode = h.CVode()
        self.cvode.active(False)  # Disable variable time step
        self.pc.barrier()

    def update(self):
        """
        Progress bar update
        """
        if self.rank == 0:
            tnow = np.round(h.t, 4)
            self.pbar.update(np.round(tnow - self.pbar.n, 4))
            self.cvode.event(np.round(h.t + self.pstep, 4), self.update)

    def refresh(self, total=None, desc=None):
        """
        Refresh progress bar. Max steps and/or description can also be modified.
        """
        if self.rank == 0:
            if total is not None:
                self.pbar.total = total
            if desc is not None:
                self.pbar.desc = desc
            self.pbar.refresh()
        self.pc.barrier()

    def initialise(self, tstop=None, v=None, secondorder=2, maxstep=None, desc=None):
        """
        Initialise NEURON simulation. Execute before pm.run().
        """
        self.pc.barrier()
        h.secondorder = secondorder
        if not self.pc.gid_exists(self.rank):
            self.pc.set_gid2node(self.rank, self.rank)
        h.load_file("stdrun.hoc")
        if tstop is None:
            h.tstop = self.tstop
        else:
            h.tstop = tstop
            self.tstop = tstop
        h.dt = self.tstep
        if self.rank == 0:
            self.pbar = tqdm(
                bar_format="{l_bar}{bar}| {n_fmt:.05}/{total_fmt} [{elapsed}<{remaining}, {postfix}{rate_fmt}]",
                total=h.tstop,
                desc=desc,
            )
        if v is None:
            h.finitialize()
        else:
            h.finitialize(v)
        h.fcurrent()
        if maxstep is None:
            self.pc.set_maxstep(1)
        else:
            self.pc.set_maxstep(maxstep)
        self.update()
        self.pc.barrier()

    def run(self, tstop=None):
        """
        Execute NEURON simulation.
        """
        if tstop is None or tstop == h.tstop:
            self.pc.psolve(h.tstop)
        else:
            self.refresh(total=tstop)
            self.pc.psolve(tstop)

    def finalise(self):
        """
        Finalise NEURON simulation. Execute after pm.run().
        """
        self.pc.barrier()
        if self.rank == 0:
            self.pbar.close()
        self.pc.barrier()

    def execute(self, tstop=None, v=None, maxstep=None, desc=None):
        """
        Wrapper for execution: initialise, run, and finalise.
        """
        self.initialise(tstop=tstop, v=v, maxstep=maxstep, desc=desc)
        self.run(tstop=tstop)
        self.finalise()

    def quit(self):
        """
        Close whole script
        """
        self.pc.barrier()
        self.pc.done()
        h.quit()

    def __enter__(self, close=True):
        """
        Enable with block and set flag to quit the script at the end of with block
        """
        self.close_script = close
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        If the close flag is True, the script is closed.
        """
        """Context manager exit point."""
        if exc_type is not None:
            print(f"Exception occurred: {exc_value}")
        if self.rank == 0 and hasattr(self, "pbar"):
            self.pbar.close()
        self.pc.barrier()
        if self.close_script:
            self.quit()
        return False


if __name__ == "__main__":
    with ProgressManager(tstop=1000) as pm:
        # Preparation of cells
        N = 10000
        somatae = [h.Section(name=f"soma[{i}]") for i in range(pm.rank, N, pm.size)]
        stim = []
        for i, soma in enumerate(somatae):
            soma.insert("hh")
            iclamp = h.IClamp(soma(0.5))
            iclamp.delay = 100  # ms
            iclamp.dur = 900  # ms
            iclamp.amp = i / 100  # nA
            stim.append(iclamp)

        # Preparation of recording
        t = h.Vector().record(h._ref_t)
        v = [
            h.Vector().record(somatae[0](0.5)._ref_v),
            h.Vector().record(somatae[-1](0.5)._ref_v),
        ]

        if pm.pc.id() == 0:
            print(f"t = {h.t:.1f} ms, v = {somatae[0].v:.1f} mV")
        t0 = time.time()

        # Simulation
        pm.execute()

        # Results of recording
        if pm.pc.id() == 0:
            print(f"{time.time() - t0:11.6f} s")
            x = np.vstack([t, v[0], v[1]]).T
            x = np.round(x[:: np.round(1 / h.dt).astype(int)], 1)
            print(x[:10])
            print(x[-10:])
