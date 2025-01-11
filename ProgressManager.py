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

import sys
import time

import numpy as np
from neuron import h
from tqdm.auto import tqdm


class altpbar:
    """
    Alternative progression bar for the environment that tqdm does not work (`not sys.stdout.isatty()`).
    This class is generated with ChatGPT 4o.
    """

    def __init__(self, total, desc=None):
        """
        Initialize an alternative progress bar.

        Parameters:
        total (int): Total number of steps.
        desc (str): Description of the progress bar.
        """
        self.total = total
        self.desc = f"{desc}: " if desc else ""
        self.n = 0
        self.start_time = time.time()
        self.width = len(str(total))

    def update(self, step=1):
        """
        Update the progress bar by a given step.

        Parameters:
        step (int): Number of steps to increment. Default is 1.
        """
        self.n += step
        elapsed_time = time.time() - self.start_time

        # Estimated remaining time
        if self.n > 0:
            rate = elapsed_time / self.n
            eta = rate * (self.total - self.n)
            rate_fmt = f"{1 / rate:.2f} it/s"
        else:
            eta = float("inf")
            rate_fmt = "N/A"

        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        eta_str = (
            time.strftime("%H:%M:%S", time.gmtime(eta)) if eta < float("inf") else "?"
        )

        # Format current and total to match width with space padding
        current = str(self.n).rjust(self.width)
        total = str(self.total).rjust(self.width)

        # Print progress
        print(
            f"{self.desc}{current}/{total} ({self.n / self.total:7.2%}) [{elapsed_str}<{eta_str}, {rate_fmt}]",
            flush=True,
            file=sys.stderr,
        )

    def refresh(self, total=None, desc=None):
        """
        Refresh the total steps or description.

        Parameters:
        total (int): New total steps. Optional.
        desc (str): New description. Optional.
        """
        if total is not None:
            self.total = total
            self.width = len(str(total))
        if desc is not None:
            self.desc = desc

    def close(self):
        """
        Finalize the progress bar.
        """
        elapsed_time = time.time() - self.start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print(
            f"{self.desc}Completed {str(self.n).rjust(self.width)}/{str(self.total).rjust(self.width)} in {elapsed_str}.",
            flush=True,
        )


class ProgressManager:
    def __init__(self, pc=None, tstop=1.0, tstep=1.0, secondorder=2, pstep=None):
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
        # self.fih = h.FInitializeHandler(2, self.update) # Avoid double schedule
        if pstep is None:
            self.pstep = tstep  # ms, same as time step
        else:
            self.pstep = pstep  # ms, updating time for progress bar
        self.cvode = h.CVode()
        self.cvode.active(False)  # Disable variable time step
        h.secondorder = secondorder
        self.pc.barrier()

    def update(self):
        """
        Progress bar update
        """
        if self.rank == 0:
            tnow = np.round(h.t, 4)
            self.pbar.update(np.round(tnow - self.pbar.n, 4))
            tnext = np.round(h.t + self.pstep, 4)
            if tnext <= h.tstop:
                self.cvode.event(tnext, self.update)

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

    def initialise(self, tstop=None, v=None, secondorder=None, maxstep=None, desc=None):
        """
        Initialise NEURON simulation. Execute before pm.run().
        """
        self.pc.barrier()
        if secondorder is not None:
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
            if sys.stderr.isatty():
                self.pbar = tqdm(
                    bar_format="{l_bar}{bar}| {n_fmt:.05}/{total_fmt} [{elapsed}<{remaining}, {postfix}{rate_fmt}]",
                    total=h.tstop,
                    desc=desc,
                )
            else:
                self.pbar = altpbar(total=h.tstop, desc=desc)
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
        if tstop is not None and tstop != h.tstop:
            h.tstop = tstop
            self.refresh(total=tstop)
        self.pc.psolve(h.tstop)
        print(f"{self.rank}th thread is completed.")

    def finalise(self):
        """
        Finalise NEURON simulation. Execute after pm.run().
        """
        self.pc.barrier()
        if self.rank == 0:
            self.pbar.close()
        self.pc.barrier()

    def execute(self, tstop=None, v=None, secondorder=None, maxstep=None, desc=None):
        """
        Wrapper for execution: initialise, run, and finalise.
        """
        self.initialise(
            tstop=tstop, v=v, secondorder=secondorder, maxstep=maxstep, desc=desc
        )
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
