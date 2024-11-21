#!/usr/bin/env python
"""
Enable tqdm with verbose option.

Usage:
```
from NeuronUtils import vtqdm

# turn on progress bar
print("turn on tqdm")
for i in vtqdm(range(5), verbose=True):
    vtqdm.write(f"{i}")

# turn off progress bar
print("turn off tqdm")
for i in vtqdm(range(5), verbose=False):
    vtqdm.write(f"{i}")
```
"""

from tqdm.auto import tqdm


class vtqdm:
    def __init__(self, iterable, verbose=False, **kwargs):
        """
        A wrapper around tqdm to provide optional verbosity.

        Parameters:
        - iterable: The iterable to wrap.
        - verbose: Whether to enable tqdm progress bar.
        - **kwargs: Additional keyword arguments to pass to tqdm.
        """
        if not hasattr(iterable, "__iter__"):
            raise TypeError("First argument must be an iterable.")
        if not isinstance(verbose, bool):
            raise TypeError("Second argument 'verbose' must be a boolean.")

        self.verbose = verbose
        self.is_async = hasattr(iterable, "__aiter__")  # no sync iterator

        if verbose:
            if self.is_async:
                self.iterator = tqdm(iterable, **kwargs)
            else:
                self.iterator = iter(tqdm(iterable, **kwargs))
        else:
            if self.is_async:
                self.iterator = iterable.__aiter__()
            else:
                self.iterator = iter(iterable)

    def __iter__(self):
        if self.is_async:
            raise TypeError("Use 'async for' with asynchronous iterables.")
        return self

    def __next__(self):
        if self.is_async:
            raise TypeError("Use 'async for' with asynchronous iterables.")
        return next(self.iterator)

    def __aiter__(self):
        if not self.is_async:
            raise TypeError("Use 'for' with synchronous iterables.")
        return self.iterator

    async def __anext__(self):
        if not self.is_async:
            raise TypeError("Use 'for' with synchronous iterables.")
        return await self.iterator.__anext__()

    @staticmethod
    def write(*args, **kwargs):
        tqdm.write(*args, **kwargs)


if __name__ == "__main__":
    # turn on progress bar
    print("turn on tqdm")
    for i in vtqdm(range(5), verbose=True):
        vtqdm.write(f"{i}")

    # turn off progress bar
    print("turn off tqdm")
    for i in vtqdm(range(5), verbose=False):
        vtqdm.write(f"{i}")
