"""Stable PA-STFed command-line entry point.

The implementation lives in :mod:`experiment_runtime`; this thin wrapper
keeps ``python code/run.py ...`` stable while making runtime orchestration
importable without depending on the CLI module.
"""

from experiment_runtime import main


if __name__ == "__main__":
    main()
