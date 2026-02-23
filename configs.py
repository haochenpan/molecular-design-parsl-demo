"""Parsl configuration helpers for standalone Colmena examples.

Provides:
  - make_parsl_config: Build a Parsl Config that auto-detects the platform:
      • macOS  → ThreadPoolExecutor  (avoids fork / ZMQ pickling issues)
      • Linux  → HighThroughputExecutor + SlurmProvider  (Midway3 cluster)
  - start_task_server / stop_task_server: Lifecycle helpers that work on
    both macOS (thread) and Linux (subprocess).
"""

import os
import platform
from threading import Thread

from colmena.task_server.parsl import ParslTaskServer
from parsl import Config


def make_parsl_config(n_workers: int) -> Config:
    """Create a Parsl Config appropriate for the current platform.

    macOS:  ThreadPoolExecutor — runs tasks in-process on threads.
            This avoids ``spawn``-mode pickling errors and ZMQ fork issues.

    Linux:  HighThroughputExecutor backed by SlurmProvider, targeting the
            Midway3 *caslake* partition.  The active virtualenv (if any) is
            forwarded to Slurm workers automatically.
    """
    if platform.system() == 'Darwin':
        from parsl.executors import ThreadPoolExecutor

        return Config(
            executors=[
                ThreadPoolExecutor(
                    label='local_threads',
                    max_threads=n_workers,
                )
            ]
        )

    # Linux — Midway3 / Slurm
    from parsl.addresses import address_by_hostname
    from parsl.executors import HighThroughputExecutor
    from parsl.launchers import SrunLauncher
    from parsl.providers import SlurmProvider

    venv = os.environ.get("VIRTUAL_ENV")
    worker_init_parts = [
        "export TMPDIR=/tmp",
        "export TEMP=/tmp",
        "export TMP=/tmp",
    ]
    if venv:
        worker_init_parts.insert(0, f"source {venv}/bin/activate")
    worker_init = "; ".join(worker_init_parts)

    return Config(
        executors=[
            HighThroughputExecutor(
                label="midway3_htex",
                provider=SlurmProvider(
                    partition="caslake",
                    account="pi-chard",
                    nodes_per_block=1,
                    init_blocks=1,
                    min_blocks=1,
                    max_blocks=1,
                    walltime="00:15:00",
                    worker_init=worker_init,
                    exclusive=False,
                    launcher=SrunLauncher(),
                ),
                address=address_by_hostname(),
                worker_debug=True,
                max_workers_per_node=n_workers,
            )
        ]
    )


def start_task_server(task_server: ParslTaskServer):
    """Start the task server in a way that works on both macOS and Linux."""
    if platform.system() == 'Darwin':
        t = Thread(target=task_server.run, daemon=True)
        t.start()
        return t
    else:
        task_server.start()
        return None


def stop_task_server(task_server: ParslTaskServer, server_thread):
    """Stop the task server cleanly."""
    if server_thread is not None:
        server_thread.join(timeout=30)
    else:
        task_server.join()
        print(f'Process exited with {task_server.exitcode} code')
