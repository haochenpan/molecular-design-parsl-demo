"""Parsl configuration helpers for standalone Colmena examples.

Provides:
  - make_parsl_config: Build a Parsl Config that auto-detects the platform:
      • macOS  → ThreadPoolExecutor  (avoids fork / ZMQ pickling issues)
      • Linux  → HighThroughputExecutor + SlurmProvider  (Midway3 cluster)
  - make_queues: Build a Colmena queue that auto-detects the platform:
      • macOS  → PipeQueues  (in-process, no Redis needed)
      • Linux  → RedisQueues (cross-node communication via Redis)
  - start_task_server / stop_task_server: Lifecycle helpers that work on
    both macOS (thread) and Linux (subprocess).
"""

import os
import platform
from threading import Thread
from typing import List

from colmena.queue.base import ColmenaQueues
from colmena.task_server.parsl import ParslTaskServer
from parsl import Config


def make_queues(topics: List[str], redis_host: str = None, redis_port: int = None) -> ColmenaQueues:
    """Create the appropriate Colmena queue for the current platform.

    macOS:  PipeQueues — in-process Python pipes. No Redis server needed.

    Linux:  RedisQueues — uses a Redis server to communicate between the
            login node (where the thinker runs) and compute nodes (where
            Parsl workers execute tasks).  Requires a running Redis server.

    Args:
        topics: Queue topic names (e.g. ['simulate', 'train', 'infer']).
        redis_host: Hostname of the Redis server (Linux only).
                    Defaults to REDIS_HOST env var, or 'localhost'.
        redis_port: Port of the Redis server (Linux only).
                    Defaults to REDIS_PORT env var, or 6379.
    """
    if platform.system() == 'Darwin':
        from colmena.queue.python import PipeQueues
        return PipeQueues(topics=topics, serialization_method='pickle')

    if redis_host is None:
        redis_host = os.environ.get('REDIS_HOST', 'localhost')
    if redis_port is None:
        redis_port = int(os.environ.get('REDIS_PORT', '6379'))

    from colmena.queue.redis import RedisQueues
    return RedisQueues(
        hostname=redis_host,
        port=redis_port,
        topics=topics,
    )


def make_parsl_config(n_workers: int) -> Config:
    """Create a Parsl Config appropriate for the current platform.

    macOS:  ThreadPoolExecutor — runs tasks in-process on threads.
            This avoids ``spawn``-mode pickling errors and ZMQ fork issues.

    Linux:  HighThroughputExecutor backed by SlurmProvider, targeting the
            Midway3 *caslake* partition.  The active conda env or virtualenv
            is forwarded to Slurm workers automatically.
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

    # Build worker_init to replicate the login-node environment on compute nodes.
    # Order matters: module load → activate env → set env vars.
    worker_init_parts = [
        "module load python/miniforge-25.3.0",
    ]

    conda_prefix = os.environ.get("CONDA_PREFIX")
    venv = os.environ.get("VIRTUAL_ENV")
    if conda_prefix:
        # Conda prefix env (e.g. --prefix ./env)
        worker_init_parts.append(f"source activate {conda_prefix}")
    elif venv:
        # Standard virtualenv / venv
        worker_init_parts.append(f"source {venv}/bin/activate")

    worker_init_parts.extend([
        "export TMPDIR=/tmp",
        "export TEMP=/tmp",
        "export TMP=/tmp",
    ])
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
