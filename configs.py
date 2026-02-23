"""Parsl configuration helpers for Colmena molecular design examples.

Provides named configurations for different platforms:
  - "local"  → ThreadPoolExecutor + PipeQueues (no Redis needed)
  - "midway" → HighThroughputExecutor + SlurmProvider + RedisQueues

Usage:
  queues = make_queues("local", topics=["simulate", "train", "infer"])
  config = make_parsl_config("local", n_workers=4)
"""

import os
from threading import Thread
from typing import List

from colmena.queue.base import ColmenaQueues
from colmena.task_server.parsl import ParslTaskServer
from parsl import Config


# ---------------------------------------------------------------------------
# Queue factory functions
# ---------------------------------------------------------------------------

def _make_local_queues(topics, **kwargs):
    from colmena.queue.python import PipeQueues
    return PipeQueues(topics=topics, serialization_method='pickle')


def _make_midway_queues(topics, redis_host='localhost', redis_port=6379, **kwargs):
    from colmena.queue.redis import RedisQueues
    return RedisQueues(hostname=redis_host, port=redis_port, topics=topics)


QUEUE_CONFIGS = {
    'local': _make_local_queues,
    'midway': _make_midway_queues,
}


# ---------------------------------------------------------------------------
# Parsl config factory functions
# ---------------------------------------------------------------------------

def _make_local_parsl_config(n_workers):
    from parsl.executors import ThreadPoolExecutor
    return Config(
        executors=[
            ThreadPoolExecutor(
                label='local_threads',
                max_threads=n_workers,
            )
        ]
    )


def _make_midway_parsl_config(n_workers):
    from parsl.addresses import address_by_hostname
    from parsl.executors import HighThroughputExecutor
    from parsl.launchers import SrunLauncher
    from parsl.providers import SlurmProvider

    worker_init_parts = [
        "module load python/miniforge-25.3.0",
    ]

    conda_prefix = os.environ.get("CONDA_PREFIX")
    venv = os.environ.get("VIRTUAL_ENV")
    if conda_prefix:
        worker_init_parts.append(f"source activate {conda_prefix}")
    elif venv:
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


PARSL_CONFIGS = {
    'local': _make_local_parsl_config,
    'midway': _make_midway_parsl_config,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_queues(config_name: str, topics: List[str], **kwargs) -> ColmenaQueues:
    """Create Colmena queues for the given configuration.

    Args:
        config_name: Configuration name (e.g. "local", "midway").
        topics: Queue topic names (e.g. ['simulate', 'train', 'infer']).
        **kwargs: Extra arguments forwarded to the queue factory
                  (e.g. redis_host, redis_port for midway).
    """
    if config_name not in QUEUE_CONFIGS:
        raise ValueError(
            f"Unknown config '{config_name}'. Available: {sorted(QUEUE_CONFIGS.keys())}"
        )
    return QUEUE_CONFIGS[config_name](topics, **kwargs)


def make_parsl_config(config_name: str, n_workers: int) -> Config:
    """Create a Parsl Config for the given configuration.

    Args:
        config_name: Configuration name (e.g. "local", "midway").
        n_workers: Number of parallel workers.
    """
    if config_name not in PARSL_CONFIGS:
        raise ValueError(
            f"Unknown config '{config_name}'. Available: {sorted(PARSL_CONFIGS.keys())}"
        )
    return PARSL_CONFIGS[config_name](n_workers)


def start_task_server(task_server: ParslTaskServer) -> Thread:
    """Start the task server in a daemon thread.

    Uses a thread instead of a subprocess so the same code works on macOS
    (where ``spawn`` multiprocessing cannot pickle the task server) and Linux.
    """
    t = Thread(target=task_server.run, daemon=True)
    t.start()
    return t


def stop_task_server(server_thread: Thread):
    """Wait for the task-server thread to finish."""
    server_thread.join(timeout=30)
