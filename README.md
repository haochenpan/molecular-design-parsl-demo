# ML-in-the-loop Molecular Design with Parsl

This repository demonstrates how [Parsl](https://parsl-project.org/) and
[Colmena](https://colmena.readthedocs.io/) can drive a machine-learning-guided
search for molecules with high ionization energy (IE).

**Objective:** identify molecules with the largest IE from the QM9 search space
(~130k molecules).  IE is computed with
[xTB](https://xtb-docs.readthedocs.io/en/latest/contents.html); because each
simulation is expensive, we use active learning to prioritize the most
promising candidates.

## Setup

On most systems (macOS / local Linux):
```bash
conda env create -f environment.yml
conda activate moldesign-demo
pip install -e .
```

If you are running on the UChicago RCC (Midway) cluster:
```bash
module load python/miniforge-25.3.0
mamba env create --prefix ./env -f environment.yml
source activate ./env
pip install -e .
```

## Running

### macOS (local)

No Redis needed — the scripts use in-process `PipeQueues` automatically.

```bash
# Example 1: Random molecule evaluation (~4 min for 4 molecules)
python example1_random.py

# Example 2: ML-steered batched optimization (~5 min for 8 molecules)
python example2_batched.py
```

### Linux / HPC (Midway)

On Linux the scripts use `RedisQueues` so the thinker (login node) can
communicate with Parsl workers (compute nodes) across the network.
You must start a Redis server before running the examples.

**1. Start Redis** (in a separate terminal or background):
```bash
redis-server redis.conf --port 6379 &
```

The included `redis.conf` disables persistence (`appendonly no`, `save ''`)
since we only need Redis as an ephemeral message broker.

**2. Run the script:**
```bash
# Example 1
python example1_random.py

# Example 2
python example2_batched.py
```

**3. Stop Redis** when finished:
```bash
redis-cli shutdown
```

> **Tip:** If you want to use a non-default Redis host/port, set the
> `REDIS_HOST` and `REDIS_PORT` environment variables before running.

Results are saved to `run-data/`. Clean up Parsl state between runs with `rm -rf runinfo`.

## What Each Script Does

### `example1_random.py`

Picks molecules at random from the QM9 search space and submits XTB quantum
chemistry calculations (`compute_vertical` — ionization potential) in parallel
via Colmena + Parsl.  Results are saved to `run-data/random-results.json`.

### `example2_batched.py`

Adds ML steering on top of simulation:
1. Simulates `initial_count` random molecules first.
2. Trains a KNN model (k=4, Jaccard distance on Morgan fingerprints) to predict
   ionization energy.
3. Uses the model to re-rank the remaining ~130k molecules, prioritizing
   high-IE candidates.
4. Repeats training/inference every `batch_size` completed simulations.
5. Saves results to `run-data/batched-results.json`.

> **Note:** `initial_count` must be ≥ 4 because the KNN model uses
> `n_neighbors=4` and needs at least that many training samples.

## Configuration

| Parameter | `example1_random.py` | `example2_batched.py` |
|---|---|---|
| `search_count` | 4 | 8 |
| `initial_count` | — | 4 |
| `batch_size` | — | 2 |
| `n_workers` | min(4, cpu_count) | min(4, cpu_count) |

Increase `search_count` for larger experiments.  Each XTB simulation takes
30–130 seconds depending on molecule size and CPU.

## Project Layout

| File | Purpose |
|---|---|
| `example1_random.py` | Entry point — random evaluation |
| `example2_batched.py` | Entry point — ML-steered batched optimization |
| `thinkers.py` | `RandomThinker`, `StandaloneBatchedThinker` |
| `configs.py` | `make_parsl_config`, `make_queues`, `start_task_server`, `stop_task_server` |
| `chemfunctions.py` | `compute_vertical`, `train_model`, `run_model` |
| `redis.conf` | Minimal Redis config (no persistence) for HPC runs |

## Cross-Platform Details

| Aspect | macOS | Linux / HPC |
|---|---|---|
| Executor | `ThreadPoolExecutor` | `HighThroughputExecutor` + `SlurmProvider` |
| Queue backend | `PipeQueues` (in-process) | `RedisQueues` (cross-node via Redis) |
| Task server | Thread (avoids fork + ZMQ issues) | Subprocess |

> **Why ThreadPoolExecutor on macOS?** macOS defaults to the `spawn`
> multiprocessing start method, which can't pickle the task server's lock
> objects.  `ThreadPoolExecutor` avoids multiprocessing entirely.

> **Why RedisQueues on Linux?** On HPC clusters the thinker runs on the login
> node while Parsl dispatches work to compute nodes.  `PipeQueues` only works
> within a single process — `RedisQueues` provides a network-accessible message
> broker that bridges this gap.
