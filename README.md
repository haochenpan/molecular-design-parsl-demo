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

All runs go through `main.py` with two required flags:

```bash
python main.py --thinker <name> --config <name>
```

### Local (macOS / Linux desktop)

No Redis needed — `--config local` uses in-process `PipeQueues`.

```bash
# Random molecule evaluation (~4 min for 4 molecules)
python main.py --thinker random --config local

# ML-steered batched optimization (~5 min for 8 molecules)
python main.py --thinker batched --config local --search-count 8
```

### HPC (Midway)

On Midway `--config midway` uses `RedisQueues` + `SlurmProvider`.
You must start a Redis server before running.

**1. Start Redis** (in a separate terminal or background):
```bash
redis-server redis.conf --port 6379 &
```

**2. Run:**
```bash
python main.py --thinker random --config midway
python main.py --thinker batched --config midway --search-count 8
```

**3. Stop Redis** when finished:
```bash
redis-cli shutdown
```

> **Tip:** Use the `--redis-host` and `--redis-port` CLI flags to use a
> non-default Redis host/port.

Results are saved to `run-data/`. Clean up Parsl state between runs with `rm -rf runinfo`.

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--thinker` | *(required)* | Strategy: `random` or `batched` |
| `--config` | *(required)* | Platform config: `local` or `midway` |
| `--n-workers` | min(4, cpu_count) | Number of parallel workers |
| `--search-count` | 4 | Molecules to evaluate |
| `--initial-count` | 4 | Random sims before first ML training (batched only) |
| `--batch-size` | 2 | Simulations between retraining (batched only) |
| `--data-file` | `data/QM9-search.tsv` | Path to search space TSV |
| `--output-dir` | `run-data` | Output directory |
| `--redis-host` | `localhost` | Redis host (midway config only) |
| `--redis-port` | `6379` | Redis port (midway config only) |

Increase `--search-count` for larger experiments.  Each XTB simulation takes
30–130 seconds depending on molecule size and CPU.

## Thinker Strategies

### `random`

Picks molecules at random from the QM9 search space and submits XTB quantum
chemistry calculations (`compute_vertical` — ionization potential) in parallel
via Colmena + Parsl.

### `batched`

Adds ML steering on top of simulation:
1. Simulates `initial_count` random molecules first.
2. Trains a KNN model (k=4, Jaccard distance on Morgan fingerprints) to predict
   ionization energy.
3. Uses the model to re-rank the remaining ~130k molecules, prioritizing
   high-IE candidates.
4. Repeats training/inference every `batch_size` completed simulations.

> **Note:** `--initial-count` must be ≥ 4 because the KNN model uses
> `n_neighbors=4` and needs at least that many training samples.

## Project Layout

| File | Purpose |
|---|---|
| `main.py` | Unified CLI entry point |
| `thinkers.py` | `RandomThinker`, `BatchedThinker` |
| `configs.py` | Named platform configs, `make_parsl_config`, `make_queues` |
| `chemfunctions.py` | `compute_vertical`, `train_model`, `run_model` |
| `redis.conf` | Minimal Redis config (no persistence) for HPC runs |

## Adding a New Config

To add a new platform (e.g. `polaris`), edit [configs.py](configs.py):

1. Add a queue factory: `_make_polaris_queues(topics, **kwargs)`
2. Add a Parsl config factory: `_make_polaris_parsl_config(n_workers)`
3. Register both in the `QUEUE_CONFIGS` and `PARSL_CONFIGS` dicts

Then run with `--config polaris`.

## Cross-Platform Details

| Aspect | `local` | `midway` |
|---|---|---|
| Executor | `ThreadPoolExecutor` | `HighThroughputExecutor` + `SlurmProvider` |
| Queue backend | `PipeQueues` (in-process) | `RedisQueues` (cross-node via Redis) |
| Task server | Daemon thread (unified across platforms) | Daemon thread (unified across platforms) |

> **Why a thread (not a subprocess) for the task server?** macOS defaults
> to the `spawn` multiprocessing start method, which can't pickle the task
> server's lock objects.  A daemon thread avoids this and works identically
> on both macOS and Linux.

> **Why RedisQueues on HPC?** On clusters the thinker runs on the login
> node while Parsl dispatches work to compute nodes.  `PipeQueues` only works
> within a single process — `RedisQueues` provides a network-accessible message
> broker that bridges this gap.
