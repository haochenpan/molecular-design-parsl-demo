# Standalone Example Scripts

Two standalone Python scripts adapted from the Jupyter notebook
[`1_interleaving-simulation-and-steering.ipynb`](1_interleaving-simulation-and-steering.ipynb),
updated for the latest libraries and cross-platform (macOS + Linux) support.

## Setup

On most systems:
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

```bash
# Example 1: Random molecule evaluation (~4 min for 4 molecules on a laptop)
python example1_random.py

# Example 2: ML-steered batched optimization (~5 min for 8 molecules on a laptop)
python example2_batched.py
```

Results are saved to `run-data/`. Clean up Parsl state between runs with `rm -rf runinfo`.

## What Each Script Does

### `example1_random.py`

Picks molecules at random from the QM9 search space and submits XTB quantum chemistry
calculations (`compute_vertical` — ionization potential) in parallel via Colmena + Parsl.
Results are collected and saved to `run-data/random-results.json`.

### `example2_batched.py`

Adds ML steering on top of simulation:
1. Simulates `initial_count` random molecules first.
2. Trains a KNN model (k=4, Jaccard distance on Morgan fingerprints) to predict ionization energy.
3. Uses the model to re-rank the remaining ~130k molecules, prioritizing high-IE candidates.
4. Repeats training/inference every `batch_size` completed simulations.
5. Saves results to `run-data/batched-results.json`.

> **Note:** `initial_count` must be ≥ 4 because the KNN model uses `n_neighbors=4` and
> needs at least that many training samples.

## Configuration

| Parameter | `example1_random.py` | `example2_batched.py` |
|---|---|---|
| `search_count` | 4 | 8 |
| `initial_count` | — | 4 |
| `batch_size` | — | 2 |
| `n_workers` | min(4, cpu_count) | min(4, cpu_count) |

Increase `search_count` for larger experiments. Each XTB simulation takes 30–130 seconds
depending on molecule size and CPU.

## Project Layout

| File | Purpose |
|---|---|
| `example1_random.py` | Entry point — random evaluation |
| `example2_batched.py` | Entry point — ML-steered batched optimization |
| `thinkers.py` | `RandomThinker`, `BatchedThinker` (Jupyter), `StandaloneBatchedThinker` |
| `configs.py` | `make_parsl_config`, `start_task_server`, `stop_task_server` |
| `chemfunctions.py` | `compute_vertical`, `train_model`, `run_model` |

## Changes from Original Notebook

### API / Library Updates

| Change | Original | Updated | Why |
|---|---|---|---|
| Saving results | `result.json()` | `result.model_dump_json()` | Pydantic V2 |
| Reading TSV | `delim_whitespace=True` | `sep=r'\s+'` | pandas deprecation |
| Parsl executor | `max_workers=N` | `max_workers_per_node=N` | Parsl 2026.x |
| numpy stacking | `np.vstack(iterator)` | `np.vstack(list(iterator))` | numpy ≥ 2.x |
| Morgan fingerprints | `GetMorganFingerprintAsBitVect` | `rdFingerprintGenerator.GetMorganGenerator` | RDKit deprecation |
| zip → list | `send_inputs(smiles, ie, ...)` | `send_inputs(list(smiles), list(ie), ...)` | `zip()` returns tuples; numpy/pickle rejects them |

### Cross-Platform Compatibility

The original notebook only ran on Linux. These scripts run on **both macOS and Linux**:

| Aspect | Original (Linux-only) | Cross-platform Fix |
|---|---|---|
| CPU detection | `os.sched_getaffinity(0)` | try/except fallback to `os.cpu_count()` |
| Executor | `HighThroughputExecutor` | `ThreadPoolExecutor` on macOS, HTEX on Linux |
| CPU affinity | `cpu_affinity='block'` | Only set on Linux |
| Task server | `task_server.start()` (subprocess) | Thread on macOS, subprocess on Linux |

> **Why ThreadPoolExecutor on macOS?** macOS defaults to the `spawn` multiprocessing start method,
> which can't pickle the task server's lock objects. Using `fork` instead corrupts Jupyter's ZMQ
> sockets. `ThreadPoolExecutor` avoids multiprocessing entirely and runs tasks in-process.

### Structural Changes

| Aspect | Original Notebook | Standalone Scripts |
|---|---|---|
| Progress display | `tqdm.notebook` progress bars | Console `print()` statements |
| Dashboard | `ipywidgets.Output` + HTML | Console `print()` statements |
| Queue backend | `PipeQueues` | `PipeQueues` (same) |
| Thinker classes | Defined in `thinkers.py` | Imported from `thinkers.py` |
| Config helpers | Inline | Extracted to `configs.py` |
| Dependencies | Jupyter, ipywidgets, tqdm | None beyond core libs |
