"""Example 2: Batch optimization with interleaved simulation and ML using Colmena + Parsl.

Standalone version of the "Example 2: Batch optimization with slight overlap
between simulation and ML" section from `1_interleaving-simulation-and-steering.ipynb`.

What it does:
  1. Starts by simulating `initial_count` random molecules (XTB ionization potential).
  2. Trains a KNN model on Morgan fingerprints to predict ionization energy.
  3. Uses the model to re-prioritize which molecules to simulate next.
  4. Repeats training/inference every `batch_size` completed simulations.
  5. Saves results to run-data/batched-results.json.

Usage:
  python example2_batched.py
"""

import logging
import os
import time
from time import perf_counter

import pandas as pd
from colmena.queue.python import PipeQueues
from colmena.task_server.parsl import ParslTaskServer

from chemfunctions import compute_vertical, train_model, run_model
from configs import make_parsl_config, start_task_server, stop_task_server
from thinkers import StandaloneBatchedThinker

# --- Configuration ---
n_workers = min(4, os.cpu_count())
search_count = 8        # Number of molecules to evaluate
initial_count = 4       # Simulations before first training (must be >= 4 for KNN)
batch_size = 2          # Simulations between retraining

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.WARNING,
)


def main():
    print(f'=== Example 2: Batched Optimization with ML Steering ===')
    print(f'Workers: {n_workers}, Molecules: {search_count}, '
          f'Initial: {initial_count}, Batch: {batch_size}')

    # Load data
    search_space = pd.read_csv('data/QM9-search.tsv', sep=r'\s+')
    print(f'Search space: {len(search_space)} molecules')

    # Set up Colmena
    queues = PipeQueues(topics=['simulate', 'train', 'infer'], serialization_method='pickle')
    task_server = ParslTaskServer(
        methods=[compute_vertical, train_model, run_model],
        queues=queues,
        config=make_parsl_config(n_workers),
    )

    # Start task server
    server_thread = start_task_server(task_server)
    time.sleep(2)
    print('Task server started.')

    # Run the thinker
    start = perf_counter()
    thinker = StandaloneBatchedThinker(
        queues=queues,
        n_to_evaluate=search_count,
        n_parallel=n_workers,
        initial_count=initial_count,
        batch_size=batch_size,
        molecule_list=search_space['smiles'].values,
    )
    print('Running batched thinker...')
    thinker.run()
    elapsed = perf_counter() - start
    print(f'Done in {elapsed:.1f}s. Got {len(thinker.database)} results.')

    # Save results
    os.makedirs('run-data', exist_ok=True)
    with open('run-data/batched-results.json', 'w') as fp:
        for result in thinker.simulation_results:
            print(result.model_dump_json(), file=fp)
        for result in thinker.learning_results:
            print(result.model_dump_json(exclude={'inputs', 'value'}), file=fp)
    print(f'Saved results to run-data/batched-results.json')

    # Cleanup
    queues.send_kill_signal()
    stop_task_server(task_server, server_thread)
    print('Done.')


if __name__ == '__main__':
    main()
