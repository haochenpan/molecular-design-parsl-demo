"""Example 1: Simulate molecules in a random order using Colmena + Parsl.

Standalone version of the "Example 1: Simulating molecules in a predefined list"
section from `1_interleaving-simulation-and-steering.ipynb`.

What it does:
  - Picks molecules at random from the QM9 search space
  - Submits XTB quantum chemistry calculations (ionization potential) in parallel
  - Collects results and saves them to run-data/random-results.json

Usage:
  python example1_random.py
"""

import logging
import os
import time
from time import perf_counter

import pandas as pd
from colmena.task_server.parsl import ParslTaskServer

from chemfunctions import compute_vertical, train_model, run_model
from configs import make_parsl_config, make_queues, start_task_server, stop_task_server
from thinkers import RandomThinker

# --- Configuration ---
n_workers = min(4, os.cpu_count())
search_count = 4  # Number of molecules to evaluate

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.WARNING,
)


def main():
    print(f'=== Example 1: Random Molecule Evaluation ===')
    print(f'Workers: {n_workers}, Molecules to evaluate: {search_count}')

    # Load data
    search_space = pd.read_csv('data/QM9-search.tsv', sep=r'\s+')
    print(f'Search space: {len(search_space)} molecules')

    # Set up Colmena
    queues = make_queues(topics=['simulate', 'train', 'infer'])
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
    thinker = RandomThinker(queues, search_count, n_workers, search_space['smiles'].values)
    print('Running random thinker...')
    thinker.run()
    elapsed = perf_counter() - start
    print(f'Done in {elapsed:.1f}s. Got {len(thinker.database)} results.')

    # Save results
    os.makedirs('run-data', exist_ok=True)
    with open('run-data/random-results.json', 'w') as fp:
        for result in thinker.simulation_results:
            print(result.model_dump_json(), file=fp)
    print(f'Saved {len(thinker.simulation_results)} results to run-data/random-results.json')

    # Cleanup
    queues.send_kill_signal()
    stop_task_server(task_server, server_thread)
    print('Done.')


if __name__ == '__main__':
    main()
