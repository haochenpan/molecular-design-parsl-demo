"""Example 1: Simulate molecules in a random order using Colmena + Parsl.

Standalone version of the "Example 1: Simulating molecules in a predefined list"
section from `1_interleaving-simulation-and-steering.ipynb`.

What it does:
  - Picks molecules at random from the QM9 search space
  - Submits XTB quantum chemistry calculations (ionization potential) in parallel
  - Collects results and saves them to run-data/random-results.json

Usage:
  python example1_random.py

Platform notes:
  - macOS: Uses ThreadPoolExecutor and a thread-based task server
    (avoids multiprocessing pickling / ZMQ fork issues).
  - Linux: Uses HighThroughputExecutor with cpu_affinity='block'
    and a subprocess-based task server.
"""
import platform
import logging
import os
import json
from random import shuffle
from threading import Lock, Thread
from typing import List
from time import perf_counter

import pandas as pd
from colmena.models import Result
from colmena.task_server.parsl import ParslTaskServer
from colmena.queue.python import PipeQueues
from colmena.thinker.resources import ResourceCounter
from colmena.thinker import BaseThinker, task_submitter, result_processor
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.config import Config

from chemfunctions import compute_vertical, train_model, run_model

# --- Configuration ---
n_workers = min(4, os.cpu_count())
search_count = 4       # Number of molecules to evaluate
# ----------------------

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.WARNING
)


class RandomThinker(BaseThinker):
    """A thinker which evaluates molecules in a random order."""

    def __init__(self, queues, n_to_evaluate: int, n_parallel: int,
                 molecule_list: List[str]):
        """Initialize the thinker.

        Args:
            queues: Client side of Colmena queues
            n_to_evaluate: Number of molecules to evaluate
            n_parallel: Number of computations to run in parallel
            molecule_list: List of SMILES strings to choose from
        """
        super().__init__(
            queues,
            ResourceCounter(n_parallel, ['simulate', 'train', 'infer'])
        )
        self.molecule_list = set(molecule_list)
        self.n_to_evaluate = n_to_evaluate
        self.database = dict()
        self.simulation_results = []
        self.priority_list = list(self.molecule_list)
        shuffle(self.priority_list)
        self.priority_list_lock = Lock()
        self.rec.reallocate(None, 'simulate', n_parallel)

    @task_submitter(task_type='simulate', n_slots=1)
    def submit_calc(self):
        with self.priority_list_lock:
            next_mol = self.priority_list.pop()
        self.queues.send_inputs(next_mol, method='compute_vertical')
        print(f'  Submitted: {next_mol}')

    @result_processor
    def receive_calc(self, result: Result):
        self.rec.release('simulate', 1)
        if result.success:
            self.database[result.args[0]] = result.value
            print(f'  Result {len(self.database)}/{self.n_to_evaluate}: '
                  f'{result.args[0]} -> {result.value:.4f}')
            if len(self.database) >= self.n_to_evaluate:
                self.done.set()
        else:
            self.logger.warning(f'Simulation failure: {result.failure_info}')
        self.simulation_results.append(result)


def make_parsl_config(n_workers: int) -> Config:
    """Create a Parsl config appropriate for the current platform."""
    if platform.system() == 'Linux':
        executor = HighThroughputExecutor(
            max_workers_per_node=n_workers,
            cpu_affinity='block'
        )
    else:
        executor = ThreadPoolExecutor(max_threads=n_workers)
    return Config(executors=[executor])


def start_task_server(task_server: ParslTaskServer):
    """Start the task server in a way that works on both macOS and Linux."""
    if platform.system() == 'Darwin':
        # On macOS, run in a thread to avoid fork+ZMQ issues
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


def main():
    print(f'=== Example 1: Random Molecule Evaluation ===')
    print(f'Workers: {n_workers}, Molecules to evaluate: {search_count}')

    # Load data
    search_space = pd.read_csv('data/QM9-search.tsv', sep=r'\s+')
    print(f'Search space: {len(search_space)} molecules')

    # Set up Colmena
    queues = PipeQueues(topics=['simulate', 'train', 'infer'], serialization_method='pickle')
    config = make_parsl_config(n_workers)
    task_server = ParslTaskServer(
        methods=[compute_vertical, train_model, run_model],
        queues=queues, config=config
    )

    # Start
    import time
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
