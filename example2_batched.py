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
from threading import Lock, Event, Thread
from typing import List
from time import perf_counter

import numpy as np
import pandas as pd
from colmena.models import Result
from colmena.task_server.parsl import ParslTaskServer
from colmena.queue.python import PipeQueues
from colmena.thinker.resources import ResourceCounter
from colmena.thinker import BaseThinker, event_responder, task_submitter, result_processor
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.config import Config

from chemfunctions import compute_vertical, train_model, run_model

# --- Configuration ---
n_workers = min(4, os.cpu_count())
search_count = 8       # Number of molecules to evaluate
initial_count = 4      # Simulations before first training (must be >= 4 for KNN)
batch_size = 2         # Simulations between retraining
# ----------------------

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.WARNING
)


class BatchedThinker(BaseThinker):
    """A thinker which uses ML to prioritize which molecules to simulate.

    Stripped of Jupyter dashboard dependencies for standalone execution.
    """

    def __init__(self, queues, n_to_evaluate: int, n_parallel: int,
                 batch_size: int, initial_count: int,
                 molecule_list: List[str]):
        """Initialize the thinker.

        Args:
            queues: Client side of Colmena queues
            n_to_evaluate: Number of molecules to evaluate total
            n_parallel: Number of computations to run in parallel
            batch_size: Simulations between retraining
            initial_count: Minimum simulations before first training
            molecule_list: List of SMILES strings to choose from
        """
        super().__init__(
            queues,
            ResourceCounter(n_parallel, ['simulate', 'train', 'infer'])
        )

        self.n_to_evaluate = n_to_evaluate
        self.initial_count = initial_count
        self.batch_size = batch_size
        self.n_parallel = n_parallel

        # Ensure inference task chunks are large enough
        self.inference_tasks = max(
            len(molecule_list) // 20000,
            self.batch_size * 2
        )

        self.database = dict()
        self.already_ran = set()
        self.simulation_results = []
        self.learning_results = []

        self.priority_list = list(molecule_list)
        shuffle(self.priority_list)
        self.priority_list_lock = Lock()

        # Events to coordinate simulation/ML phases
        self.start_update = Event()
        self.task_list_ready = Event()
        self.task_list_ready.set()

        # Assign all resources to simulation to start with
        self.rec.reallocate(None, 'simulate', n_parallel)

    @task_submitter(task_type='simulate', n_slots=1)
    def submit_calc(self):
        """Submit a calculation when resources are available."""
        self.task_list_ready.wait()

        with self.priority_list_lock:
            next_mol = self.priority_list.pop()
            self.already_ran.add(next_mol)

        self.queues.send_inputs(next_mol, method='compute_vertical', topic='simulate')
        print(f'  Submitted: {next_mol}')

    @result_processor(topic='simulate')
    def receive_calc(self, result: Result):
        """Store the output of simulation if it is successful."""
        if result.success:
            self.database[result.args[0]] = result.value
            print(f'  Result {len(self.database)}/{self.n_to_evaluate}: '
                  f'{result.args[0]} -> {result.value:.4f}')

            if len(self.database) >= self.n_to_evaluate:
                self.logger.info('Completed as many as required.')
                self.done.set()

            # Start training if enough data
            if (len(self.database) >= self.initial_count and
                    len(self.database) % self.batch_size == 0):
                self.task_list_ready.clear()
                self.start_update.set()
        else:
            self.logger.warning(f'Simulation failure: {result.failure_info}')

        self.simulation_results.append(result)
        self.rec.release('simulate', 1)

    @event_responder(event_name='start_update')
    def start_training(self):
        """Start the training tasks."""
        print(f'  Starting training (database size: {len(self.database)})...')
        smiles, ie = zip(*self.database.items())
        self.queues.send_inputs(list(smiles), list(ie), method='train_model', topic='train')

    @result_processor(topic='train')
    def receive_new_model(self, result: Result):
        """Receive a finished model and launch inference tasks."""
        assert result.success, f'Model training failed! {result.failure_info.exception}'
        model = result.value
        print('  Training complete. Launching inference...')

        inf_chunks = np.array_split(self.priority_list, self.inference_tasks)
        for chunk in inf_chunks:
            self.queues.send_inputs(model, chunk, method='run_model', topic='infer')

        self.learning_results.append(result)

    @event_responder(event_name='start_update')
    def collect_inference(self):
        """Collect inference results and re-prioritize the task queue."""
        start_time = perf_counter()

        chunks = []
        for i in range(self.inference_tasks):
            result = self.queues.get_result(topic='infer')
            assert result.success, f'Inference failed! {result.failure_info.exception}'
            chunks.append(result.value)
            self.learning_results.append(result)

        # Sort by predicted property ascending (best last for pop())
        results = pd.concat(chunks, ignore_index=True).sort_values('ie', ascending=True)

        with self.priority_list_lock:
            self.priority_list.clear()
            for smiles in results['smiles']:
                if smiles not in self.already_ran:
                    self.priority_list.append(smiles)

        self.task_list_ready.set()
        print(f'  Inference done. Re-prioritized {len(self.priority_list)} molecules. '
              f'({perf_counter() - start_time:.1f}s)')


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
    print(f'=== Example 2: Batched Optimization with ML Steering ===')
    print(f'Workers: {n_workers}, Molecules: {search_count}, '
          f'Initial: {initial_count}, Batch: {batch_size}')

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
    thinker = BatchedThinker(
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
