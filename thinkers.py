"""Thinker classes for standalone Colmena examples."""

from threading import Lock, Event
from random import shuffle
from typing import List
from time import perf_counter

import numpy as np
import pandas as pd
from colmena.models import Result
from colmena.thinker import BaseThinker, event_responder, task_submitter, result_processor
from colmena.thinker.resources import ResourceCounter


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
        self.n_submitted = 0
        self.priority_list = list(self.molecule_list)
        shuffle(self.priority_list)
        self.priority_list_lock = Lock()
        self.rec.reallocate(None, 'simulate', n_parallel)

    @task_submitter(task_type='simulate', n_slots=1)
    def submit_calc(self):
        if self.n_submitted >= self.n_to_evaluate:
            self.done.set()
            return
        with self.priority_list_lock:
            next_mol = self.priority_list.pop()
        self.n_submitted += 1
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

        self.n_submitted = 0

        # Assign all resources to simulation to start with
        self.rec.reallocate(None, 'simulate', n_parallel)

    @task_submitter(task_type='simulate', n_slots=1)
    def submit_calc(self):
        """Submit a calculation when resources are available."""
        if self.n_submitted >= self.n_to_evaluate:
            self.done.set()
            return
        self.task_list_ready.wait()

        with self.priority_list_lock:
            next_mol = self.priority_list.pop()
            self.already_ran.add(next_mol)

        self.n_submitted += 1
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
