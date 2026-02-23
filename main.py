"""Unified CLI entry point for molecular design with Colmena + Parsl.

Usage:
  python main.py --thinker random --config local
  python main.py --thinker batched --config local --search-count 8
  python main.py --thinker batched --config midway --search-count 8
"""

import argparse
import logging
import os
from time import perf_counter, sleep

import pandas as pd
from colmena.task_server.parsl import ParslTaskServer

from chemfunctions import compute_vertical, train_model, run_model
from configs import make_parsl_config, make_queues, start_task_server, stop_task_server
from thinkers import RandomThinker, BatchedThinker


# ---------------------------------------------------------------------------
# Thinker registry
# ---------------------------------------------------------------------------

def _build_random_thinker(queues, n_workers, molecule_list, args):
    return RandomThinker(
        queues=queues,
        n_to_evaluate=args.search_count,
        n_parallel=n_workers,
        molecule_list=molecule_list,
    )


def _build_batched_thinker(queues, n_workers, molecule_list, args):
    return BatchedThinker(
        queues=queues,
        n_to_evaluate=args.search_count,
        n_parallel=n_workers,
        initial_count=args.initial_count,
        batch_size=args.batch_size,
        molecule_list=molecule_list,
    )


THINKERS = {
    'random': _build_random_thinker,
    'batched': _build_batched_thinker,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Molecular design with Colmena + Parsl',
    )
    parser.add_argument(
        '--thinker', required=True, choices=sorted(THINKERS.keys()),
        help='Thinker strategy to use',
    )
    parser.add_argument(
        '--config', required=True,
        help='Platform configuration name (e.g. local, midway)',
    )
    parser.add_argument(
        '--n-workers', type=int, default=min(4, os.cpu_count()),
        help='Number of parallel workers (default: min(4, cpu_count))',
    )
    parser.add_argument(
        '--search-count', type=int, default=4,
        help='Number of molecules to evaluate (default: 4)',
    )
    parser.add_argument(
        '--data-file', type=str, default='data/QM9-search.tsv',
        help='Path to search space TSV file',
    )
    parser.add_argument(
        '--output-dir', type=str, default='run-data',
        help='Directory for output files',
    )
    # Batched-thinker-specific (ignored by random thinker)
    parser.add_argument(
        '--initial-count', type=int, default=4,
        help='Simulations before first ML training (batched thinker only)',
    )
    parser.add_argument(
        '--batch-size', type=int, default=2,
        help='Simulations between retraining (batched thinker only)',
    )
    parser.add_argument(
        '--redis-host', type=str, default='localhost',
        help='Redis host (midway config only)',
    )
    parser.add_argument(
        '--redis-port', type=int, default=6379,
        help='Redis port (midway config only)',
    )
    # HPC Slurm overrides (midway config only)
    parser.add_argument(
        '--account', type=str, default='pi-chard',
        help='Slurm account (midway config only)',
    )
    parser.add_argument(
        '--partition', type=str, default='caslake',
        help='Slurm partition (midway config only)',
    )
    parser.add_argument(
        '--walltime', type=str, default='00:15:00',
        help='Slurm walltime (midway config only)',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.WARNING,
    )

    print(f'=== Molecular Design: thinker={args.thinker}, config={args.config} ===')
    print(f'Workers: {args.n_workers}, Molecules to evaluate: {args.search_count}')

    # Load data
    search_space = pd.read_csv(args.data_file, sep=r'\s+')
    print(f'Search space: {len(search_space)} molecules')

    # Set up Colmena infrastructure
    topics = ['simulate', 'train', 'infer']
    queues = make_queues(args.config, topics=topics, redis_host=args.redis_host, redis_port=args.redis_port)
    parsl_config = make_parsl_config(
        args.config, args.n_workers,
        account=args.account, partition=args.partition, walltime=args.walltime,
    )
    task_server = ParslTaskServer(
        methods=[compute_vertical, train_model, run_model],
        queues=queues,
        config=parsl_config,
    )

    # Start task server
    server_thread = start_task_server(task_server)
    sleep(2)
    print('Task server started.')

    # Build and run thinker
    start = perf_counter()
    thinker = THINKERS[args.thinker](
        queues, args.n_workers, search_space['smiles'].values, args,
    )
    print(f'Running {args.thinker} thinker...')
    thinker.run()
    elapsed = perf_counter() - start
    print(f'Done in {elapsed:.1f}s. Got {len(thinker.database)} results.')

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f'{args.thinker}-results.json')
    with open(output_file, 'w') as fp:
        for result in thinker.simulation_results:
            print(result.model_dump_json(), file=fp)
        for result in thinker.learning_results:
            print(result.model_dump_json(exclude={'inputs', 'value'}), file=fp)
    print(f'Saved results to {output_file}')

    # Cleanup
    queues.send_kill_signal()
    stop_task_server(server_thread)
    print('Done.')


if __name__ == '__main__':
    main()
