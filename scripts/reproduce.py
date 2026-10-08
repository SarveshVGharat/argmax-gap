#!/usr/bin/env python3
"""Prepare data, evaluate policies, fit methods, and generate the paper report."""
import argparse
from pathlib import Path
import subprocess
import sys

SCRIPTS = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True, help='Directory created by fetch_assets.py')
    parser.add_argument('--output', type=Path, required=True, help='New run directory')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--smoke', action='store_true', help='150-row run with reduced nonlinear search')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory is not empty; use a new directory to keep runs independent')
    args.output.mkdir(parents=True, exist_ok=True)
    assets, output = args.assets.resolve(), args.output.resolve()

    def run(script, *arguments):
        subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, arguments)], check=True)

    for split, source_format, filename in [('test', 'allie-jsonl', 'allie-test.jsonl'),
                                           ('development', 'heldout-parquet', 'heldout.parquet')]:
        position_file = output / 'data' / f'{split}.parquet'
        extra = ['--max-rows', '150'] if args.smoke else []
        run('prepare_data.py', '--source-format', source_format, '--input', assets / filename,
            '--output', position_file, *extra)
        for model, checkpoint in [('maia3', 'maia3-79m.pt'), ('allie', 'allie-medium.pt')]:
            run('evaluate_policies.py', '--model', model, '--input', position_file,
                '--output', output / 'predictions' / f'{model}_{split}.parquet',
                '--upstream-root', assets / 'upstream' / model, '--checkpoint', assets / checkpoint,
                '--device', args.device, '--batch-size', args.batch_size)
    predictions = output / 'predictions'
    run('train_methods.py', '--heldout-maia', predictions / 'maia3_development.parquet',
        '--heldout-allie', predictions / 'allie_development.parquet',
        '--test-maia', predictions / 'maia3_test.parquet', '--test-allie', predictions / 'allie_test.parquet',
        '--output', output / 'methods', '--diagnostic-time', *(['--smoke', '--smoke-rows', '150'] if args.smoke else []))
    run('report.py', '--positions', output / 'data' / 'test.parquet',
        '--maia', predictions / 'maia3_test.parquet', '--allie', predictions / 'allie_test.parquet',
        '--methods', output / 'methods', '--output', output / 'report',
        '--expected-rows', 150 if args.smoke else 884049, '--bootstrap-reps', 100 if args.smoke else 10000)
    run('audit_splits.py', '--test', output / 'data' / 'test.parquet',
        '--heldout', output / 'data' / 'development.parquet',
        '--selector-split', output / 'methods' / 'selector_split.npz',
        '--refiner-split', output / 'methods' / 'refiner_split.npz',
        '--test-maia3', predictions / 'maia3_test.parquet', '--test-allie', predictions / 'allie_test.parquet',
        '--heldout-maia3', predictions / 'maia3_development.parquet', '--heldout-allie', predictions / 'allie_development.parquet',
        '--output', output / 'audit.json')
    run('figure1.py', '--output', output / 'report' / 'figure1.svg')
    run('ensemble_sweep.py', '--maia', predictions / 'maia3_test.parquet',
        '--allie', predictions / 'allie_test.parquet', '--output', output / 'report',
        '--expected-rows', 150 if args.smoke else 884049)
    if not args.smoke:
        run('filtering_sensitivity.py', '--source-jsonl', assets / 'allie-test.jsonl',
            '--retained', output / 'data' / 'test.parquet', '--output', output / 'report' / 'filtering',
            '--reference', SCRIPTS.parent / 'reference' / 'provenance.csv')
        run('check_reference.py', '--report', output / 'report', '--require-all')
    print(f'Completed {"smoke test" if args.smoke else "paper reproduction"}: {output}')


if __name__ == '__main__':
    main()
