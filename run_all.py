import argparse
import os
from pathlib import Path

for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
from src.load_and_qc import load, denominators
from src.allometric_models import analyze
from src.cv_and_sampling import validate, sampling


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reproduce optical perfusion analyses')
    parser.add_argument('--data', default='data/pone.0288249.s001.sav')
    parser.add_argument('--out', default='results')
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    d, tables, rng = analyze(load(args.data))
    tables.update(validate(d, rng))
    tables.update(denominators=denominators(d), sampling=sampling(d))
    for name, table in tables.items():
        table.to_csv(output / (name+'.csv'), index=False)
    if not args.no_figures:
        from src.generate_figures import render_all
        render_all(d, tables, output / 'figures')
    print('Completed:', output.resolve())
