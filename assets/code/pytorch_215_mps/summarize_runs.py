"""Recompute the article summary from six or more process reports."""
import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reports', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.reports]
    if len({r['fixture_sha256'] for r in reports}) != 1:
        parser.error('Fixture hashes differ; do not combine these runs.')
    if {r['torch'] for r in reports} != {'2.14.0', '2.15.0'}:
        parser.error('Expected process reports from both pinned versions.')
    summary = []
    for version in ['2.14.0', '2.15.0']:
        rows = [r for r in reports if r['torch'] == version]
        if len(rows) < 3:
            parser.error('Provide at least three process reports per version.')
        for index, batch in enumerate([1, 64]):
            if not all(r['cases'][index]['batch'] == batch for r in rows):
                parser.error('Unexpected batch order in a process report.')
            warm = [r['cases'][index]['warm_median_ms'] for r in rows]
            cold = [r['cases'][index]['first_forward_ms'] for r in rows]
            summary.append(dict(version=version, batch=batch, median_ms=statistics.median(warm),
                min_ms=min(warm), max_ms=max(warm), first_forward_ms=cold))
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
