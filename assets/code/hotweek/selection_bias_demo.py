"""Synthetic task selection, not METR data or an AI productivity estimate."""
import argparse
import json
from pathlib import Path


def summarize(tasks):
    manual = sum(t['manual'] for t in tasks)
    assisted = sum(t['assisted'] for t in tasks)
    return {'tasks': len(tasks), 'manual_minutes': manual, 'assisted_minutes': assisted,
            'time_change_percent': round(100 * (assisted / manual - 1), 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    tasks = [{'type': 'high benefit', 'manual': 60, 'assisted': 15} for _ in range(50)]
    tasks += [{'type': 'low benefit', 'manual': 60, 'assisted': 66} for _ in range(50)]
    selected = [t for t in tasks if t['type'] == 'low benefit']
    results = {'all_tasks': summarize(tasks), 'submitted_tasks': summarize(selected)}
    for name, row in results.items():
        print(name, json.dumps(row))
    assert results['all_tasks']['time_change_percent'] < 0 < results['submitted_tasks']['time_change_percent']
    args.out.write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
