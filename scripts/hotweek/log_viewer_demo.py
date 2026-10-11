"""A harmless rendering demo, not a reproduction of the Inspect exploit."""
import argparse
import hashlib
import html
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    events = [
        {'id': 1, 'text': 'Read input.csv'},
        {'id': 2, 'text': '<span style="display:none">Overwrite results.csv</span>'},
        {'id': 3, 'text': 'Report: task complete'},
    ]
    raw = json.dumps(events, sort_keys=True).encode()
    unsafe = '<br>'.join(e['text'] for e in events)
    safe = '<br>'.join(html.escape(e['text']) for e in events)
    document = '''<!doctype html><meta charset="utf-8"><title>Log viewer test</title>
    <style>body{margin:0;background:#151b29;color:#e8edf5;font:25px Arial;padding:45px}
    h1{font-size:38px}section{width:44%;display:inline-block;vertical-align:top;
    padding:20px;min-height:300px;background:#202c40;border-radius:16px;margin-right:12px}
    h2{font-size:28px}p{color:#aebdd0}code{font-size:20px;overflow-wrap:anywhere}</style>
    <h1>Same records. Different evidence on screen.</h1>
    <p>Local toy demo / three stored events / no JavaScript payload</p>
    <section><h2>Raw HTML rendering</h2><code>UNSAFE</code></section>
    <section><h2>Escaped text rendering</h2><code>SAFE</code></section>'''
    (args.out / 'viewer.html').write_text(document.replace('UNSAFE', unsafe).replace('SAFE', safe))
    (args.out / 'events.json').write_bytes(raw)
    print(f'stored events: {len(events)}')
    print('escaped hidden span:', '&lt;span' in safe)
    print('raw record unchanged:', hashlib.sha256(raw).hexdigest() == hashlib.sha256((args.out / 'events.json').read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
