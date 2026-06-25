#!/usr/bin/env python3
"""Validate artifact schemas for SFT and benchmark files."""
import json
import pathlib
import sys

errors = 0
for split in ['gold', 'full']:
    for part in ['train', 'validation']:
        p = pathlib.Path(f'data/sft/final_v4/{split}/{part}.jsonl')
        if p.exists():
            with open(p) as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    if 'instruction' not in rec:
                        print(f'ERROR: {p}:{i} missing instruction')
                        errors += 1
                    if 'output' not in rec:
                        print(f'ERROR: {p}:{i} missing output')
                        errors += 1
print(f'Schema validation: {errors} errors')
sys.exit(1 if errors else 0)
