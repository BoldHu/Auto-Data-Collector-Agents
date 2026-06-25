#!/usr/bin/env python3
"""Compile-check all Python files in the project."""
import py_compile
import pathlib
import sys

errors = 0
for f in pathlib.Path('src').rglob('*.py'):
    try:
        py_compile.compile(str(f), doraise=True)
    except py_compile.PyCompileError as e:
        print(f'ERROR: {e}')
        errors += 1
for f in pathlib.Path('scripts').glob('*.py'):
    try:
        py_compile.compile(str(f), doraise=True)
    except py_compile.PyCompileError as e:
        print(f'ERROR: {e}')
        errors += 1
print(f'Compile check: {errors} errors')
sys.exit(1 if errors else 0)
