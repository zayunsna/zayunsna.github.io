"""Run fixed or intentionally broken helpers against the frozen test suites."""
import argparse
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--implementation", choices=["broken", "fixed"], required=True)
parser.add_argument("--suite", choices=["baseline", "all"], default="baseline")
args = parser.parse_args()
filename = ROOT / ("helpers.py" if args.implementation == "broken" else "helpers_fixed.py")
spec = importlib.util.spec_from_file_location("helpers", filename)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.modules["helpers"] = module
tests = [str(ROOT / "test_baseline.py")]
if args.suite == "all":
    tests.append(str(ROOT / "test_regressions.py"))
print(f"implementation={args.implementation}, suite={args.suite}", flush=True)
raise SystemExit(pytest.main(["-q", "--tb=short", "-p", "no:cacheprovider", *tests]))
