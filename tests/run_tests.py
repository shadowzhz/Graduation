"""不装 pytest 也能跑全部测试：python3 tests/run_tests.py"""

import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "air_hockey", ROOT / "冰壶仿真"):
    sys.path.insert(0, str(path))

passed = failed = 0
failures = []
for module_file in sorted(Path(__file__).resolve().parent.glob("test_*.py")):
    module = importlib.import_module(module_file.stem)
    for name in sorted(dir(module)):
        if not name.startswith("test_"):
            continue
        func = getattr(module, name)
        if not callable(func):
            continue
        try:
            func()
            passed += 1
            print(f"PASS {module_file.stem}::{name}")
        except Exception:
            failed += 1
            failures.append((f"{module_file.stem}::{name}", traceback.format_exc()))
            print(f"FAIL {module_file.stem}::{name}")

print(f"\npassed={passed}, failed={failed}")
for name, tb in failures:
    print(f"\n--- {name} ---\n{tb}")
sys.exit(1 if failed else 0)
