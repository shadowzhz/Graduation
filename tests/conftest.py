import sys
from pathlib import Path

# 根目录（共享的 game_state）、视觉工程、仿真工程
ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "air_hockey", ROOT / "冰壶仿真"):
    sys.path.insert(0, str(path))
