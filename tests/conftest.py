import sys
from pathlib import Path

# 项目根目录（air_hockey 包与共享的 game_state 都在这里）
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
