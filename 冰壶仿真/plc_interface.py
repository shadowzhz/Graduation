"""西门子 S7-1500 PLC 通信模块。

通过 python-snap7 把 AI 决策数据写入 PLC 的 DB 块。
需要在 TIA Portal 里创建一个 DB（比如 DB1），包含以下变量：

DB 结构（共 36 字节）：
  Offset  0: AI_TargetX   (REAL)  - AI 球槌目标 X
  Offset  4: AI_TargetY   (REAL)  - AI 球槌目标 Y
  Offset  8: AI_PosX      (REAL)  - AI 球槌当前 X
  Offset 12: AI_PosY      (REAL)  - AI 球槌当前 Y
  Offset 16: StoneX       (REAL)  - 冰壶 X
  Offset 20: StoneY       (REAL)  - 冰壶 Y
  Offset 24: StoneVX      (REAL)  - 冰壶 X 速度
  Offset 28: StoneVY      (REAL)  - 冰壶 Y 速度
  Offset 32: PlayerScore  (INT)   - 玩家得分
  Offset 34: AIScore      (INT)   - AI 得分

用法：
    plc = PLCInterface("192.168.0.1", db_number=1)
    plc.connect()
    plc.write_game_state(ai_target_x, ai_target_y, ai_x, ai_y,
                         stone_x, stone_y, stone_vx, stone_vy,
                         player_score, ai_score)
    plc.disconnect()
"""

try:
    import snap7
    from snap7.util import set_real, set_int

    HAS_SNAP7 = True
except ImportError:
    HAS_SNAP7 = False

# DB 布局
DB_NUMBER = 1
_AI_TARGET_X_OFFSET = 0
_AI_TARGET_Y_OFFSET = 4
_AI_POS_X_OFFSET = 8
_AI_POS_Y_OFFSET = 12
_STONE_X_OFFSET = 16
_STONE_Y_OFFSET = 20
_STONE_VX_OFFSET = 24
_STONE_VY_OFFSET = 28
_PLAYER_SCORE_OFFSET = 32
_AI_SCORE_OFFSET = 34
_TOTAL_BYTES = 36


class PLCInterface:
    """S7-1500 PLC 通信接口。"""

    def __init__(self, ip: str = "192.168.0.1", rack: int = 0, slot: int = 1,
                 db_number: int = DB_NUMBER) -> None:
        self.ip = ip
        self.rack = rack
        self.slot = slot
        self.db_number = db_number
        self._client = None
        self._connected = False

    def connect(self) -> bool:
        """连接 PLC。成功返回 True。"""
        if not HAS_SNAP7:
            print("[PLC] python-snap7 未安装，运行: pip install python-snap7")
            return False
        try:
            self._client = snap7.client.Client()
            self._client.connect(self.ip, self.rack, self.slot)
            self._connected = True
            print(f"[PLC] 已连接 {self.ip} (rack={self.rack}, slot={self.slot})")
            return True
        except Exception as e:
            print(f"[PLC] 连接失败: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """断开连接。"""
        if self._client and self._connected:
            try:
                self._client.disconnect()
            except Exception:
                pass
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def write_game_state(self, ai_target_x: float, ai_target_y: float,
                         ai_x: float, ai_y: float,
                         stone_x: float, stone_y: float,
                         stone_vx: float, stone_vy: float,
                         player_score: int, ai_score: int) -> bool:
        """把一帧游戏数据写入 PLC DB。成功返回 True。"""
        if not self._connected:
            return False
        try:
            data = bytearray(_TOTAL_BYTES)
            set_real(data, _AI_TARGET_X_OFFSET, ai_target_x)
            set_real(data, _AI_TARGET_Y_OFFSET, ai_target_y)
            set_real(data, _AI_POS_X_OFFSET, ai_x)
            set_real(data, _AI_POS_Y_OFFSET, ai_y)
            set_real(data, _STONE_X_OFFSET, stone_x)
            set_real(data, _STONE_Y_OFFSET, stone_y)
            set_real(data, _STONE_VX_OFFSET, stone_vx)
            set_real(data, _STONE_VY_OFFSET, stone_vy)
            set_int(data, _PLAYER_SCORE_OFFSET, player_score)
            set_int(data, _AI_SCORE_OFFSET, ai_score)
            self._client.db_write(self.db_number, 0, data)
            return True
        except Exception as e:
            print(f"[PLC] 写入失败: {e}")
            self._connected = False
            return False
