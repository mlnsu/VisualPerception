"""跨模块共享的数据契约。

所有跨模块传递的结构一律使用本文件中的 dataclass，禁止裸 tuple/dict。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

#: 轨迹点：(cx, cy, t)，t 为时间戳（秒）
Point = tuple[float, float, float]


@dataclass(frozen=True)
class Detection:
    """单帧检测结果。

    Attributes:
        xyxy: 形状 (4,) 的 float32 数组，顺序 x1, y1, x2, y2（已裁剪到帧内）。
        conf: 置信度 [0,1]。
        class_id: 类别 ID。
        class_name: 类别名。
    """

    xyxy: np.ndarray
    conf: float
    class_id: int
    class_name: str

    def center(self) -> tuple[float, float]:
        """返回检测框中心 (cx, cy)。"""
        x1, y1, x2, y2 = (float(v) for v in self.xyxy)
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def width(self) -> float:
        """返回检测框宽度（像素）。"""
        return float(self.xyxy[2]) - float(self.xyxy[0])

    def height(self) -> float:
        """返回检测框高度（像素）。"""
        return float(self.xyxy[3]) - float(self.xyxy[1])


@dataclass(frozen=True)
class Track:
    """跟踪目标（含分配的 Track ID）。

    Attributes:
        track_id: 跟踪 ID，恒 >= 1。
        xyxy: 形状 (4,) 的 float32 数组，顺序 x1, y1, x2, y2。
        conf: 置信度 [0,1]。
        class_id: 类别 ID。
        class_name: 类别名。
    """

    track_id: int
    xyxy: np.ndarray
    conf: float
    class_id: int
    class_name: str

    def center(self) -> tuple[float, float]:
        """返回跟踪框中心 (cx, cy)。"""
        x1, y1, x2, y2 = (float(v) for v in self.xyxy)
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass
class TrackHistory:
    """单个 Track ID 的轨迹历史与运动状态。

    Attributes:
        track_id: 跟踪 ID。
        points: deque(maxlen=trajectory.max_points)，元素为 (cx, cy, t)。
        direction_deg: 运动方向 [0,360)，x 轴正向为 0°，顺时针为正；不足估计时为 None。
        speed_px_s: 速度（像素/秒）；不足估计时为 None。
        speed_m_s: 速度（米/秒）；pixels_per_meter 为空时恒为 None。
        last_seen_frame: 最近一次出现该 ID 的帧号。
    """

    track_id: int
    points: deque
    direction_deg: float | None = None
    speed_px_s: float | None = None
    speed_m_s: float | None = None
    last_seen_frame: int = 0


@dataclass
class FrameResult:
    """单帧感知结果（pipeline 输出）。

    Attributes:
        frame_index: 帧号（从 0 起）。
        timestamp: 帧时间戳（秒）；文件源为视频时间，摄像头为墙上时钟相对时间。
        tracks: 当前帧的跟踪目标列表。
        histories: 以 track_id 为键的轨迹历史（当前帧出现的目标）。
        detect_ms: 检测耗时（毫秒）。
        track_ms: 跟踪耗时（毫秒）。
        total_ms: 帧端到端处理耗时（毫秒）。
        fps: EMA 平滑后的端到端 FPS。
    """

    frame_index: int
    timestamp: float
    tracks: list[Track]
    histories: dict[int, TrackHistory]
    detect_ms: float
    track_ms: float
    total_ms: float
    fps: float
