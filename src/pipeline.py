"""逐帧感知流水线：检测 -> 跟踪 -> 轨迹 -> 运动估计 -> 组装 FrameResult。

依赖全部通过构造参数注入（便于测试替换为桩对象）；
本模块不 import cv2 的任何窗口相关 API，可纯离线测试。
"""

from __future__ import annotations

import logging
import time

from src.config import MotionConfig
from src.datatypes import FrameResult
from src.detector import Detector
from src.motion import update_history
from src.tracker import Tracker
from src.trajectory import TrajectoryManager

logger = logging.getLogger(__name__)


class PerceptionPipeline:
    """逐帧编排器。时间测量一律使用 time.perf_counter()。"""

    def __init__(
        self,
        detector: Detector,
        tracker: Tracker,
        trajectory_mgr: TrajectoryManager,
        motion_cfg: MotionConfig,
        fps_ema_decay: float,
    ) -> None:
        """初始化流水线。

        Args:
            detector: 检测器（或等价桩对象）。
            tracker: 跟踪器（或等价桩对象）。
            trajectory_mgr: 轨迹管理器。
            motion_cfg: 运动估计配置。
            fps_ema_decay: FPS 的 EMA 平滑系数（来自 performance.fps_ema_decay）。
        """
        self._detector = detector
        self._tracker = tracker
        self._trajectory_mgr = trajectory_mgr
        self._motion_cfg = motion_cfg
        self._fps_ema_decay = fps_ema_decay
        self._fps: float | None = None

    def process(self, frame, frame_index: int, timestamp: float) -> FrameResult:
        """处理单帧。

        Args:
            frame: BGR 帧（仅透传给检测器）。
            frame_index: 帧号（从 0 起）。
            timestamp: 帧时间戳（秒）。

        Returns:
            FrameResult: 当前帧感知结果。
        """
        t_start = time.perf_counter()
        detections = self._detector.detect(frame)
        t_detect = time.perf_counter()

        tracks = self._tracker.update(detections)
        t_track = time.perf_counter()

        histories = self._trajectory_mgr.update(tracks, frame_index, timestamp)
        for history in histories.values():
            update_history(history, self._motion_cfg)
        # TTL 淘汰必须每帧执行（内存有界的唯一保证）
        self._trajectory_mgr.prune(frame_index)
        t_end = time.perf_counter()

        detect_ms = (t_detect - t_start) * 1000.0
        track_ms = (t_track - t_detect) * 1000.0
        total_ms = (t_end - t_start) * 1000.0

        instantaneous_fps = 1000.0 / total_ms if total_ms > 0 else 0.0
        if self._fps is None:
            self._fps = instantaneous_fps  # 首帧直接取 1/total_s
        else:
            decay = self._fps_ema_decay
            self._fps = decay * self._fps + (1.0 - decay) * instantaneous_fps

        return FrameResult(
            frame_index=frame_index,
            timestamp=timestamp,
            tracks=tracks,
            histories=histories,
            detect_ms=detect_ms,
            track_ms=track_ms,
            total_ms=total_ms,
            fps=self._fps,
        )

    def reset(self) -> None:
        """重置全部可变状态（跨视频切换时调用）。"""
        self._tracker.reset()
        self._trajectory_mgr.reset()
        self._fps = None
        logger.info("流水线状态已重置")
