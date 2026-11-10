"""轨迹历史管理：定长 deque + TTL 淘汰，保证内存有界。"""

from __future__ import annotations

import logging
from collections import deque

from src.config import TrajectoryConfig
from src.datatypes import Track, TrackHistory

logger = logging.getLogger(__name__)


class TrajectoryManager:
    """维护每个 Track ID 的轨迹历史。

    points 使用 ``deque(maxlen=max_points)``（禁止不定长 list）；
    ``prune`` 按 TTL 回收长期未见的目标，是内存有界的唯一保证。
    """

    def __init__(self, cfg: TrajectoryConfig) -> None:
        """初始化管理器。

        Args:
            cfg: 轨迹配置（max_points / ttl_frames）。
        """
        self._cfg = cfg
        self._histories: dict[int, TrackHistory] = {}

    def update(self, tracks: list[Track], frame_index: int, t: float) -> dict[int, TrackHistory]:
        """以当前帧的跟踪结果更新轨迹。

        Args:
            tracks: 当前帧跟踪目标。
            frame_index: 当前帧号。
            t: 当前帧时间戳（秒）。

        Returns:
            当前帧出现的目标的 {track_id: TrackHistory}。
        """
        current: dict[int, TrackHistory] = {}
        for track in tracks:
            history = self._histories.get(track.track_id)
            if history is None:
                history = TrackHistory(
                    track_id=track.track_id,
                    points=deque(maxlen=self._cfg.max_points),
                    last_seen_frame=frame_index,
                )
                self._histories[track.track_id] = history
            cx, cy = track.center()
            history.points.append((cx, cy, t))
            history.last_seen_frame = frame_index
            current[track.track_id] = history
        return current

    def get_histories(self) -> dict[int, TrackHistory]:
        """返回全部存活轨迹（含 TTL 内但当前帧未出现的目标）的副本。"""
        return dict(self._histories)

    def prune(self, frame_index: int) -> None:
        """回收超过 ttl_frames 未见到的目标（内存有界的唯一保证）。"""
        stale = [
            track_id
            for track_id, history in self._histories.items()
            if frame_index - history.last_seen_frame > self._cfg.ttl_frames
        ]
        for track_id in stale:
            del self._histories[track_id]
        if stale:
            logger.debug("prune 回收了 %d 条过期轨迹: %s", len(stale), stale)

    def reset(self) -> None:
        """清空全部轨迹状态（跨视频切换时调用）。"""
        self._histories.clear()
