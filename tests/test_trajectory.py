"""trajectory.py 测试：定长 deque 与 TTL 淘汰（内存有界的唯一保证）。"""

from __future__ import annotations

import numpy as np

from src.config import TrajectoryConfig
from src.datatypes import Track
from src.trajectory import TrajectoryManager


def make_tracks(*items: tuple[int, float, float]) -> list[Track]:
    """按 (track_id, cx, cy) 列表构造跟踪目标。"""
    tracks = []
    for track_id, cx, cy in items:
        tracks.append(
            Track(
                track_id=track_id,
                xyxy=np.array([cx - 10, cy - 10, cx + 10, cy + 10], dtype=np.float32),
                conf=0.9,
                class_id=0,
                class_name="person",
            )
        )
    return tracks


def test_points_bounded_by_maxlen():
    """轨迹点数恒 <= max_points（deque(maxlen=N) 生效）。"""
    cfg = TrajectoryConfig(max_points=10, ttl_frames=60)
    mgr = TrajectoryManager(cfg)
    for frame in range(25):
        mgr.update(make_tracks((1, 100.0 + frame, 200.0)), frame, frame / 30.0)
    histories = mgr.get_histories()
    assert len(histories) == 1
    assert len(histories[1].points) == 10


def test_update_returns_current_histories():
    """update 返回当前帧出现目标的 {track_id: TrackHistory}。"""
    cfg = TrajectoryConfig(max_points=100, ttl_frames=60)
    mgr = TrajectoryManager(cfg)
    current = mgr.update(make_tracks((1, 50.0, 50.0), (2, 300.0, 300.0)), frame_index=0, t=0.0)
    assert set(current.keys()) == {1, 2}
    assert current[1].last_seen_frame == 0


def test_points_content_and_last_seen():
    """轨迹点内容为 (cx, cy, t)，last_seen_frame 随帧更新。"""
    cfg = TrajectoryConfig(max_points=100, ttl_frames=60)
    mgr = TrajectoryManager(cfg)
    mgr.update(make_tracks((1, 10.0, 20.0)), frame_index=3, t=0.1)
    mgr.update(make_tracks((1, 12.0, 20.0)), frame_index=4, t=0.1333)
    history = mgr.get_histories()[1]
    assert list(history.points) == [(10.0, 20.0, 0.1), (12.0, 20.0, 0.1333)]
    assert history.last_seen_frame == 4


def test_prune_removes_stale_tracks():
    """prune 后超期轨迹被回收，字典长度下降。"""
    cfg = TrajectoryConfig(max_points=100, ttl_frames=5)
    mgr = TrajectoryManager(cfg)
    mgr.update(make_tracks((1, 50.0, 50.0)), frame_index=0, t=0.0)
    mgr.update(make_tracks((2, 400.0, 400.0)), frame_index=1, t=0.033)
    assert len(mgr.get_histories()) == 2

    mgr.prune(frame_index=10)  # 10 - 1 > 5，两条都应过期
    assert len(mgr.get_histories()) == 0


def test_prune_keeps_fresh_tracks():
    """TTL 内的轨迹不被 prune 误删。"""
    cfg = TrajectoryConfig(max_points=100, ttl_frames=60)
    mgr = TrajectoryManager(cfg)
    mgr.update(make_tracks((1, 50.0, 50.0)), frame_index=10, t=0.3)
    mgr.prune(frame_index=30)  # 30 - 10 = 20 <= 60
    assert len(mgr.get_histories()) == 1
    mgr.prune(frame_index=71)  # 71 - 10 = 61 > 60
    assert len(mgr.get_histories()) == 0


def test_reset_clears_all():
    """reset 清空全部轨迹。"""
    cfg = TrajectoryConfig(max_points=100, ttl_frames=60)
    mgr = TrajectoryManager(cfg)
    mgr.update(make_tracks((1, 50.0, 50.0)), frame_index=0, t=0.0)
    mgr.reset()
    assert mgr.get_histories() == {}
