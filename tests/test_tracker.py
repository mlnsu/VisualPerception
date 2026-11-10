"""tracker.py 测试：真实 sv.ByteTrack + 合成 Detection 序列。"""

from __future__ import annotations

import pytest

from src.config import TrackingConfig
from src.tracker import Tracker
from tests.conftest import make_detection


def moving_sequence(n_frames: int, start=(300.0, 240.0), step=(10.0, 0.0), size=100.0):
    """生成匀速平移的 Detection 序列。"""
    seq = []
    cx, cy = start
    for _ in range(n_frames):
        seq.append([make_detection(cx, cy, size=size)])
        cx += step[0]
        cy += step[1]
    return seq


def test_track_id_stable_during_motion(tracking_cfg):
    """同一目标穿过画面时 Track ID 保持不变。"""
    tracker = Tracker(tracking_cfg)
    ids = set()
    for frame_dets in moving_sequence(20):
        tracks = tracker.update(frame_dets)
        assert len(tracks) == 1
        ids.add(tracks[0].track_id)
    assert len(ids) == 1


def test_track_ids_are_positive(tracking_cfg):
    """任何情况下 track_id >= 1。"""
    tracker = Tracker(tracking_cfg)
    for frame_dets in moving_sequence(5):
        for track in tracker.update(frame_dets):
            assert track.track_id >= 1


def test_track_survives_short_detection_gap(tracking_cfg):
    """中间 3 帧空检测不中断跟踪，目标重现后 ID 不变。"""
    tracker = Tracker(tracking_cfg)
    first = tracker.update(moving_sequence(1)[0])
    original_id = first[0].track_id

    for _ in range(3):
        assert tracker.update([]) == []  # 空帧也必须调用 update

    # 重现位置 = 按速度外推的期望位置
    reappear = make_detection(300.0 + 10.0 * 4, 240.0)
    tracks = tracker.update([reappear])
    assert len(tracks) == 1
    assert tracks[0].track_id == original_id


def test_new_id_after_lost_track_buffer():
    """超过 lost_track_buffer 后重现的目标获得新 ID。"""
    cfg = TrackingConfig(
        enabled=True,
        track_activation_threshold=0.25,
        lost_track_buffer=5,
        minimum_matching_threshold=0.8,
        minimum_consecutive_frames=1,
    )
    tracker = Tracker(cfg)
    first = tracker.update([make_detection(300.0, 240.0)])
    original_id = first[0].track_id

    for _ in range(10):  # 远超 lost_track_buffer=5
        tracker.update([])

    # 新轨迹需要连续 2 帧才被 ByteTrack 激活返回（supervision 实现语义）
    new_ids: set[int] = set()
    for _ in range(3):
        for track in tracker.update([make_detection(300.0, 240.0)]):
            new_ids.add(track.track_id)
    assert new_ids, "重现目标应被重新跟踪"
    assert original_id not in new_ids, "超过 lost_track_buffer 后重现必须获得新 ID"


def test_disabled_mode_assigns_increasing_ids():
    """tracking.enabled=false：每个检测分配临时 id，且逐帧递增。"""
    cfg = TrackingConfig(
        enabled=False,
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        minimum_consecutive_frames=1,
    )
    tracker = Tracker(cfg)
    frame1 = tracker.update([make_detection(100.0, 100.0), make_detection(400.0, 300.0)])
    frame2 = tracker.update([make_detection(100.0, 100.0), make_detection(400.0, 300.0)])

    ids1 = {t.track_id for t in frame1}
    ids2 = {t.track_id for t in frame2}
    assert len(frame1) == 2 and len(frame2) == 2
    assert all(t.track_id >= 1 for t in frame1 + frame2)
    assert not (ids1 & ids2), "降级模式下同一位置的检测每帧都应拿到新 id"


def test_reset_restarts_ids(tracking_cfg):
    """reset 后跟踪状态清零，ID 从头开始分配。"""
    tracker = Tracker(tracking_cfg)
    first = tracker.update([make_detection(300.0, 240.0)])
    id_before_reset = first[0].track_id

    tracker.reset()
    after = tracker.update([make_detection(300.0, 240.0)])
    assert after[0].track_id == id_before_reset


def test_track_fields_mapped(tracking_cfg):
    """Detection 字段到 Track 的映射保持一致（类别名、置信度、坐标）。"""
    tracker = Tracker(tracking_cfg)
    det = make_detection(200.0, 300.0, size=80.0, conf=0.75, class_id=1, class_name="bicycle")
    tracks = tracker.update([det])
    assert len(tracks) == 1
    track = tracks[0]
    assert track.conf == pytest.approx(0.75)
    assert track.class_id == 1
    assert track.class_name == "bicycle"
    assert track.xyxy.dtype.name == "float32"
    assert track.xyxy.tolist() == [160.0, 260.0, 240.0, 340.0]
