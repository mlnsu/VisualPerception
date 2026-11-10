"""pipeline.py 测试：注入桩 detector/tracker，跑合成帧，验证 FrameResult 与 CSV 行数。"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from src.config import MotionConfig
from src.csv_logger import CsvLogger
from src.datatypes import Detection
from src.pipeline import PerceptionPipeline
from src.tracker import Tracker
from src.trajectory import TrajectoryManager
from tests.conftest import make_detection


class StubDetector:
    """脚本化桩检测器：按调用次序返回预设检测。"""

    def __init__(self, script: list[list[Detection]]):
        self._script = script
        self._call = 0

    def detect(self, frame):
        index = min(self._call, len(self._script) - 1)
        self._call += 1
        return self._script[index]


def build_pipeline(script, tracking_cfg, trajectory_cfg):
    """组装被测流水线（真实 tracker/trajectory + 桩 detector）。"""
    motion_cfg = MotionConfig(window=15, min_points=5, pixels_per_meter=None)
    return PerceptionPipeline(
        StubDetector(script),
        Tracker(tracking_cfg),
        TrajectoryManager(trajectory_cfg),
        motion_cfg,
        fps_ema_decay=0.9,
    )


def rightward_script(n_frames: int, conf: float = 0.9):
    """单目标向右匀速平移的检测脚本。"""
    return [
        [make_detection(100.0 + 10.0 * i, 240.0, conf=conf)] for i in range(n_frames)
    ]


@pytest.mark.parametrize("n_frames", [1, 5, 30])
def test_frameresult_fields(n_frames, tracking_cfg, trajectory_cfg):
    """FrameResult 各字段类型/取值正确。"""
    pipeline = build_pipeline(rightward_script(n_frames), tracking_cfg, trajectory_cfg)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(n_frames):
        result = pipeline.process(frame, frame_index=i, timestamp=i / 30.0)
        assert result.frame_index == i
        assert result.timestamp == pytest.approx(i / 30.0)
        assert len(result.tracks) == 1
        assert result.histories.keys() == {result.tracks[0].track_id}
        assert result.detect_ms >= 0.0
        assert result.track_ms >= 0.0
        assert result.total_ms >= result.detect_ms + result.track_ms - 1e-6
        assert result.fps > 0.0


def test_fps_is_ema_smoothed(tracking_cfg, trajectory_cfg):
    """FPS 为 EMA 平滑值：首帧直接取瞬时值，后续为加权平均。"""
    pipeline = build_pipeline(rightward_script(10), tracking_cfg, trajectory_cfg)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    first = pipeline.process(frame, 0, 0.0)
    assert first.fps == pytest.approx(1000.0 / first.total_ms, rel=1e-3)
    second = pipeline.process(frame, 1, 1.0 / 30.0)
    expected = 0.9 * first.fps + 0.1 * (1000.0 / second.total_ms)
    assert second.fps == pytest.approx(expected, rel=1e-3)


def test_speed_estimated_in_pipeline(tracking_cfg, trajectory_cfg):
    """流水线贯通后轨迹速度被正确估计（≈300 px/s）。"""
    script = [
        [make_detection(100.0 + 300.0 * (i / 30.0), 240.0)] for i in range(15)
    ]
    pipeline = build_pipeline(script, tracking_cfg, trajectory_cfg)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(15):
        result = pipeline.process(frame, i, i / 30.0)
    history = result.histories[result.tracks[0].track_id]
    assert history.speed_px_s == pytest.approx(300.0, rel=0.05)
    assert history.direction_deg == pytest.approx(0.0, abs=1.0)


def test_csv_rows_match_tracks(tracking_cfg, trajectory_cfg, tmp_path):
    """CSV 行数 == 每帧通过 display_conf 的目标数之和，列契约完整。"""
    script = rightward_script(10)
    pipeline = build_pipeline(script, tracking_cfg, trajectory_cfg)
    csv_path = tmp_path / "detections.csv"
    csv_logger = CsvLogger(csv_path, flush_every=3, display_conf=0.30)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        for i in range(10):
            csv_logger.log(pipeline.process(frame, i, i / 30.0))
    finally:
        csv_logger.close()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10  # 每帧 1 个目标 × 10 帧
    expected_columns = {
        "frame_idx", "timestamp", "track_id", "class_id", "class_name",
        "x1", "y1", "x2", "y2", "cx", "cy", "conf",
        "direction_deg", "speed_px_s", "speed_m_s", "fps",
    }
    assert set(rows[0].keys()) == expected_columns
    assert all(row["track_id"] and int(row["track_id"]) >= 1 for row in rows)


def test_csv_filters_low_conf(tracking_cfg, trajectory_cfg, tmp_path):
    """conf < display_conf 的目标不写入 CSV。"""
    script = rightward_script(5, conf=0.20)  # 低于 display_conf=0.30
    pipeline = build_pipeline(script, tracking_cfg, trajectory_cfg)
    csv_path = tmp_path / "detections.csv"
    csv_logger = CsvLogger(csv_path, flush_every=100, display_conf=0.30)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        for i in range(5):
            csv_logger.log(pipeline.process(frame, i, i / 30.0))
    finally:
        csv_logger.close()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == []


def test_prune_called_each_frame(tracking_cfg, trajectory_cfg):
    """流水线每帧调用 prune：长期未见的目标被回收（内存有界）。"""
    # 前 5 帧有目标，之后 100 帧空检测（TTL=60 内应被清掉）
    script = rightward_script(5) + [[] for _ in range(100)]
    trajectory_cfg_local = type(trajectory_cfg)(max_points=100, ttl_frames=30)
    pipeline = build_pipeline(script, tracking_cfg, trajectory_cfg_local)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(len(script)):
        pipeline.process(frame, i, i / 30.0)
    # 桩 detector 已耗尽脚本，直接检查轨迹管理器状态
    assert pipeline._trajectory_mgr.get_histories() == {}  # noqa: SLF001


def test_reset_clears_state(tracking_cfg, trajectory_cfg):
    """reset 后轨迹清空、跟踪器状态复位、ID 重新开始。"""
    pipeline = build_pipeline(rightward_script(5), tracking_cfg, trajectory_cfg)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    first = pipeline.process(frame, 0, 0.0)
    id_first = first.tracks[0].track_id

    pipeline.reset()
    assert pipeline._trajectory_mgr.get_histories() == {}  # noqa: SLF001

    after = pipeline.process(frame, 1, 0.0)
    assert after.tracks[0].track_id == id_first
