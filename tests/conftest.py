"""pytest 共享夹具。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import (
    DetectionConfig,
    ModelConfig,
    MotionConfig,
    TrackingConfig,
    TrajectoryConfig,
)
from src.datatypes import Detection


def make_detection(
    cx: float,
    cy: float,
    size: float = 100.0,
    conf: float = 0.9,
    class_id: int = 0,
    class_name: str = "person",
) -> Detection:
    """构造以 (cx, cy) 为中心、边长 size 的正方形检测框。"""
    half = size / 2.0
    return Detection(
        xyxy=np.array(
            [cx - half, cy - half, cx + half, cy + half], dtype=np.float32
        ),
        conf=conf,
        class_id=class_id,
        class_name=class_name,
    )


@pytest.fixture
def detection_factory():
    """暴露构造函数给测试用。"""
    return make_detection


@pytest.fixture
def model_cfg() -> ModelConfig:
    """标准模型配置（测试不加载真实权重）。"""
    return ModelConfig(
        weights=Path("models/yolo11n.pt"),
        imgsz=640,
        half=False,
        verbose=False,
    )


@pytest.fixture
def detection_cfg() -> DetectionConfig:
    """标准检测配置。"""
    return DetectionConfig(
        conf=0.10, iou=0.45, max_det=300, classes=[0], display_conf=0.30
    )


@pytest.fixture
def tracking_cfg() -> TrackingConfig:
    """标准跟踪配置。"""
    return TrackingConfig(
        enabled=True,
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        minimum_consecutive_frames=1,
    )


@pytest.fixture
def trajectory_cfg() -> TrajectoryConfig:
    """标准轨迹配置。"""
    return TrajectoryConfig(max_points=200, ttl_frames=60)


@pytest.fixture
def motion_cfg() -> MotionConfig:
    """标准运动估计配置（无标定，px/s）。"""
    return MotionConfig(window=15, min_points=5, pixels_per_meter=None)
