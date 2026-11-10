"""motion.py 测试：匀速合成数据上的方向与速度精度。"""

from __future__ import annotations

import pytest

from src.config import MotionConfig
from src.datatypes import TrackHistory
from src.motion import estimate_direction, estimate_speed, update_history


def make_points(n: int, x_fn, y_fn, t0: float = 0.0, dt: float = 1.0 / 30.0):
    """按时间均匀采样生成 (cx, cy, t) 序列。"""
    return [(x_fn(t), y_fn(t), t) for i in range(n) for t in [t0 + i * dt]]


def test_speed_rightward_uniform_motion():
    """x 从 100 到 700（60 帧 / 2s）：速度 ≈300 px/s，方向 ≈0°。"""
    points = make_points(60, x_fn=lambda t: 100.0 + 300.0 * t, y_fn=lambda t: 240.0)
    speed_px_s, speed_m_s = estimate_speed(points, window=15, min_points=5, pixels_per_meter=None)
    assert speed_px_s == pytest.approx(300.0, rel=0.05)
    assert speed_m_s is None  # 无标定时不得输出 m/s
    assert estimate_direction(points, window=15) == pytest.approx(0.0, abs=1.0)


def test_direction_leftward():
    """反向（-x）运动方向 ≈180°。"""
    points = make_points(60, x_fn=lambda t: 700.0 - 300.0 * t, y_fn=lambda t: 240.0)
    assert estimate_direction(points, window=15) == pytest.approx(180.0, abs=1.0)
    speed_px_s, _ = estimate_speed(points, 15, 5, None)
    assert speed_px_s == pytest.approx(300.0, rel=0.05)


def test_direction_downward():
    """向下（+y，图像坐标系顺时针）方向 ≈90°。"""
    points = make_points(60, x_fn=lambda t: 300.0, y_fn=lambda t: 100.0 + 200.0 * t)
    assert estimate_direction(points, window=15) == pytest.approx(90.0, abs=1.0)


def test_direction_upward():
    """向上（-y）方向 ≈270°。"""
    points = make_points(60, x_fn=lambda t: 300.0, y_fn=lambda t: 500.0 - 200.0 * t)
    assert estimate_direction(points, window=15) == pytest.approx(270.0, abs=1.0)


def test_insufficient_points_returns_none():
    """点数不足 min_points 时返回 (None, None)。"""
    points = make_points(3, x_fn=lambda t: 100.0 + 300.0 * t, y_fn=lambda t: 240.0)
    assert estimate_speed(points, window=15, min_points=5, pixels_per_meter=None) == (None, None)


def test_degenerate_time_returns_none():
    """时间跨度退化（同一时刻）时不估计。"""
    points = [(100.0 + i, 240.0, 1.0) for i in range(10)]
    assert estimate_direction(points, window=15) is None
    assert estimate_speed(points, 15, 5, None) == (None, None)


def test_pixels_per_meter_conversion():
    """pixels_per_meter 非空时输出 m/s，数值 = px/s ÷ 换算系数。"""
    points = make_points(60, x_fn=lambda t: 100.0 + 300.0 * t, y_fn=lambda t: 240.0)
    speed_px_s, speed_m_s = estimate_speed(points, 15, 5, pixels_per_meter=100.0)
    assert speed_px_s == pytest.approx(300.0, rel=0.05)
    assert speed_m_s == pytest.approx(3.0, rel=0.05)


def test_window_limits_points_used():
    """window 只取最近 N 点：前段反向扰动不影响估计。"""
    dt = 1.0 / 30.0
    # 前 40 帧向左，后 40 帧向右；window=15 时应只看到向右段
    first = make_points(40, x_fn=lambda t: 100.0 - 100.0 * t, y_fn=lambda t: 50.0, dt=dt)
    second = make_points(
        40, x_fn=lambda t: 500.0 + 100.0 * t, y_fn=lambda t: 50.0, t0=40 * dt, dt=dt
    )
    points = first + second
    assert estimate_direction(points, window=15) == pytest.approx(0.0, abs=1.0)


def test_update_history_fills_fields():
    """update_history 就地填充 direction_deg / speed 字段。"""
    from collections import deque

    cfg = MotionConfig(window=15, min_points=5, pixels_per_meter=None)
    history = TrackHistory(track_id=1, points=deque(maxlen=100))
    for i in range(20):
        t = i / 30.0
        history.points.append((100.0 + 150.0 * t, 200.0, t))
    update_history(history, cfg)
    assert history.direction_deg == pytest.approx(0.0, abs=1.0)
    assert history.speed_px_s == pytest.approx(150.0, rel=0.05)
    assert history.speed_m_s is None
