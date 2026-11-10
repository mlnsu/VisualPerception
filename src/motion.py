"""运动估计：最小二乘线性拟合方向与速度。

V1 无标定：pixels_per_meter 为 None 时只输出 px/s，绝不隐式假设 m/s。
方向定义：图像平面 x 轴正向为 0°，顺时针为正（y 轴向下），范围 [0, 360)。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from src.config import MotionConfig
from src.datatypes import Point, TrackHistory

#: 时间跨度小于该值（秒）视为退化，不估计
_MIN_TIME_SPAN_S = 1e-9


def _recent_points(points: Sequence[Point], window: int) -> list[Point]:
    """取最近 window 个轨迹点；window <= 0 表示全部。"""
    pts = list(points)
    if window > 0:
        pts = pts[-window:]
    return pts


def _fit_velocity(pts: list[Point]) -> tuple[float, float] | None:
    """对 (cx, cy, t) 序列做最小二乘线性拟合，返回 (vx, vy) px/s。

    点数不足或时间跨度退化时返回 None。
    """
    if len(pts) < 2:
        return None
    ts = np.asarray([p[2] for p in pts], dtype=np.float64)
    if float(ts.max() - ts.min()) < _MIN_TIME_SPAN_S:
        return None
    xs = np.asarray([p[0] for p in pts], dtype=np.float64)
    ys = np.asarray([p[1] for p in pts], dtype=np.float64)
    vx = float(np.polyfit(ts, xs, 1)[0])
    vy = float(np.polyfit(ts, ys, 1)[0])
    return vx, vy


def estimate_direction(points: Sequence[Point], window: int) -> float | None:
    """估计运动方向（度，[0, 360)，x 轴正向为 0°，顺时针为正）。

    Args:
        points: 轨迹点序列 (cx, cy, t)。
        window: 参与估计的最近点数窗口。

    Returns:
        方向角；点数不足、时间退化或速度为零时为 None。
    """
    velocity = _fit_velocity(_recent_points(points, window))
    if velocity is None:
        return None
    vx, vy = velocity
    if vx == 0.0 and vy == 0.0:
        return None
    return math.degrees(math.atan2(vy, vx)) % 360.0


def estimate_speed(
    points: Sequence[Point],
    window: int,
    min_points: int,
    pixels_per_meter: float | None,
) -> tuple[float | None, float | None]:
    """估计速度。

    Args:
        points: 轨迹点序列 (cx, cy, t)。
        window: 参与估计的最近点数窗口。
        min_points: 少于该点数不估计。
        pixels_per_meter: 像素/米换算；None 时不输出 m/s。

    Returns:
        (speed_px_s, speed_m_s)；无法估计时均为 None，
        pixels_per_meter 为 None 时 speed_m_s 恒为 None。
    """
    pts = _recent_points(points, window)
    if len(pts) < max(2, int(min_points)):
        return None, None
    velocity = _fit_velocity(pts)
    if velocity is None:
        return None, None
    speed_px_s = math.hypot(velocity[0], velocity[1])
    speed_m_s = speed_px_s / pixels_per_meter if pixels_per_meter else None
    return speed_px_s, speed_m_s


def update_history(history: TrackHistory, cfg: MotionConfig) -> None:
    """就地更新单条轨迹的方向与速度字段。

    Args:
        history: 待更新的轨迹历史。
        cfg: 运动估计配置。
    """
    history.direction_deg = estimate_direction(history.points, cfg.window)
    history.speed_px_s, history.speed_m_s = estimate_speed(
        history.points, cfg.window, cfg.min_points, cfg.pixels_per_meter
    )
