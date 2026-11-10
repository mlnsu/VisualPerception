"""可视化：检测框 / 轨迹 / 标签 / HUD 绘制。

职责边界：只做绘制，不做任何推理或跟踪。
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.config import VisualizationConfig
from src.datatypes import FrameResult

logger = logging.getLogger(__name__)

#: OpenCV 字体（Hershey Simplex，无外部字体依赖）
_FONT = cv2.FONT_HERSHEY_SIMPLEX

#: id -> 色相的散列步长（与 180 互质，保证同 id 跨帧颜色恒定且分布均匀）
_HSV_HUE_STRIDE = 47

#: HUD 半透明底的透明度（值越大越不透明）
_HUD_ALPHA = 0.5
_HUD_BG_COLOR = (0, 0, 0)
_HUD_TEXT_COLOR = (255, 255, 255)
_HUD_MARGIN = 8
_HUD_LINE_SPACING = 4

#: 标签底色与文字色（底色取目标色，文字反白）
_LABEL_TEXT_COLOR = (255, 255, 255)
_LABEL_PADDING = 2

#: 绘制轨迹至少需要的点数
_TRAJECTORY_MIN_POINTS = 2


def color_for(track_id: int) -> tuple[int, int, int]:
    """由 track_id 确定性生成 BGR 颜色（同 id 跨帧恒定）。"""
    hue = (track_id * _HSV_HUE_STRIDE) % 180
    hsv = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


class Visualizer:
    """画面叠加绘制器：框 + 标签 + 轨迹 + HUD。

    只绘制 conf >= display_conf 的目标；
    show() 仅做 imshow + waitKey(1)，headless 场景由调用方跳过。
    """

    def __init__(
        self,
        cfg: VisualizationConfig,
        display_conf: float,
        window_name: str | None = None,
    ) -> None:
        """初始化绘制器。

        Args:
            cfg: 可视化配置。
            display_conf: 绘制过滤阈值（来自 detection.display_conf）。
            window_name: 窗口名覆盖；None 时用 cfg.window_name。
        """
        self._cfg = cfg
        self._display_conf = display_conf
        self._window_name = window_name if window_name else cfg.window_name
        self._window_created = False

    def draw(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """在帧上绘制全部叠加元素（原地修改并返回）。

        Args:
            frame: BGR 帧。
            result: 当前帧感知结果。

        Returns:
            绘制后的帧（同一数组）。
        """
        for track in result.tracks:
            if track.conf < self._display_conf:
                continue
            color = color_for(track.track_id)
            if self._cfg.draw_trajectory:
                history = result.histories.get(track.track_id)
                if history is not None:
                    self._draw_trajectory(frame, history.points, color)
            self._draw_box_and_label(frame, track, result, color)
        if self._cfg.draw_hud:
            self._draw_hud(frame, result)
        return frame

    def _draw_trajectory(
        self, frame: np.ndarray, points, color: tuple[int, int, int]
    ) -> None:
        """用 polylines 连接轨迹点。"""
        pts = list(points)
        if len(pts) < _TRAJECTORY_MIN_POINTS:
            return
        arr = np.asarray(pts, dtype=np.float32)[:, :2]
        arr = np.round(arr).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [arr], isClosed=False, color=color, thickness=self._cfg.line_width)

    def _draw_box_and_label(
        self, frame: np.ndarray, track, result: FrameResult, color: tuple[int, int, int]
    ) -> None:
        """绘制单个目标的框与标签。"""
        x1, y1, x2, y2 = (int(round(v)) for v in track.xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self._cfg.line_width)

        label = f"ID {track.track_id} {track.class_name} {track.conf:.2f}"
        if self._cfg.draw_speed:
            history = result.histories.get(track.track_id)
            if history is not None and history.speed_m_s is not None:
                label += f" {history.speed_m_s:.1f}m/s"
            elif history is not None and history.speed_px_s is not None:
                label += f" {history.speed_px_s:.1f}px/s"

        thickness = max(1, self._cfg.line_width - 1)
        (text_w, text_h), baseline = cv2.getTextSize(
            label, _FONT, self._cfg.font_scale, thickness
        )
        # 标签置于框上方；空间不足时放框内顶部
        top = y1 - text_h - baseline - _LABEL_PADDING
        if top < 0:
            top = y1 + _LABEL_PADDING
        cv2.rectangle(
            frame,
            (x1, top),
            (x1 + text_w, top + text_h + baseline + _LABEL_PADDING),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1, top + text_h),
            _FONT,
            self._cfg.font_scale,
            _LABEL_TEXT_COLOR,
            thickness,
        )

    def _draw_hud(self, frame: np.ndarray, result: FrameResult) -> None:
        """左上角半透明 HUD：FPS / 检测耗时 / 跟踪耗时 / 目标数。"""
        lines = [
            f"FPS: {result.fps:.1f}",
            f"detect: {result.detect_ms:.1f} ms",
            f"track: {result.track_ms:.1f} ms",
            f"tracks: {len(result.tracks)}",
        ]
        thickness = max(1, self._cfg.line_width - 1)
        sizes = [
            cv2.getTextSize(text, _FONT, self._cfg.font_scale, thickness)[0]
            for text in lines
        ]
        text_h = max(size[1] for size in sizes) + _HUD_LINE_SPACING
        width = max(size[0] for size in sizes) + _HUD_MARGIN * 2
        height = text_h * len(lines) + _HUD_MARGIN * 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), _HUD_BG_COLOR, -1)
        cv2.addWeighted(overlay, _HUD_ALPHA, frame, 1.0 - _HUD_ALPHA, 0.0, frame)
        for i, text in enumerate(lines):
            y = _HUD_MARGIN + text_h * i + sizes[i][1]
            cv2.putText(
                frame,
                text,
                (_HUD_MARGIN, y),
                _FONT,
                self._cfg.font_scale,
                _HUD_TEXT_COLOR,
                thickness,
            )

    def show(self, frame: np.ndarray) -> int:
        """显示一帧并返回按键值（waitKey(1) 结果）。headless 场景勿调用。"""
        if not self._cfg.enabled:
            return -1
        cv2.imshow(self._window_name, frame)
        self._window_created = True
        return int(cv2.waitKey(1)) & 0xFF

    def close(self) -> None:
        """销毁窗口（若创建过）。"""
        if self._window_created:
            try:
                cv2.destroyAllWindows()
            except cv2.error as exc:  # 无 GUI 环境下兜底
                logger.debug("destroyAllWindows 失败（忽略）: %s", exc)
            self._window_created = False
