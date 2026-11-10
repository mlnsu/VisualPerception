"""多目标跟踪：supervision ByteTrack 独立封装。

架构边界：检测器只出框，本模块独立完成跟踪；
禁止使用 model.track() / persist=True。
"""

from __future__ import annotations

import logging

import numpy as np
import supervision as sv

from src.config import TrackingConfig
from src.datatypes import Detection, Track

logger = logging.getLogger(__name__)

#: supervision 的 tracker_id 编号起点因版本而异，统一 +1 保证 >= 1
_ID_OFFSET = 1


class Tracker:
    """ByteTrack 跟踪器封装。

    tracking.enabled 为 False 时退化为「每个检测分配临时 id 且逐帧递增」，
    便于 A/B 对比跟踪效果。
    """

    def __init__(self, cfg: TrackingConfig) -> None:
        """初始化跟踪器。

        Args:
            cfg: 跟踪配置。
        """
        self._cfg = cfg
        self._byte_track: sv.ByteTrack | None = None
        self._next_temp_id = 1
        if cfg.enabled:
            self._create_byte_track()

    def _create_byte_track(self) -> None:
        """按配置新建 ByteTrack 实例（也用于 reset）。"""
        self._byte_track = sv.ByteTrack(
            track_activation_threshold=self._cfg.track_activation_threshold,
            lost_track_buffer=self._cfg.lost_track_buffer,
            minimum_matching_threshold=self._cfg.minimum_matching_threshold,
            minimum_consecutive_frames=self._cfg.minimum_consecutive_frames,
        )

    @staticmethod
    def _to_sv_detections(detections: list[Detection]) -> sv.Detections:
        """把 list[Detection] 组装为 sv.Detections（字段一一对应）。"""
        if not detections:
            return sv.Detections.empty()
        xyxy = np.stack([np.asarray(d.xyxy, dtype=np.float32) for d in detections], axis=0)
        confidence = np.asarray([d.conf for d in detections], dtype=np.float32)
        class_id = np.asarray([d.class_id for d in detections], dtype=int)
        return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

    def update(self, detections: list[Detection]) -> list[Track]:
        """推进一帧跟踪。每帧必须调用（无检测时传空列表）。

        Args:
            detections: 当前帧检测列表（可为空）。

        Returns:
            当前帧的跟踪目标列表。
        """
        if not self._cfg.enabled:
            return self._update_passthrough(detections)

        assert self._byte_track is not None
        tracked = self._byte_track.update_with_detections(self._to_sv_detections(detections))
        # class_id -> class_name 映射来自当前帧输入
        name_by_class = {d.class_id: d.class_name for d in detections}
        tracks: list[Track] = []
        count = len(tracked)
        if count == 0 or tracked.tracker_id is None:
            return tracks
        for i in range(count):
            raw_id = tracked.tracker_id[i]
            if raw_id is None:
                continue
            class_id = int(tracked.class_id[i])
            tracks.append(
                Track(
                    track_id=int(raw_id) + _ID_OFFSET,
                    xyxy=np.asarray(tracked.xyxy[i], dtype=np.float32),
                    conf=float(tracked.confidence[i]),
                    class_id=class_id,
                    class_name=name_by_class.get(class_id, f"class_{class_id}"),
                )
            )
        return tracks

    def _update_passthrough(self, detections: list[Detection]) -> list[Track]:
        """降级路径：不跟踪，每个检测分配逐帧递增的临时 id。"""
        tracks: list[Track] = []
        for det in detections:
            tracks.append(
                Track(
                    track_id=self._next_temp_id,
                    xyxy=np.asarray(det.xyxy, dtype=np.float32),
                    conf=float(det.conf),
                    class_id=det.class_id,
                    class_name=det.class_name,
                )
            )
            self._next_temp_id += 1
        return tracks

    def reset(self) -> None:
        """重置跟踪状态（跨视频切换时调用）。"""
        if self._cfg.enabled:
            self._create_byte_track()
        self._next_temp_id = 1
        logger.info("跟踪器状态已重置")
