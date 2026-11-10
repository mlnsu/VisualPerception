"""CSV 记录：逐帧写出通过 display_conf 的目标行。"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from src.datatypes import FrameResult

logger = logging.getLogger(__name__)

#: 固定列契约（pandas.read_csv 可直读）
CSV_COLUMNS = [
    "frame_idx",
    "timestamp",
    "track_id",
    "class_id",
    "class_name",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    "conf",
    "direction_deg",
    "speed_px_s",
    "speed_m_s",
    "fps",
]


class CsvLogger:
    """帧级检测/跟踪结果 CSV 写入器。

    只写有目标的行（无目标帧不写）；
    speed/direction 缺失（None）写为空字段，不产生 NaN 混入。
    """

    def __init__(self, path: Path, flush_every: int, display_conf: float, append: bool = False) -> None:
        """初始化写入器。

        Args:
            path: CSV 输出路径（父目录自动创建）。
            flush_every: 每写多少行 flush 一次。
            display_conf: 记录过滤阈值（来自 detection.display_conf）。
            append: True 时追加（保留原表头）；False 覆盖。
        """
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._flush_every = max(1, int(flush_every))
        self._display_conf = display_conf
        self._rows_since_flush = 0
        self._closed = False

        file_exists = self._path.exists()
        self._file = self._path.open(
            "a" if append else "w", newline="", encoding="utf-8"
        )
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS)
        if not (append and file_exists):
            self._writer.writeheader()
        logger.info(
            "CSV 日志已打开: %s（模式=%s）", self._path, "追加" if append else "覆盖"
        )

    def log(self, result: FrameResult) -> None:
        """写出该帧所有 conf >= display_conf 的目标行。"""
        if self._closed:
            return
        for track in result.tracks:
            if track.conf < self._display_conf:
                continue
            history = result.histories.get(track.track_id)
            cx, cy = track.center()
            x1, y1, x2, y2 = (float(v) for v in track.xyxy)
            row = {
                "frame_idx": result.frame_index,
                "timestamp": result.timestamp,
                "track_id": track.track_id,
                "class_id": track.class_id,
                "class_name": track.class_name,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": cx,
                "cy": cy,
                "conf": track.conf,
                "direction_deg": history.direction_deg if history else None,
                "speed_px_s": history.speed_px_s if history else None,
                "speed_m_s": history.speed_m_s if history else None,
                "fps": result.fps,
            }
            self._writer.writerow(row)
            self._rows_since_flush += 1
            if self._rows_since_flush >= self._flush_every:
                self._file.flush()
                self._rows_since_flush = 0

    def close(self) -> None:
        """flush 并关闭文件。幂等。"""
        if self._closed:
            return
        self._closed = True
        try:
            self._file.flush()
            self._file.close()
        except OSError as exc:
            logger.warning("关闭 CSV 文件失败: %s", exc)
