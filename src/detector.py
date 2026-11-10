"""YOLO 检测封装：只输出检测框，不做跟踪、不做绘制。"""

from __future__ import annotations

import logging

import numpy as np
from ultralytics import YOLO

from src.config import DetectionConfig, ModelConfig
from src.datatypes import Detection

logger = logging.getLogger(__name__)


def _class_name(names: object, class_id: int) -> str:
    """从模型 names 映射取类别名，兼容 dict 与 list 两种形式。"""
    if isinstance(names, dict):
        return str(names.get(class_id, f"class_{class_id}"))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return f"class_{class_id}"


class Detector:
    """YOLO11 检测器（ultralytics 封装）。

    职责边界：仅把帧转换为 ``list[Detection]``；
    跟踪由 :mod:`src.tracker` 独立完成，本模块不画图、不缓存状态。
    """

    def __init__(self, cfg: ModelConfig, det_cfg: DetectionConfig, device: str) -> None:
        """初始化检测器。

        Args:
            cfg: 模型配置（权重路径、imgsz 等）。
            det_cfg: 检测配置（conf/iou/max_det/classes）。
            device: 推理设备（如 "cpu" / "cuda:0"，需已解析，不含 "auto"）。

        Raises:
            RuntimeError: 权重加载失败（缺失且无法下载等）。
        """
        self._cfg = cfg
        self._det_cfg = det_cfg
        self._device = device
        # 空列表表示不过滤类别，ultralytics 约定传 None
        self._classes: list[int] | None = list(det_cfg.classes) if det_cfg.classes else None
        try:
            self._model = YOLO(str(cfg.weights))
        except Exception as exc:  # noqa: BLE001 - 统一转为清晰错误
            raise RuntimeError(
                f"加载 YOLO 权重失败: {cfg.weights}（{exc}）。"
                "请确认文件存在，或联网自动下载。"
            ) from exc

    def warmup(self) -> None:
        """跑一次空推理，避免首帧计时失真。失败仅记警告。"""
        try:
            dummy = np.zeros((self._cfg.imgsz, self._cfg.imgsz, 3), dtype=np.uint8)
            self.detect(dummy)
            logger.info("检测器预热完成（device=%s, imgsz=%d）", self._device, self._cfg.imgsz)
        except Exception as exc:  # noqa: BLE001
            logger.warning("检测器预热失败（忽略，继续运行）: %s", exc)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """对单帧执行检测。

        Args:
            frame: BGR 帧，形状 (H, W, 3)。

        Returns:
            检测列表；坐标已裁剪到帧内，x2<=x1 或 y2<=y1 的框已丢弃。
        """
        height, width = frame.shape[:2]
        predict_kwargs: dict = {
            "conf": self._det_cfg.conf,
            "iou": self._det_cfg.iou,
            "imgsz": self._cfg.imgsz,
            "max_det": self._det_cfg.max_det,
            "classes": self._classes,
            "verbose": self._cfg.verbose,
            "device": self._device,
        }
        if self._cfg.half:
            predict_kwargs["half"] = True  # ultralytics 8.4+ 已弃用 half，仅在显式开启时传参
        results = self._model.predict(frame, **predict_kwargs)
        detections: list[Detection] = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None:
            return detections

        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy()
        names = self._model.names

        for i in range(xyxy.shape[0]):
            x1 = float(np.clip(xyxy[i, 0], 0.0, float(width)))
            y1 = float(np.clip(xyxy[i, 1], 0.0, float(height)))
            x2 = float(np.clip(xyxy[i, 2], 0.0, float(width)))
            y2 = float(np.clip(xyxy[i, 3], 0.0, float(height)))
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = int(cls[i])
            detections.append(
                Detection(
                    xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
                    conf=float(conf[i]),
                    class_id=class_id,
                    class_name=_class_name(names, class_id),
                )
            )
        return detections
