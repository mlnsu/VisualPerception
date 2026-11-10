"""detector.py 测试：monkeypatch 伪造 ultralytics 输出，不加载真实权重。"""

from __future__ import annotations

import numpy as np
import pytest

import src.detector
from src.detector import Detector


class FakeTensor:
    """模拟 torch.Tensor 的 detach().cpu().numpy() 链。"""

    def __init__(self, array):
        self._array = np.asarray(array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class FakeBoxes:
    """模拟 ultralytics 的 Results.boxes。"""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = FakeTensor(xyxy)
        self.conf = FakeTensor(conf)
        self.cls = FakeTensor(cls)


class FakeResult:
    """模拟 ultralytics 的 Results。"""

    def __init__(self, boxes):
        self.boxes = boxes


class FakeYOLO:
    """伪造 YOLO：记录 predict 参数并返回脚本化结果。"""

    names = {0: "person", 1: "bicycle"}

    def __init__(self, weights: str):
        self.weights = weights
        self.predict_calls: list[dict] = []
        self.script: list[list[FakeBoxes]] = []

    def set_script(self, script: list[list[FakeBoxes]]) -> None:
        self.script = script

    def predict(self, source, **kwargs):
        self.predict_calls.append(kwargs)
        if self.script:
            return [FakeResult(self.script.pop(0))]
        return [FakeResult(FakeBoxes(np.empty((0, 4)), np.empty((0,)), np.empty((0,))))]


@pytest.fixture
def fake_yolo(monkeypatch):
    """注入伪造 YOLO 类并返回其可控制实例钩子。"""
    created: list[FakeYOLO] = []

    def factory(weights: str):
        model = FakeYOLO(weights)
        created.append(model)
        return model

    monkeypatch.setattr(src.detector, "YOLO", factory)
    return created


def make_detector(fake_yolo_list, model_cfg, detection_cfg, boxes_script=None):
    """构造接伪造模型的 Detector。"""
    detector = Detector(model_cfg, detection_cfg, device="cpu")
    if boxes_script is not None:
        fake_yolo_list[0].set_script(boxes_script)
    return detector


def test_basic_detection(fake_yolo, model_cfg, detection_cfg):
    """正常检测：字段、类型与 class_name 映射正确。"""
    det = make_detector(
        fake_yolo,
        model_cfg,
        detection_cfg,
        boxes_script=[FakeBoxes([[100, 100, 200, 240]], [0.88], [0])],
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = det.detect(frame)
    assert len(detections) == 1
    d = detections[0]
    assert d.xyxy.dtype == np.float32
    assert d.xyxy.tolist() == [100.0, 100.0, 200.0, 240.0]
    assert d.conf == pytest.approx(0.88)
    assert d.class_id == 0
    assert d.class_name == "person"
    assert d.center() == (150.0, 170.0)
    assert d.width() == pytest.approx(100.0)
    assert d.height() == pytest.approx(140.0)


def test_coordinates_clipped_to_frame(fake_yolo, model_cfg, detection_cfg):
    """越界坐标被裁剪到帧尺寸内。"""
    det = make_detector(
        fake_yolo,
        model_cfg,
        detection_cfg,
        boxes_script=[FakeBoxes([[-20, -10, 700, 999]], [0.9], [0])],
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = det.detect(frame)
    assert len(detections) == 1
    x1, y1, x2, y2 = detections[0].xyxy.tolist()
    assert x1 == 0.0 and y1 == 0.0
    assert x2 == 640.0 and y2 == 480.0


def test_invalid_boxes_dropped(fake_yolo, model_cfg, detection_cfg):
    """x2<=x1 或 y2<=y1 的框被丢弃。"""
    det = make_detector(
        fake_yolo,
        model_cfg,
        detection_cfg,
        boxes_script=[
            FakeBoxes(
                [[50, 50, 50, 200], [80, 200, 200, 100], [10, 10, 90, 90]],
                [0.9, 0.8, 0.7],
                [0, 0, 0],
            )
        ],
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = det.detect(frame)
    assert len(detections) == 1
    assert detections[0].xyxy.tolist() == [10.0, 10.0, 90.0, 90.0]


def test_empty_result(fake_yolo, model_cfg, detection_cfg):
    """无检测帧返回空列表，不抛异常。"""
    det = make_detector(fake_yolo, model_cfg, detection_cfg)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert det.detect(frame) == []


def test_classes_empty_list_passes_none(fake_yolo, model_cfg, detection_cfg):
    """classes 为空列表时向 predict 传 None（不过滤）。"""
    cfg = type(detection_cfg)(
        conf=detection_cfg.conf,
        iou=detection_cfg.iou,
        max_det=detection_cfg.max_det,
        classes=[],
        display_conf=detection_cfg.display_conf,
    )
    det = make_detector(fake_yolo, model_cfg, cfg)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det.detect(frame)
    kwargs = fake_yolo[0].predict_calls[0]
    assert kwargs["classes"] is None


def test_predict_receives_thresholds(fake_yolo, model_cfg, detection_cfg):
    """conf/iou/imgsz 等阈值按配置透传给 predict。"""
    det = make_detector(fake_yolo, model_cfg, detection_cfg)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det.detect(frame)
    kwargs = fake_yolo[0].predict_calls[0]
    assert kwargs["conf"] == pytest.approx(detection_cfg.conf)
    assert kwargs["iou"] == pytest.approx(detection_cfg.iou)
    assert kwargs["imgsz"] == model_cfg.imgsz
    assert kwargs["max_det"] == detection_cfg.max_det
    assert kwargs["classes"] == detection_cfg.classes
    assert kwargs["verbose"] is False
    assert kwargs["device"] == "cpu"
