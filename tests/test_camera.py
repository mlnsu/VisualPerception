"""采集层离线测试：camera.py / devices.py / main.py 的采集相关纯逻辑。

全部用 monkeypatch 伪造 cv2.VideoCapture，不打开任何真实摄像头。
"""

from __future__ import annotations

import sys
import time
import types

import cv2
import numpy as np
import pytest

import src.camera as camera_mod
import src.devices as devices_mod
import src.main as main_mod
from src.camera import (
    FrameSource,
    ThreadedFrameSource,
    measure_capture_fps,
    open_capture,
    resolve_backend,
)
from src.config import PROJECT_ROOT, CameraConfig
from src.devices import resolve_index_by_name

#: Windows 专属回退链测试的跳过条件
_SKIP_NON_WIN = pytest.mark.skipif(sys.platform != "win32", reason="回退链仅在 Windows 生效")


def make_camera_cfg(
    source: int | str = 1,
    backend: str = "auto",
    width: int = 1280,
    height: int = 720,
    fourcc: str = "",
) -> CameraConfig:
    """构造测试用 CameraConfig（不依赖配置文件）。"""
    return CameraConfig(
        source=source,
        device_name="",
        backend=backend,
        width=width,
        height=height,
        fps=30.0,
        fourcc=fourcc,
        buffer_size=1,
        loop_video=False,
        reconnect=True,
        reconnect_interval_s=2.0,
        probe_max_index=4,
    )


# ---------------------------------------------------------------- resolve_backend


def test_resolve_backend_explicit():
    """显式 backend 映射到对应 cv2 常量。"""
    assert resolve_backend(make_camera_cfg(backend="msmf")) == cv2.CAP_MSMF
    assert resolve_backend(make_camera_cfg(backend="dshow")) == cv2.CAP_DSHOW
    assert resolve_backend(make_camera_cfg(backend="default")) == cv2.CAP_ANY == 0


def test_resolve_backend_auto():
    """auto：Windows 实时源 -> 非 0 常量（MSMF）；文件源 -> 0（默认）。"""
    live = resolve_backend(make_camera_cfg(source=1, backend="auto"))
    if sys.platform == "win32":
        assert live != 0
    else:
        assert live == 0
    assert resolve_backend(make_camera_cfg(source="x.mp4", backend="auto")) == 0


# ---------------------------------------------------------------- open_capture


class FakeCap:
    """伪造 VideoCapture：按后端常量决定可否打开，尺寸可协商。"""

    def __init__(self, idx: int, backend: int, open_ok: bool, size: tuple[int, int] = (640, 480)):
        self.idx = idx
        self.backend = backend
        self._open_ok = open_ok
        self._size = size
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - 与 cv2 API 命名一致
        return self._open_ok and not self.released

    def set(self, prop: int, value: float) -> bool:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            self._size = (int(value), self._size[1])
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            self._size = (self._size[0], int(value))
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._size[0])
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._size[1])
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        return 0.0

    def read(self):
        return self.isOpened(), None

    def release(self) -> None:
        self.released = True


def _patch_videocapture(monkeypatch, open_backends: set[int]) -> list[FakeCap]:
    """把 cv2.VideoCapture 换成按后端集合决定成败的工厂，返回创建的实例列表。"""
    created: list[FakeCap] = []

    def factory(idx: int, backend: int = cv2.CAP_ANY) -> FakeCap:
        cap = FakeCap(idx, backend, open_ok=backend in open_backends)
        created.append(cap)
        return cap

    monkeypatch.setattr(cv2, "VideoCapture", factory)
    return created


@_SKIP_NON_WIN
def test_open_capture_auto_fallback(monkeypatch):
    """auto 链上 MSMF 打开失败、DSHOW 成功：返回 DSHOW，失败后端被 release。"""
    created = _patch_videocapture(monkeypatch, {cv2.CAP_DSHOW})
    cap, backend_name = open_capture(1, make_camera_cfg(source=1, backend="auto"))
    assert backend_name == "DSHOW"
    assert not cap.released
    assert created[0].backend == cv2.CAP_MSMF and created[0].released
    assert created[1].backend == cv2.CAP_DSHOW and not created[1].released


@_SKIP_NON_WIN
def test_open_capture_all_backends_fail(monkeypatch):
    """全部后端失败：抛 RuntimeError，错误含索引与后端列表，且无资源泄漏。"""
    created = _patch_videocapture(monkeypatch, set())
    with pytest.raises(RuntimeError) as excinfo:
        open_capture(1, make_camera_cfg(source=1, backend="auto"))
    message = str(excinfo.value)
    assert "1" in message
    assert "MSMF" in message and "DSHOW" in message
    assert all(cap.released for cap in created)


def test_open_capture_configures_resolution(monkeypatch):
    """显式后端成功时按配置协商分辨率，并回报后端名。"""
    _patch_videocapture(monkeypatch, {cv2.CAP_DSHOW})
    cap, backend_name = open_capture(2, make_camera_cfg(source=2, backend="dshow"))
    assert backend_name == "DSHOW"
    assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 1280.0
    assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == 720.0


@_SKIP_NON_WIN
def test_frame_source_reports_backend_and_negotiated(monkeypatch):
    """FrameSource 摄像头分支回报实际后端名与协商参数。"""
    _patch_videocapture(monkeypatch, {cv2.CAP_MSMF})
    source = FrameSource(make_camera_cfg(source=1))
    source.open()
    try:
        assert source.backend_name == "MSMF"
        negotiated = source.negotiated
        assert negotiated["width"] == 1280.0
        assert negotiated["height"] == 720.0
        assert negotiated["fps"] == 30.0
    finally:
        source.close()


# ---------------------------------------------------------------- measure_capture_fps


def test_measure_capture_fps_fixed_clock(monkeypatch):
    """伪造每次 read() 前进 10ms：实测 FPS 应为 100，误差 <5%。"""
    clock = {"t": 1000.0}

    class FakeTimedCap:
        def read(self):
            clock["t"] += 0.01
            return True, None

    fake_time = types.SimpleNamespace(perf_counter=lambda: clock["t"])
    monkeypatch.setattr(camera_mod, "time", fake_time)
    fps = measure_capture_fps(FakeTimedCap(), frames=60)
    assert fps == pytest.approx(100.0, rel=0.05)


def test_measure_capture_fps_never_ok(monkeypatch):
    """一帧都读不到时返回 0.0（不抛异常、不死循环）。"""
    clock = {"t": 1000.0}

    class FakeDeadCap:
        def read(self):
            clock["t"] += 0.01
            return False, None

    fake_time = types.SimpleNamespace(perf_counter=lambda: clock["t"])
    monkeypatch.setattr(camera_mod, "time", fake_time)
    assert measure_capture_fps(FakeDeadCap(), frames=10) == 0.0


# ---------------------------------------------------------------- dropped_frames


class StubSource:
    """可数的桩帧源：产出 total 帧后结束，帧内携带可识别序号。"""

    def __init__(self, total: int) -> None:
        self._total = total
        self._i = 0
        self.is_live = True
        self.fps_hint = 30.0

    def open(self) -> None:
        pass

    def read(self):
        if self._i >= self._total:
            return False, None
        self._i += 1
        return True, np.full((2, 2), self._i, dtype=np.int32)

    def close(self) -> None:
        pass


def test_threaded_source_drops_old_frames():
    """生产远快于消费：dropped_frames > 0，且消费者拿到的是最新帧。"""
    source = ThreadedFrameSource(StubSource(300), maxsize=2)
    with source:
        time.sleep(0.5)  # 等生产线程读完 300 帧（队列中只剩最新 2 帧）
        ok1, frame1 = source.read(timeout=5.0)
        ok2, frame2 = source.read(timeout=5.0)
    assert ok1 and frame1 is not None
    assert ok2 and frame2 is not None
    v1, v2 = int(frame1[0, 0]), int(frame2[0, 0])
    assert source.dropped_frames > 0
    assert v1 >= 299 and v2 >= 299  # "取最新"语义：拿到的是最后两帧
    assert v2 >= v1


# ---------------------------------------------------------------- resolve_cli_source


def test_resolve_cli_source_digit():
    """纯数字 -> 摄像头索引。"""
    assert main_mod.resolve_cli_source("1") == 1


def test_resolve_cli_source_video_path(monkeypatch):
    """非设备名 -> 视频路径（相对项目根解析）。"""
    monkeypatch.setattr(main_mod, "resolve_index_by_name", lambda raw, mx: None)
    resolved = main_mod.resolve_cli_source("data/videos/x.mp4")
    assert resolved == str(PROJECT_ROOT / "data" / "videos" / "x.mp4")


def test_resolve_cli_source_device_name(monkeypatch):
    """设备名命中 -> 摄像头索引（优先于按路径处理）。"""
    monkeypatch.setattr(main_mod, "resolve_index_by_name", lambda raw, mx: 1)
    assert main_mod.resolve_cli_source("Astra Pro HD Camera") == 1


# ---------------------------------------------------------------- resolve_index_by_name


def test_resolve_index_by_name_case_insensitive(monkeypatch):
    """大小写不敏感的包含匹配。"""
    monkeypatch.setattr(
        devices_mod,
        "device_names",
        lambda: ["XiaoMi USB 2.0 Webcam", "Astra Pro HD Camera", "OBS Virtual Camera"],
    )
    assert resolve_index_by_name("ASTRA PRO", 4) == 1


def test_resolve_index_by_name_no_hit(monkeypatch):
    """无命中返回 None。"""
    monkeypatch.setattr(devices_mod, "device_names", lambda: ["XiaoMi USB 2.0 Webcam"])
    assert resolve_index_by_name("Astra Pro HD Camera", 4) is None


def test_resolve_index_by_name_multi_hit_smallest(monkeypatch, caplog):
    """多个命中：取最小索引并告警列出全部候选。"""
    monkeypatch.setattr(devices_mod, "device_names", lambda: ["Astra Cam A", "Astra Cam B"])
    with caplog.at_level("WARNING"):
        assert resolve_index_by_name("astra", 4) == 0
    assert any("候选" in record.getMessage() for record in caplog.records)
