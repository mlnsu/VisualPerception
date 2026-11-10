"""帧采集：FrameSource 统一摄像头与视频文件读帧，ThreadedFrameSource 提供线程化采集。

本模块只负责提供帧，不关心绘制窗口。
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import cv2
import numpy as np

from src.config import CameraConfig

logger = logging.getLogger(__name__)

#: 摄像头读帧重连的最大失败次数（超过即抛 RuntimeError）
_MAX_RECONNECT_FAILURES = 5

#: 文件源读不到 fps 时的回退值
_DEFAULT_FILE_FPS = 30.0

#: 线程化采集的读帧超时（秒）
_THREADED_READ_TIMEOUT_S = 10.0


class FrameSource:
    """统一的帧采集源：int 源为摄像头，str 源为视频文件或 URL。

    用法::

        with FrameSource(cfg) as source:
            ok, frame = source.read()
    """

    def __init__(self, cfg: CameraConfig) -> None:
        self._cfg = cfg
        self._cap: cv2.VideoCapture | None = None
        self._is_live = isinstance(cfg.source, int)
        self._file_fps: float | None = None

    @property
    def is_live(self) -> bool:
        """是否为实时摄像头源。"""
        return self._is_live

    @property
    def fps_hint(self) -> float:
        """帧率提示：摄像头返回配置值，文件返回文件自身 fps。"""
        if self._is_live:
            return float(self._cfg.fps)
        if self._file_fps is not None:
            return self._file_fps
        return _DEFAULT_FILE_FPS

    def open(self) -> None:
        """打开采集源；失败抛 RuntimeError（含源与原因）。幂等。"""
        if self._cap is not None and self._cap.isOpened():
            return
        self._open_capture()

    def _open_capture(self) -> None:
        """底层打开逻辑，供 open 与重连共用。"""
        if self._is_live:
            index = int(self._cfg.source)
            if self._cfg.backend.strip().lower() == "dshow":
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(
                    f"无法打开摄像头 {index}（backend={self._cfg.backend}），"
                    "请检查设备占用或索引是否正确"
                )
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
            cap.set(cv2.CAP_PROP_FPS, self._cfg.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self._cfg.buffer_size)
            logger.info(
                "摄像头 %d 已打开，实际生效 %dx%d @ %.1f fps",
                index,
                cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                cap.get(cv2.CAP_PROP_FPS),
            )
            self._cap = cap
        else:
            path = str(self._cfg.source)
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"无法打开视频源: {path}（文件不存在或编码不支持）")
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if not 0.0 < fps < 1000.0:
                logger.warning("视频 %s 的 fps 元数据无效(%.2f)，回退为 %.1f", path, fps, _DEFAULT_FILE_FPS)
                fps = _DEFAULT_FILE_FPS
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info("视频源已打开: %s（%.2f fps，约 %d 帧）", path, fps, frames)
            self._file_fps = fps
            self._cap = cap

    def read(self) -> tuple[bool, np.ndarray | None]:
        """读取一帧。

        Returns:
            (True, frame) 成功；(False, None) 结束（文件播完 / 不可恢复的失败）。

        Raises:
            RuntimeError: 摄像头按配置重连后仍连续失败超过上限。
        """
        if self._cap is None or not self._cap.isOpened():
            if self._is_live and self._cfg.reconnect:
                return self._read_with_reconnect()
            logger.warning("FrameSource 未打开或已关闭，read 返回 (False, None)")
            return False, None

        ok, frame = self._cap.read()
        if ok:
            return True, frame

        if self._is_live:
            if self._cfg.reconnect:
                return self._read_with_reconnect()
            return False, None

        # 文件源：到达末尾
        if self._cfg.loop_video:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
            if ok:
                return True, frame
        return False, None

    def _read_with_reconnect(self) -> tuple[bool, np.ndarray | None]:
        """摄像头读帧失败后的重连循环。"""
        failures = 0
        while True:
            if self._cap is not None and self._cap.isOpened():
                ok, frame = self._cap.read()
                if ok:
                    return True, frame
            failures += 1
            if failures > _MAX_RECONNECT_FAILURES:
                raise RuntimeError(
                    f"摄像头(源 {self._cfg.source})连续读帧失败超过 "
                    f"{_MAX_RECONNECT_FAILURES} 次，放弃重连"
                )
            logger.warning(
                "摄像头读帧失败，%.1fs 后重试（第 %d/%d 次）",
                self._cfg.reconnect_interval_s,
                failures,
                _MAX_RECONNECT_FAILURES,
            )
            time.sleep(self._cfg.reconnect_interval_s)
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            try:
                self._open_capture()
            except RuntimeError as exc:
                logger.warning("重连失败: %s", exc)

    def close(self) -> None:
        """释放采集资源。幂等。"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("FrameSource 已关闭")

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class ThreadedFrameSource:
    """实时摄像头的线程化采集包装：后台线程读帧，队列满则丢旧帧。

    仅用于摄像头路径以保证低延迟；文件源不启用线程（保证测试确定性）。
    接口与 :class:`FrameSource` 一致。
    """

    def __init__(self, inner: FrameSource, maxsize: int = 2) -> None:
        self._inner = inner
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()

    @property
    def is_live(self) -> bool:
        """是否为实时摄像头源（透传内部源）。"""
        return self._inner.is_live

    @property
    def fps_hint(self) -> float:
        """帧率提示（透传内部源）。"""
        return self._inner.fps_hint

    def open(self) -> None:
        """打开内部源并启动采集线程。"""
        self._inner.open()
        self._stopped.clear()
        # 清空可能残留的旧帧
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._thread = threading.Thread(target=self._reader_loop, name="frame-reader", daemon=True)
        self._thread.start()
        logger.info("线程化采集已启动（队列容量 %d，满则丢旧帧）", self._queue.maxsize)

    def _reader_loop(self) -> None:
        """后台读帧线程主循环。"""
        while not self._stopped.is_set():
            try:
                ok, frame = self._inner.read()
            except RuntimeError as exc:
                logger.error("采集线程终止: %s", exc)
                self._queue.put(None)
                return
            if not ok:
                self._queue.put(None)
                return
            if self._queue.full():
                try:
                    self._queue.get_nowait()  # 丢旧帧保低延迟
                except queue.Empty:
                    pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                continue  # 极端竞争下放弃本帧

    def read(self, timeout: float = _THREADED_READ_TIMEOUT_S) -> tuple[bool, np.ndarray | None]:
        """从队列取一帧；超时或流结束返回 (False, None)。"""
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            logger.warning("线程化采集读帧超时（%.1fs）", timeout)
            return False, None
        if item is None:
            return False, None
        return True, item

    def close(self) -> None:
        """停止采集线程并释放内部源。"""
        self._stopped.set()
        # 排空队列，让可能阻塞在 put 上的线程有机会退出
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._inner.close()

    def __enter__(self) -> "ThreadedFrameSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
