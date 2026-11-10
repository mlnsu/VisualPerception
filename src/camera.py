"""帧采集：FrameSource 统一摄像头与视频文件读帧，ThreadedFrameSource 提供线程化采集。

本模块只负责提供帧，不关心绘制窗口。
摄像头后端选择（resolve_backend/open_capture）、协商参数与实测采集帧率
（measure_capture_fps）也在这里。注意 CAP_PROP_FPS 只是请求值/元数据，
实际帧率必须用 measure_capture_fps 计时实测。
"""

from __future__ import annotations

import logging
import queue
import sys
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

#: measure_capture_fps 连续读帧失败的上限（防止死循环）
_MEASURE_MAX_FAILURES = 30

#: 后端常量 -> 显示名
_BACKEND_DISPLAY: dict[int, str] = {
    cv2.CAP_MSMF: "MSMF",
    cv2.CAP_DSHOW: "DSHOW",
    cv2.CAP_ANY: "DEFAULT",
}


def resolve_backend(cfg: CameraConfig) -> int:
    """按 cfg.backend 决定首选 cv2 后端常量。

    auto 时：Windows 且 source 为 int（实时）-> 先 MSMF；非 Windows 或文件源 -> 0（默认）。
    返回具体常量，并记录 DEBUG 日志说明选择依据。
    """
    backend = cfg.backend.strip().lower()
    is_live = isinstance(cfg.source, int)
    if backend == "msmf":
        chosen, reason = cv2.CAP_MSMF, "配置指定 msmf"
    elif backend == "dshow":
        chosen, reason = cv2.CAP_DSHOW, "配置指定 dshow"
    elif backend == "default":
        chosen, reason = cv2.CAP_ANY, "配置指定 default"
    else:  # auto
        if sys.platform == "win32" and is_live:
            chosen = cv2.CAP_MSMF
            reason = "auto：Windows 实时源优先 MSMF（DSHOW 在 720p 只有约 10 FPS）"
        else:
            chosen = cv2.CAP_ANY
            reason = "auto：非 Windows 或文件源，使用默认后端"
    logger.debug(
        "resolve_backend: %r -> %s（%s）",
        cfg.backend,
        _BACKEND_DISPLAY.get(chosen, str(chosen)),
        reason,
    )
    return chosen


def _backend_reason(cfg: CameraConfig) -> str:
    """返回后端选择的简短原因（用于 INFO 日志）。"""
    backend = cfg.backend.strip().lower()
    if backend == "auto":
        if sys.platform == "win32" and isinstance(cfg.source, int):
            return "auto：Windows 实时源优先 MSMF（DSHOW 在 720p 只有约 10 FPS）"
        return "auto：非 Windows 或文件源，使用默认后端"
    return f"配置 backend={backend}"


def _backend_chain(cfg: CameraConfig) -> list[int]:
    """返回依次尝试的后端常量列表（auto 在 Windows 实时源时带回退链）。"""
    if cfg.backend.strip().lower() == "auto":
        if sys.platform == "win32" and isinstance(cfg.source, int):
            return [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
        return [cv2.CAP_ANY]
    return [resolve_backend(cfg)]


def decode_fourcc(value: float) -> str:
    """把 CAP_PROP_FOURCC 的数值解码为 4 字符串；非 ASCII 可打印或无效时返回空串。"""
    code = int(value) & 0xFFFFFFFF
    if code <= 0:
        return ""
    chars = [chr((code >> (8 * shift)) & 0xFF) for shift in range(4)]
    if not all(0x20 <= ord(c) <= 0x7E for c in chars):
        return ""
    return "".join(chars).strip()


def _configure_capture(cap: cv2.VideoCapture, cfg: CameraConfig) -> None:
    """按配置设置 fourcc/分辨率/帧率/缓冲。

    MSMF 每次会触发重协商的 set 约耗 3~4 秒（Astra 实测），因此先 get 再按需 set；
    BUFFERSIZE 在 MSMF 上是廉价 no-op、在 DSHOW 上生效，保持直接设置。
    """
    if cfg.fourcc and decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)) != cfg.fourcc:
        if not cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc)):
            logger.warning("设置 fourcc=%s 失败（后端可能忽略该属性）", cfg.fourcc)
    if int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) != cfg.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    if int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) != cfg.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    if abs(float(cap.get(cv2.CAP_PROP_FPS)) - cfg.fps) > 0.5:
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.buffer_size)


def open_capture(idx: int, cfg: CameraConfig) -> tuple[cv2.VideoCapture, str]:
    """打开并配置摄像头，返回 (cap, 实际后端名)。

    按候选后端依次尝试；任一环节（打开/配置/协商）失败则 release 并换下一个后端。
    全部失败抛 RuntimeError，错误信息含索引、尝试过的后端列表与每个后端的失败原因。
    """
    chain = _backend_chain(cfg)
    errors: list[str] = []
    for backend in chain:
        display = _BACKEND_DISPLAY.get(backend, str(backend))
        cap = cv2.VideoCapture(idx, backend)
        failure: str | None = None
        width = height = 0.0
        try:
            if not cap.isOpened():
                failure = "打开失败"
            else:
                _configure_capture(cap, cfg)
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if width <= 0 or height <= 0:
                    failure = f"协商分辨率无效({width:g}x{height:g})"
        except Exception as exc:  # noqa: BLE001 - 任一环节异常即换下一个后端
            failure = repr(exc)
        if failure is None:
            logger.info(
                "摄像头 %d 经 %s 打开成功（%s），候选链 %s",
                idx,
                display,
                _backend_reason(cfg),
                "/".join(_BACKEND_DISPLAY.get(b, str(b)) for b in chain),
            )
            return cap, display
        cap.release()
        logger.debug("后端 %s 打开摄像头 %d 失败：%s", display, idx, failure)
        errors.append(f"{display}: {failure}")
    names = "/".join(_BACKEND_DISPLAY.get(b, str(b)) for b in chain)
    raise RuntimeError(
        f"无法打开摄像头 {idx}（尝试后端 {names}）：{'；'.join(errors)}，"
        "请检查设备占用、索引是否正确或后端配置"
    )


def measure_capture_fps(cap: cv2.VideoCapture, frames: int = 60) -> float:
    """连续 read() 计时，返回实际采集 FPS。不睡眠、不画图。

    连续失败达到上限时提前结束；返回值为实际读到的帧数 / 耗时。
    禁止用 CAP_PROP_FPS 代替本实测（它只返回请求值/元数据）。
    """
    if frames <= 0:
        raise ValueError("frames 必须 > 0")
    got = 0
    failures = 0
    start = time.perf_counter()
    while got < frames and failures < _MEASURE_MAX_FAILURES:
        ok, _ = cap.read()
        if ok:
            got += 1
            failures = 0
        else:
            failures += 1
    elapsed = time.perf_counter() - start
    if got == 0 or elapsed <= 0.0:
        return 0.0
    if got < frames:
        logger.warning("采集测速提前结束：仅读到 %d/%d 帧", got, frames)
    return got / elapsed


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
        self._backend_name: str = ""
        self._negotiated: dict[str, float | str] = {}

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

    @property
    def backend_name(self) -> str:
        """实际生效的后端名（MSMF / DSHOW / DEFAULT / 文件）。"""
        return self._backend_name

    @property
    def negotiated(self) -> dict[str, float | str]:
        """协商到的采集参数 {"width", "height", "fps", "fourcc"}。"""
        return dict(self._negotiated)

    @property
    def capture(self) -> cv2.VideoCapture | None:
        """底层 VideoCapture（仅供测速等只读场景，勿直接操作生命周期）。"""
        return self._cap

    def open(self) -> None:
        """打开采集源；失败抛 RuntimeError（含源与原因）。幂等。"""
        if self._cap is not None and self._cap.isOpened():
            return
        self._open_capture()

    def _open_capture(self) -> None:
        """底层打开逻辑，供 open 与重连共用。"""
        if self._is_live:
            index = int(self._cfg.source)
            cap, backend_name = open_capture(index, self._cfg)
            self._cap = cap
            self._backend_name = backend_name
            self._negotiated = {
                "width": float(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": float(cap.get(cv2.CAP_PROP_FPS)),
                "fourcc": decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)),
            }
            neg = self._negotiated
            if int(neg["width"]) != self._cfg.width or int(neg["height"]) != self._cfg.height:
                logger.warning(
                    "协商分辨率 %gx%g 与请求 %dx%d 不一致（后端 %s）",
                    neg["width"],
                    neg["height"],
                    self._cfg.width,
                    self._cfg.height,
                    backend_name,
                )
            logger.info(
                "摄像头 %d 已打开 | 后端: %s | 实际生效 %gx%g @ %.1f fps %s",
                index,
                backend_name,
                neg["width"],
                neg["height"],
                neg["fps"],
                neg["fourcc"] or "(默认格式)",
            )
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
            self._backend_name = "文件"
            self._negotiated = {
                "width": float(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": float(fps),
                "fourcc": decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)),
            }
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
        self._dropped = 0
        self._dropped_lock = threading.Lock()

    @property
    def is_live(self) -> bool:
        """是否为实时摄像头源（透传内部源）。"""
        return self._inner.is_live

    @property
    def fps_hint(self) -> float:
        """帧率提示（透传内部源）。"""
        return self._inner.fps_hint

    @property
    def backend_name(self) -> str:
        """实际生效的后端名（透传内部源）。"""
        return self._inner.backend_name

    @property
    def negotiated(self) -> dict[str, float | str]:
        """协商到的采集参数（透传内部源）。"""
        return self._inner.negotiated

    @property
    def dropped_frames(self) -> int:
        """队列满丢帧的累计计数（丢旧帧 + 极端竞争下放弃的新帧）。"""
        with self._dropped_lock:
            return self._dropped

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
                    with self._dropped_lock:
                        self._dropped += 1
                except queue.Empty:
                    pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                with self._dropped_lock:
                    self._dropped += 1
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
