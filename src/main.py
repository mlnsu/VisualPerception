"""CLI 入口：组装各模块并驱动主循环。

支持 ``python src/main.py`` 与 ``python -m src.main`` 两种运行方式。
除最终摘要外，运行期输出一律走 logging。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 以 `python src/main.py` 方式运行时，保证仓库根在 sys.path 中
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import dataclasses
import logging
import time
from datetime import datetime

import cv2
import numpy as np

from src.camera import FrameSource, ThreadedFrameSource, measure_capture_fps
from src.config import PROJECT_ROOT, Config, load_config
from src.csv_logger import CsvLogger
from src.devices import (
    device_names,
    get_device_name,
    list_devices,
    resolve_index_by_name,
)
from src.detector import Detector
from src.pipeline import PerceptionPipeline
from src.tracker import Tracker
from src.trajectory import TrajectoryManager
from src.visualizer import Visualizer

logger = logging.getLogger("src.main")

#: 退出按键：q / Q / ESC
_EXIT_KEYS = frozenset((ord("q"), ord("Q"), 27))

#: 采集测速前的预热帧数（MSMF 首帧出流约有 0.5s 启动延迟，避免污染计时）
_CAPTURE_WARMUP_FRAMES = 5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 CLI 参数。"""
    parser = argparse.ArgumentParser(
        prog="src.main", description="VisualPerception V1 实时感知 CLI"
    )
    parser.add_argument("--config", type=str, default=None, help="配置文件路径（默认 configs/config.yaml）")
    parser.add_argument("--source", type=str, default=None, help="覆盖采集源：摄像头索引（如 0）、设备名或视频路径")
    parser.add_argument("--list-cameras", action="store_true", help="列出可用视频输入设备后退出（不加载模型）")
    parser.add_argument("--camera-name", type=str, default=None, help="按设备名打开摄像头（优先级高于配置 device_name）")
    parser.add_argument(
        "--bench-capture",
        type=int,
        default=None,
        metavar="N",
        help="只采集不推理：实测连续 N 帧的采集 FPS 后退出（不加载模型）",
    )
    parser.add_argument("--headless", action="store_true", help="无窗口模式（服务器/自动化自检用）")
    parser.add_argument("--max-frames", type=int, default=None, help="最多处理帧数")
    parser.add_argument("--no-log", action="store_true", help="禁用 CSV 记录")
    parser.add_argument("--save-video", action="store_true", help="把标注后的画面写为 mp4")
    parser.add_argument("--append", action="store_true", help="CSV 追加模式（默认覆盖）")
    parser.add_argument("--device", type=str, default=None, help="覆盖推理设备（cpu / cuda:0 / auto）")
    parser.add_argument(
        "--log-level", type=str, default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="覆盖日志级别",
    )
    return parser.parse_args(argv)


def setup_logging(level: str) -> None:
    """初始化根 logger。"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_device(device: str) -> str:
    """把 "auto" 解析为实际设备名。"""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def resolve_cli_source(raw: str, probe_max: int | None = None) -> int | str:
    """解析 --source：纯数字为索引；否则先按设备名匹配；最后按路径。

    路径口径：绝对路径 -> 相对 CWD -> 相对项目根。
    """
    if raw.isdigit():
        return int(raw)
    if probe_max is None:
        probe_max = 4  # 与 DEFAULTS camera.probe_max_index 一致的兜底值
    index = resolve_index_by_name(raw, probe_max)
    if index is not None:
        return index
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    if path.exists():
        return str(path.resolve())
    return str(PROJECT_ROOT / path)


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """把 CLI 参数覆盖进配置（返回新 Config，不改原对象）。"""
    if args.source is not None:
        resolved = resolve_cli_source(args.source, cfg.camera.probe_max_index)
        cfg = dataclasses.replace(cfg, camera=dataclasses.replace(cfg.camera, source=resolved))
    if args.device is not None:
        cfg = dataclasses.replace(cfg, runtime=dataclasses.replace(cfg.runtime, device=args.device))
    if args.log_level is not None:
        cfg = dataclasses.replace(cfg, runtime=dataclasses.replace(cfg.runtime, log_level=args.log_level))
    return cfg


def _describe_source(source: int | str) -> str:
    """给日志/摘要用的采集源描述。"""
    if isinstance(source, int):
        name = get_device_name(source)
        return f"{name} (idx {source})" if name else f"摄像头 (idx {source})"
    return str(source)


def resolve_capture_target(cfg: Config, args: argparse.Namespace) -> tuple[Config, str] | None:
    """按优先级解析最终采集源：--source > --camera-name > device_name > source。

    Returns:
        (新配置, 设备描述)；名字匹配失败时返回 None（调用方以非零码退出）。
    """
    if args.source is not None:
        # 已在 apply_overrides 中解析（数字/设备名/路径）
        return cfg, _describe_source(cfg.camera.source)

    name: str | None = None
    if args.camera_name is not None:
        name = args.camera_name
    elif cfg.camera.device_name:
        name = cfg.camera.device_name

    if name is not None:
        index = resolve_index_by_name(name, cfg.camera.probe_max_index)
        if index is None:
            available = device_names()
            if available:
                logger.error(
                    "按名字找不到摄像头 %r；实际可用设备：%s",
                    name,
                    ", ".join(f"{idx}:{dev}" for idx, dev in enumerate(available)),
                )
            else:
                logger.error(
                    "按名字找不到摄像头 %r；且无法枚举设备名"
                    "（安装可选依赖 pygrabber 后可按名字匹配，或改用 --list-cameras / --source）",
                    name,
                )
            return None
        logger.info("按设备名 %r 解析到摄像头索引 %d", name, index)
        cfg = dataclasses.replace(cfg, camera=dataclasses.replace(cfg.camera, source=index))
        return cfg, f"{name} (idx {index})"

    return cfg, _describe_source(cfg.camera.source)


def run_list_cameras(max_index: int) -> int:
    """--list-cameras：列出视频设备后退出（不加载模型）。无设备时以 0 退出。"""
    logger.info("开始探测视频输入设备（索引 0..%d，MSMF 后端）", max(0, max_index - 1))
    devices = list_devices(max_index)
    usable = [d for d in devices if d.opens]
    print("========== 视频输入设备 ==========")
    if not usable:
        print(f"未检测到可打开的视频输入设备（已探测索引 0..{max(0, max_index - 1)}）。")
        print("请确认设备已连接且未被其他程序占用；无摄像头时可用 --source <视频文件> 运行。")
        return 0
    print("索引  可打开  设备名                       协商分辨率  fourcc")
    for d in devices:
        name = d.name or "(未知)"
        if d.opens:
            res = f"{d.width}x{d.height}"
            fourcc = d.fourcc or "(默认)"
            mark = "是"
        else:
            res, fourcc, mark = "-", "-", "否"
        print(f"{d.index:<5} {mark:<6}  {name:<28} {res:<11} {fourcc}")
    return 0


def _fmt_negotiated(neg: dict[str, float | str]) -> str:
    """协商参数格式化为 ``1280x720@30.0 YUY2`` 样式。"""
    return (
        f"{neg.get('width', 0.0):g}x{neg.get('height', 0.0):g}"
        f"@{float(neg.get('fps', 0.0)):.1f} {neg.get('fourcc') or '(默认)'}"
    )


def _warmup_capture(cap: cv2.VideoCapture, frames: int = _CAPTURE_WARMUP_FRAMES) -> None:
    """采集测速前预热：MSMF 首帧出流有启动延迟，避免污染 60 帧计时。"""
    for _ in range(frames):
        if not cap.read()[0]:
            break


def _measure_capture_fps_standalone(cfg: Config, frames: int = 60) -> float | None:
    """主循环结束后重开实时源，单独实测采集 FPS；失败返回 None。"""
    probe = FrameSource(cfg.camera)
    try:
        probe.open()
        cap = probe.capture
        if cap is None:
            return None
        _warmup_capture(cap)
        return measure_capture_fps(cap, frames)
    except RuntimeError as exc:
        logger.warning("采集测速失败（重开设备失败）: %s", exc)
        return None
    finally:
        probe.close()


def run_bench_capture(cfg: Config, frames: int, device_label: str) -> int:
    """--bench-capture：只采集不推理，实测采集 FPS 后退出（不加载模型）。"""
    source = FrameSource(cfg.camera)
    if not source.is_live:
        print("采集测速仅适用于实时摄像头源；当前为文件源，未执行。")
        return 0
    try:
        source.open()
        cap = source.capture
        if cap is None:
            logger.error("采集测速失败：未获得底层 VideoCapture")
            return 1
        _warmup_capture(cap)
        fps = measure_capture_fps(cap, frames)
        print("========== 采集测速 ==========")
        print(f"采集源       : {device_label}")
        print(f"后端         : {source.backend_name}")
        print(f"协商参数     : {_fmt_negotiated(source.negotiated)}")
        print(
            f"实测采集 FPS : {fps:.1f}（连续 read {frames} 帧计时，"
            f"另预热 {_CAPTURE_WARMUP_FRAMES} 帧不计时）"
        )
        return 0
    except RuntimeError as exc:
        logger.error("打开采集源失败: %s", exc)
        return 1
    finally:
        source.close()


def print_summary(
    processed: int,
    wall_elapsed: float,
    sum_detect_ms: float,
    sum_track_ms: float,
    sum_total_ms: float,
    max_track_id: int,
    csv_path: str,
    capture_fps: float | None,
    device_desc: str,
    dropped_frames: int | None,
    configured_fps: float,
) -> None:
    """打印最终摘要（CLI 唯一允许的 print 输出）。

    采集 FPS（连续 read 实测）与处理 FPS（主循环端到端）分列，避免口径混淆。
    """
    avg = lambda total: total / processed if processed else 0.0  # noqa: E731
    print("========== 运行摘要 ==========")
    print(f"处理帧数     : {processed}")
    if capture_fps is None:
        print("采集 FPS     : 不适用（文件源）")
    else:
        print(f"采集 FPS     : {capture_fps:.1f}（连续 read 实测）")
    print(f"处理 FPS     : {processed / wall_elapsed:.1f}（主循环端到端，含推理/绘制/写出）")
    print(f"采集设备     : {device_desc}")
    if dropped_frames is None:
        print("丢帧数       : 不适用（文件源）")
    else:
        print(f"丢帧数       : {dropped_frames}（队列满丢旧帧）")
    print(f"平均检测耗时 : {avg(sum_detect_ms):.1f} ms")
    print(f"平均跟踪耗时 : {avg(sum_track_ms):.1f} ms")
    print(f"平均处理耗时 : {avg(sum_total_ms):.1f} ms（pipeline 端到端）")
    print(f"最大 Track ID: {max_track_id}")
    print(f"CSV 路径     : {csv_path}")
    if capture_fps is not None and capture_fps < 0.7 * configured_fps:
        print(
            f"WARNING      : 实际采集帧率 {capture_fps:.1f} 低于配置值 "
            f"{configured_fps:.1f} 的 70%，请检查后端/分辨率设置"
        )


def run(args: argparse.Namespace) -> int:
    """主流程：加载配置 -> 解析源 -> 初始化组件 -> 主循环 -> 摘要。"""
    cfg = apply_overrides(load_config(args.config), args)
    setup_logging(cfg.runtime.log_level)
    logger.info("配置加载完成（device=%s, source=%s）", cfg.runtime.device, cfg.camera.source)

    if args.list_cameras:
        return run_list_cameras(cfg.camera.probe_max_index)

    resolved = resolve_capture_target(cfg, args)
    if resolved is None:
        return 1  # 名字匹配失败：错误信息已列出可用设备
    cfg, device_label = resolved

    if args.bench_capture is not None:
        if args.bench_capture <= 0:
            logger.error("--bench-capture N 必须 > 0")
            return 1
        return run_bench_capture(cfg, args.bench_capture, device_label)

    device = resolve_device(cfg.runtime.device)
    logger.info("推理设备: %s", device)

    try:
        detector = Detector(cfg.model, cfg.detection, device)
        detector.warmup()
    except RuntimeError as exc:
        logger.error("初始化检测器失败: %s", exc)
        return 1

    tracker = Tracker(cfg.tracking)
    trajectory_mgr = TrajectoryManager(cfg.trajectory)
    pipeline = PerceptionPipeline(
        detector, tracker, trajectory_mgr, cfg.motion, cfg.performance.fps_ema_decay
    )
    visualizer = Visualizer(cfg.visualization, display_conf=cfg.detection.display_conf)

    csv_logger: CsvLogger | None = None
    if not args.no_log and cfg.logging.csv_enabled:
        csv_logger = CsvLogger(
            cfg.logging.csv_path,
            cfg.logging.csv_flush_every,
            cfg.detection.display_conf,
            append=args.append,
        )

    writer: cv2.VideoWriter | None = None
    video_path: Path | None = None

    source: FrameSource | ThreadedFrameSource = FrameSource(cfg.camera)
    was_live = source.is_live
    if was_live:
        source = ThreadedFrameSource(source)

    processed = 0
    sum_detect_ms = 0.0
    sum_track_ms = 0.0
    sum_total_ms = 0.0
    max_track_id = 0
    loop_start = time.perf_counter()
    wall_start = time.perf_counter()
    device_desc = f"{device_label} / (未打开)"

    try:
        with source:
            # 设备打开/重协商耗时可能很长（MSMF 实测约 11s），不计入处理 FPS 口径
            loop_start = time.perf_counter()
            neg = source.negotiated
            device_desc = f"{device_label} / {source.backend_name} / {_fmt_negotiated(neg)}"
            logger.info(
                "采集设备: %s | 后端: %s | 协商: %gx%g@%.1f %s",
                device_label,
                source.backend_name,
                neg.get("width", 0.0),
                neg.get("height", 0.0),
                float(neg.get("fps", 0.0)),
                neg.get("fourcc") or "(默认格式)",
            )
            frame_index = 0
            while True:
                if args.max_frames is not None and processed >= args.max_frames:
                    break
                frame_start = time.perf_counter()
                ok, frame = source.read()
                if not ok or frame is None:
                    logger.info("采集结束（共读取 %d 帧）", frame_index)
                    break

                if source.is_live:
                    timestamp = time.perf_counter() - wall_start
                else:
                    timestamp = frame_index / max(source.fps_hint, 1e-6)

                result = pipeline.process(frame, frame_index, timestamp)

                if cfg.visualization.enabled:
                    annotated = visualizer.draw(frame, result)
                else:
                    annotated = frame

                if args.save_video and writer is None:
                    out_dir = cfg.project.output_dir / "videos"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    video_path = out_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.mp4"
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        source.fps_hint,
                        (width, height),
                    )
                if writer is not None:
                    writer.write(annotated)

                if csv_logger is not None:
                    csv_logger.log(result)

                if not args.headless and cfg.visualization.enabled:
                    key = visualizer.show(annotated)
                    if key in _EXIT_KEYS:
                        logger.info("收到退出按键，结束主循环")
                        break

                # 窗口模式下按文件帧率回放（headless 全速）
                if not source.is_live and not args.headless:
                    target_dt = 1.0 / max(source.fps_hint, 1e-6)
                    elapsed = time.perf_counter() - frame_start
                    if elapsed < target_dt:
                        time.sleep(target_dt - elapsed)

                processed += 1
                frame_index += 1
                sum_detect_ms += result.detect_ms
                sum_track_ms += result.track_ms
                sum_total_ms += result.total_ms
                if result.tracks:
                    max_track_id = max(max_track_id, *(t.track_id for t in result.tracks))
    finally:
        if writer is not None:
            writer.release()
            logger.info("标注视频已写出: %s", video_path)
        if csv_logger is not None:
            csv_logger.close()
        visualizer.close()

    wall_elapsed = max(time.perf_counter() - loop_start, 1e-9)

    # 采集 FPS 与处理 FPS 分列：主循环结束后重开实时源单独测速（文件源不适用）
    capture_fps: float | None = None
    if was_live:
        capture_fps = _measure_capture_fps_standalone(cfg)
        if capture_fps is not None and capture_fps < 0.7 * cfg.camera.fps:
            logger.warning(
                "实际采集帧率 %.1f 低于配置值 %.1f 的 70%%，请检查后端/分辨率设置",
                capture_fps,
                cfg.camera.fps,
            )
    dropped_frames = source.dropped_frames if isinstance(source, ThreadedFrameSource) else None

    print_summary(
        processed,
        wall_elapsed,
        sum_detect_ms,
        sum_track_ms,
        sum_total_ms,
        max_track_id,
        str(cfg.logging.csv_path) if csv_logger is not None else "未启用",
        capture_fps,
        device_desc,
        dropped_frames,
        cfg.camera.fps,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """入口函数：统一异常兜底，保证非零清晰退出而非栈回溯乱码。"""
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n收到中断信号，已退出（资源已在 finally 中释放）")
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI 顶层兜底
        logger.error("运行失败: %s", exc)
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception("详细栈回溯（--log-level DEBUG 可见）")
        return 1


if __name__ == "__main__":
    sys.exit(main())
