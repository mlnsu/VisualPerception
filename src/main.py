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

from src.camera import FrameSource, ThreadedFrameSource
from src.config import PROJECT_ROOT, Config, load_config
from src.csv_logger import CsvLogger
from src.detector import Detector
from src.pipeline import PerceptionPipeline
from src.tracker import Tracker
from src.trajectory import TrajectoryManager
from src.visualizer import Visualizer

logger = logging.getLogger("src.main")

#: 退出按键：q / Q / ESC
_EXIT_KEYS = frozenset((ord("q"), ord("Q"), 27))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 CLI 参数。"""
    parser = argparse.ArgumentParser(
        prog="src.main", description="VisualPerception V1 实时感知 CLI"
    )
    parser.add_argument("--config", type=str, default=None, help="配置文件路径（默认 configs/config.yaml）")
    parser.add_argument("--source", type=str, default=None, help="覆盖采集源：摄像头索引（如 0）或视频路径")
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


def resolve_cli_source(raw: str) -> int | str:
    """解析 --source：纯数字为摄像头索引，否则为路径（相对 CWD 优先，其次项目根）。"""
    if raw.isdigit():
        return int(raw)
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    if path.exists():
        return str(path.resolve())
    return str(PROJECT_ROOT / path)


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """把 CLI 参数覆盖进配置（返回新 Config，不改原对象）。"""
    if args.source is not None:
        cfg = dataclasses.replace(cfg, camera=dataclasses.replace(cfg.camera, source=resolve_cli_source(args.source)))
    if args.device is not None:
        cfg = dataclasses.replace(cfg, runtime=dataclasses.replace(cfg.runtime, device=args.device))
    if args.log_level is not None:
        cfg = dataclasses.replace(cfg, runtime=dataclasses.replace(cfg.runtime, log_level=args.log_level))
    return cfg


def print_summary(
    processed: int,
    wall_elapsed: float,
    sum_detect_ms: float,
    sum_track_ms: float,
    sum_total_ms: float,
    max_track_id: int,
    csv_path: str,
) -> None:
    """打印最终摘要（CLI 唯一允许的 print 输出）。"""
    avg = lambda total: total / processed if processed else 0.0  # noqa: E731
    print("========== 运行摘要 ==========")
    print(f"处理帧数     : {processed}")
    print(f"平均 FPS     : {processed / wall_elapsed:.1f}（口径：主循环端到端，含读帧/推理/绘制/写出）")
    print(f"平均检测耗时 : {avg(sum_detect_ms):.1f} ms")
    print(f"平均跟踪耗时 : {avg(sum_track_ms):.1f} ms")
    print(f"平均处理耗时 : {avg(sum_total_ms):.1f} ms（pipeline 端到端）")
    print(f"最大 Track ID: {max_track_id}")
    print(f"CSV 路径     : {csv_path}")


def run(args: argparse.Namespace) -> int:
    """主流程：加载配置 -> 初始化组件 -> 主循环 -> 摘要。"""
    cfg = apply_overrides(load_config(args.config), args)
    setup_logging(cfg.runtime.log_level)
    logger.info("配置加载完成（device=%s, source=%s）", cfg.runtime.device, cfg.camera.source)

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
    if source.is_live:
        source = ThreadedFrameSource(source)

    processed = 0
    sum_detect_ms = 0.0
    sum_track_ms = 0.0
    sum_total_ms = 0.0
    max_track_id = 0
    loop_start = time.perf_counter()
    wall_start = time.perf_counter()

    try:
        with source:
            # 设备打开/重协商耗时可能很长（MSMF 实测约 11s），不计入处理 FPS 口径
            loop_start = time.perf_counter()
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
    print_summary(
        processed,
        wall_elapsed,
        sum_detect_ms,
        sum_track_ms,
        sum_total_ms,
        max_track_id,
        str(cfg.logging.csv_path) if csv_logger is not None else "未启用",
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
