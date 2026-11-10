"""配置加载、默认值合并与路径解析。

所有相对路径一律基于 PROJECT_ROOT 解析，禁止依赖当前工作目录。
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

#: 项目根目录（本文件位于 <root>/src/，向上两级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: 默认配置文件路径
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class ConfigError(ValueError):
    """配置文件结构或类型错误。"""


@dataclass(frozen=True)
class ProjectConfig:
    """项目级输出设置。"""

    output_dir: Path


@dataclass(frozen=True)
class RuntimeConfig:
    """运行时设置。"""

    device: str
    log_level: str


@dataclass(frozen=True)
class CameraConfig:
    """采集源设置。

    source 为 int 时表示摄像头索引；为 str 时表示视频文件路径或 URL。
    """

    source: int | str
    backend: str
    width: int
    height: int
    fps: float
    buffer_size: int
    loop_video: bool
    reconnect: bool
    reconnect_interval_s: float


@dataclass(frozen=True)
class ModelConfig:
    """模型设置。"""

    weights: Path
    imgsz: int
    half: bool
    verbose: bool


@dataclass(frozen=True)
class DetectionConfig:
    """检测设置。

    conf 喂给跟踪器（低阈值，ByteTrack 二段关联依赖低分框）；
    display_conf 仅用于绘制与 CSV 过滤。
    """

    conf: float
    iou: float
    max_det: int
    classes: list[int]
    display_conf: float


@dataclass(frozen=True)
class TrackingConfig:
    """跟踪设置。"""

    enabled: bool
    track_activation_threshold: float
    lost_track_buffer: int
    minimum_matching_threshold: float
    minimum_consecutive_frames: int


@dataclass(frozen=True)
class TrajectoryConfig:
    """轨迹设置。"""

    max_points: int
    ttl_frames: int


@dataclass(frozen=True)
class MotionConfig:
    """运动估计设置。pixels_per_meter 为 None 时不输出 m/s。"""

    window: int
    min_points: int
    pixels_per_meter: float | None


@dataclass(frozen=True)
class VisualizationConfig:
    """可视化设置。"""

    enabled: bool
    window_name: str
    draw_trajectory: bool
    draw_speed: bool
    draw_hud: bool
    line_width: int
    font_scale: float


@dataclass(frozen=True)
class LoggingConfig:
    """CSV 记录设置。"""

    csv_enabled: bool
    csv_path: Path
    csv_flush_every: int


@dataclass(frozen=True)
class PerformanceConfig:
    """性能统计设置。"""

    fps_ema_decay: float


@dataclass(frozen=True)
class Config:
    """完整配置（与 config.yaml 结构一一对应）。"""

    project: ProjectConfig
    runtime: RuntimeConfig
    camera: CameraConfig
    model: ModelConfig
    detection: DetectionConfig
    tracking: TrackingConfig
    trajectory: TrajectoryConfig
    motion: MotionConfig
    visualization: VisualizationConfig
    logging: LoggingConfig
    performance: PerformanceConfig


#: 默认配置（与 configs/config.yaml 保持一致；缺键时以此合并）
DEFAULTS: dict[str, Any] = {
    "project": {"output_dir": "outputs"},
    "runtime": {"device": "auto", "log_level": "INFO"},
    "camera": {
        "source": 0,
        "backend": "dshow",
        "width": 1280,
        "height": 720,
        "fps": 30,
        "buffer_size": 1,
        "loop_video": False,
        "reconnect": True,
        "reconnect_interval_s": 2.0,
    },
    "model": {
        "weights": "models/yolo11n.pt",
        "imgsz": 640,
        "half": False,
        "verbose": False,
    },
    "detection": {
        "conf": 0.10,
        "iou": 0.45,
        "max_det": 300,
        "classes": [0],
        "display_conf": 0.30,
    },
    "tracking": {
        "enabled": True,
        "track_activation_threshold": 0.25,
        "lost_track_buffer": 30,
        "minimum_matching_threshold": 0.8,
        "minimum_consecutive_frames": 1,
    },
    "trajectory": {"max_points": 200, "ttl_frames": 60},
    "motion": {"window": 15, "min_points": 5, "pixels_per_meter": None},
    "visualization": {
        "enabled": True,
        "window_name": "VisualPerception",
        "draw_trajectory": True,
        "draw_speed": True,
        "draw_hud": True,
        "line_width": 2,
        "font_scale": 0.5,
    },
    "logging": {
        "csv_enabled": True,
        "csv_path": "outputs/logs/detections.csv",
        "csv_flush_every": 100,
    },
    "performance": {"fps_ema_decay": 0.9},
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并字典，override 优先。"""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _find_unknown_keys(
    reference: dict[str, Any], user: dict[str, Any], prefix: str = ""
) -> list[str]:
    """收集用户配置中未被默认结构覆盖的键（点分路径表示）。"""
    unknown: list[str] = []
    for key, value in user.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in reference:
            unknown.append(path)
        elif isinstance(reference[key], dict) and isinstance(value, dict):
            unknown.extend(_find_unknown_keys(reference[key], value, path))
    return unknown


def _resolve_path(value: str | Path) -> Path:
    """相对路径基于 PROJECT_ROOT 解析为绝对路径。"""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _require_number(raw: Any, key: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ConfigError(f"{key} 必须是数值，实际为 {type(raw).__name__}")
    return float(raw)


def _require_int(raw: Any, key: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ConfigError(f"{key} 必须是整数，实际为 {type(raw).__name__}")
    return int(raw)


def _build_camera(raw: dict[str, Any]) -> CameraConfig:
    """构建并校验 CameraConfig（重点：source 的 int/str 判别）。"""
    source_raw = raw["source"]
    if isinstance(source_raw, bool) or not isinstance(source_raw, (int, str)):
        raise ConfigError(
            f"camera.source 必须是 int（摄像头索引）或 str（视频路径），"
            f"实际为 {type(source_raw).__name__}"
        )
    source: int | str
    if isinstance(source_raw, int):
        source = source_raw
    elif "://" in source_raw:
        source = source_raw  # URL 直传
    else:
        source = str(_resolve_path(source_raw))

    return CameraConfig(
        source=source,
        backend=str(raw["backend"]),
        width=_require_int(raw["width"], "camera.width"),
        height=_require_int(raw["height"], "camera.height"),
        fps=_require_number(raw["fps"], "camera.fps"),
        buffer_size=_require_int(raw["buffer_size"], "camera.buffer_size"),
        loop_video=bool(raw["loop_video"]),
        reconnect=bool(raw["reconnect"]),
        reconnect_interval_s=_require_number(
            raw["reconnect_interval_s"], "camera.reconnect_interval_s"
        ),
    )


def _build(raw: dict[str, Any]) -> Config:
    """从合并后的原始字典构建强类型 Config，含轻量校验。"""
    detection = raw["detection"]
    for key in ("conf", "iou", "display_conf"):
        value = _require_number(detection[key], f"detection.{key}")
        if not 0.0 <= value <= 1.0:
            raise ConfigError(f"detection.{key} 必须在 [0,1] 区间，实际为 {value}")
    classes_raw = detection["classes"]
    if not isinstance(classes_raw, list) or any(
        isinstance(c, bool) or not isinstance(c, int) for c in classes_raw
    ):
        raise ConfigError("detection.classes 必须是整数列表")

    trajectory = raw["trajectory"]
    if _require_int(trajectory["max_points"], "trajectory.max_points") < 1:
        raise ConfigError("trajectory.max_points 必须 >= 1")
    if _require_int(trajectory["ttl_frames"], "trajectory.ttl_frames") < 0:
        raise ConfigError("trajectory.ttl_frames 必须 >= 0")

    motion = raw["motion"]
    ppm_raw = motion["pixels_per_meter"]
    if ppm_raw is not None:
        ppm = _require_number(ppm_raw, "motion.pixels_per_meter")
        if ppm <= 0:
            raise ConfigError("motion.pixels_per_meter 必须 > 0")
    else:
        ppm = None

    decay = _require_number(raw["performance"]["fps_ema_decay"], "performance.fps_ema_decay")
    if not 0.0 <= decay <= 1.0:
        raise ConfigError("performance.fps_ema_decay 必须在 [0,1] 区间")

    return Config(
        project=ProjectConfig(output_dir=_resolve_path(raw["project"]["output_dir"])),
        runtime=RuntimeConfig(
            device=str(raw["runtime"]["device"]), log_level=str(raw["runtime"]["log_level"])
        ),
        camera=_build_camera(raw["camera"]),
        model=ModelConfig(
            weights=_resolve_path(raw["model"]["weights"]),
            imgsz=_require_int(raw["model"]["imgsz"], "model.imgsz"),
            half=bool(raw["model"]["half"]),
            verbose=bool(raw["model"]["verbose"]),
        ),
        detection=DetectionConfig(
            conf=float(detection["conf"]),
            iou=float(detection["iou"]),
            max_det=_require_int(detection["max_det"], "detection.max_det"),
            classes=[int(c) for c in classes_raw],
            display_conf=float(detection["display_conf"]),
        ),
        tracking=TrackingConfig(
            enabled=bool(raw["tracking"]["enabled"]),
            track_activation_threshold=float(raw["tracking"]["track_activation_threshold"]),
            lost_track_buffer=_require_int(raw["tracking"]["lost_track_buffer"], "tracking.lost_track_buffer"),
            minimum_matching_threshold=float(raw["tracking"]["minimum_matching_threshold"]),
            minimum_consecutive_frames=_require_int(
                raw["tracking"]["minimum_consecutive_frames"],
                "tracking.minimum_consecutive_frames",
            ),
        ),
        trajectory=TrajectoryConfig(
            max_points=int(trajectory["max_points"]),
            ttl_frames=int(trajectory["ttl_frames"]),
        ),
        motion=MotionConfig(
            window=_require_int(motion["window"], "motion.window"),
            min_points=_require_int(motion["min_points"], "motion.min_points"),
            pixels_per_meter=ppm,
        ),
        visualization=VisualizationConfig(
            enabled=bool(raw["visualization"]["enabled"]),
            window_name=str(raw["visualization"]["window_name"]),
            draw_trajectory=bool(raw["visualization"]["draw_trajectory"]),
            draw_speed=bool(raw["visualization"]["draw_speed"]),
            draw_hud=bool(raw["visualization"]["draw_hud"]),
            line_width=_require_int(raw["visualization"]["line_width"], "visualization.line_width"),
            font_scale=_require_number(raw["visualization"]["font_scale"], "visualization.font_scale"),
        ),
        logging=LoggingConfig(
            csv_enabled=bool(raw["logging"]["csv_enabled"]),
            csv_path=_resolve_path(raw["logging"]["csv_path"]),
            csv_flush_every=_require_int(raw["logging"]["csv_flush_every"], "logging.csv_flush_every"),
        ),
        performance=PerformanceConfig(fps_ema_decay=decay),
    )


def load_config(path: str | Path | None = None) -> Config:
    """加载配置文件并与默认值合并。

    Args:
        path: 配置文件路径；None 时使用 ``configs/config.yaml``，
            文件不存在时直接使用默认值（记警告）。

    Returns:
        Config: 强类型嵌套配置对象。

    Raises:
        ConfigError: 配置结构或类型非法时抛出。
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    user: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise ConfigError(f"配置文件顶层必须是映射，实际为 {type(loaded).__name__}: {config_path}")
            user = loaded
    else:
        logger.warning("配置文件不存在，使用默认配置: %s", config_path)

    unknown = _find_unknown_keys(DEFAULTS, user)
    if unknown:
        logger.warning("配置中存在未知键（已忽略）: %s", ", ".join(unknown))

    merged = _deep_merge(copy.deepcopy(DEFAULTS), user)
    return _build(merged)
