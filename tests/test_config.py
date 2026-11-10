"""config.py 测试：默认值合并、类型校验、路径解析、source 判别。"""

from __future__ import annotations

import pytest

from src.config import PROJECT_ROOT, ConfigError, load_config


def write_yaml(tmp_path, content: str):
    """把 YAML 内容写入临时文件并返回路径。"""
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_defaults_when_file_missing(tmp_path):
    """配置文件不存在时使用默认值。"""
    cfg = load_config(tmp_path / "nope.yaml")
    assert cfg.model.imgsz == 640
    assert cfg.detection.conf == pytest.approx(0.10)
    assert cfg.detection.display_conf == pytest.approx(0.30)
    assert cfg.tracking.enabled is True
    assert cfg.motion.pixels_per_meter is None
    assert cfg.performance.fps_ema_decay == pytest.approx(0.9)


def test_user_values_override_defaults(tmp_path):
    """用户键覆盖默认值，未给的键保持默认。"""
    path = write_yaml(
        tmp_path,
        "detection:\n  conf: 0.2\ntrajectory:\n  max_points: 50\n",
    )
    cfg = load_config(path)
    assert cfg.detection.conf == pytest.approx(0.2)
    assert cfg.trajectory.max_points == 50
    assert cfg.detection.iou == pytest.approx(0.45)  # 未覆盖，取默认


def test_unknown_key_warns(tmp_path, caplog):
    """未知键仅告警不报错，且指明键名。"""
    path = write_yaml(
        tmp_path,
        "detection:\n  conf: 0.1\n  bogus_option: 3\nnot_a_section:\n  x: 1\n",
    )
    with caplog.at_level("WARNING"):
        cfg = load_config(path)
    joined = "\n".join(record.getMessage() for record in caplog.records)
    assert "detection.bogus_option" in joined
    assert "not_a_section" in joined
    assert cfg.detection.conf == pytest.approx(0.1)  # 正常加载


def test_paths_resolved_against_project_root(tmp_path):
    """相对路径一律解析为基于 PROJECT_ROOT 的绝对路径。"""
    cfg = load_config(tmp_path / "nope.yaml")
    assert cfg.model.weights == PROJECT_ROOT / "models" / "yolo11n.pt"
    assert cfg.model.weights.is_absolute()
    assert cfg.logging.csv_path == PROJECT_ROOT / "outputs" / "logs" / "detections.csv"
    assert cfg.project.output_dir == PROJECT_ROOT / "outputs"


def test_source_int_is_camera(tmp_path):
    """YAML 纯整数 source -> 摄像头索引。"""
    path = write_yaml(tmp_path, "camera:\n  source: 1\n")
    cfg = load_config(path)
    assert cfg.camera.source == 1
    assert isinstance(cfg.camera.source, int)


def test_source_str_is_file_path(tmp_path):
    """带引号的字符串 source -> 视频文件路径（解析为绝对路径）。"""
    path = write_yaml(tmp_path, 'camera:\n  source: "data/videos/test.mp4"\n')
    cfg = load_config(path)
    assert isinstance(cfg.camera.source, str)
    assert cfg.camera.source == str(PROJECT_ROOT / "data" / "videos" / "test.mp4")


def test_source_url_kept_as_is(tmp_path):
    """URL 形式的 source 原样保留。"""
    path = write_yaml(tmp_path, 'camera:\n  source: "rtsp://192.168.1.1/stream"\n')
    cfg = load_config(path)
    assert cfg.camera.source == "rtsp://192.168.1.1/stream"


def test_source_type_error_raises(tmp_path):
    """source 类型非法（列表）时抛 ConfigError。"""
    path = write_yaml(tmp_path, "camera:\n  source: [0, 1]\n")
    with pytest.raises(ConfigError):
        load_config(path)


def test_conf_out_of_range_raises(tmp_path):
    """conf 超出 [0,1] 时抛 ConfigError。"""
    path = write_yaml(tmp_path, "detection:\n  conf: 1.5\n")
    with pytest.raises(ConfigError):
        load_config(path)


def test_top_level_not_mapping_raises(tmp_path):
    """顶层不是映射时抛 ConfigError。"""
    path = tmp_path / "config.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_camera_new_keys_defaults(tmp_path):
    """新增 camera 键（device_name/backend/fourcc/probe_max_index）的默认值合并。"""
    cfg = load_config(tmp_path / "nope.yaml")
    assert cfg.camera.device_name == ""
    assert cfg.camera.backend == "auto"
    assert cfg.camera.fourcc == ""
    assert cfg.camera.probe_max_index == 4


def test_camera_backend_invalid_raises(tmp_path):
    """backend 非法值抛 ConfigError 并指明键名。"""
    path = write_yaml(tmp_path, "camera:\n  backend: vfw\n")
    with pytest.raises(ConfigError, match="camera.backend"):
        load_config(path)


def test_camera_fourcc_length_validation(tmp_path):
    """fourcc 为空或恰好 4 字符合法（归一化为大写）；其他长度抛 ConfigError。"""
    assert load_config(write_yaml(tmp_path, 'camera:\n  fourcc: "MJPG"\n')).camera.fourcc == "MJPG"
    assert load_config(write_yaml(tmp_path, 'camera:\n  fourcc: "mjpg"\n')).camera.fourcc == "MJPG"
    assert load_config(write_yaml(tmp_path, 'camera:\n  fourcc: ""\n')).camera.fourcc == ""
    with pytest.raises(ConfigError, match="camera.fourcc"):
        load_config(write_yaml(tmp_path, 'camera:\n  fourcc: "MJP"\n'))
    with pytest.raises(ConfigError, match="camera.fourcc"):
        load_config(write_yaml(tmp_path, 'camera:\n  fourcc: "MJPGX"\n'))
