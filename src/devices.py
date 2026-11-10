"""视频输入设备枚举与按名字解析。

设备名获取优先使用 pygrabber（可选增强依赖，不写入 requirements.txt）：
DirectShow 枚举顺序与 cv2 摄像头索引一致；import 失败时退化为只报索引与可打开性。
不使用 ffmpeg 子进程解析（不可移植且解析脆弱）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2

from src.camera import decode_fourcc

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """单个视频输入设备的探测结果。"""

    index: int
    name: str  # 拿不到设备名时为空串
    opens: bool
    width: int  # 协商分辨率；未打开时为 0
    height: int
    fourcc: str  # 协商格式；未打开时为空串


def device_names() -> list[str]:
    """DirectShow 视频输入设备名列表（索引序）；pygrabber 不可用时返回空列表。"""
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        logger.debug("pygrabber 不可用，无法枚举设备名（可选依赖，pip install pygrabber）")
        return []
    try:
        return [str(name) for name in FilterGraph().get_input_devices()]
    except Exception as exc:  # noqa: BLE001 - COM 枚举失败按"无名"处理
        logger.warning("枚举 DirectShow 设备名失败: %s", exc)
        return []


def get_device_name(index: int) -> str:
    """按索引查设备名（只查名字列表，不打开设备）；未知返回空串。"""
    names = device_names()
    if 0 <= index < len(names):
        return names[index]
    return ""


def list_devices(max_index: int) -> list[DeviceInfo]:
    """探测索引 0..max_index-1 的视频设备。

    每个索引只用 MSMF 快速判定：打开 -> 读一帧 -> 取协商参数 -> 立即 release，
    整体耗时尽力控制在数秒内；单设备失败不影响其余探测。
    """
    names = device_names()
    results: list[DeviceInfo] = []
    for idx in range(max(0, max_index)):
        name = names[idx] if idx < len(names) else ""
        info = DeviceInfo(index=idx, name=name, opens=False, width=0, height=0, fourcc="")
        cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
        try:
            if cap.isOpened():
                ok, _ = cap.read()  # 读一帧确认真正出流
                if ok:
                    info.opens = True
                    info.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    info.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    info.fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
        except Exception as exc:  # noqa: BLE001 - 单个设备异常不终止枚举
            logger.debug("探测索引 %d 异常: %s", idx, exc)
        finally:
            cap.release()
        results.append(info)
    return results


def resolve_index_by_name(name: str, max_index: int) -> int | None:
    """按设备名解析索引（大小写不敏感的包含匹配，设备名可能带前后缀）。

    多个命中时取最小索引并记录 WARNING 列出所有候选；无命中返回 None。
    """
    needle = name.strip().lower()
    if not needle:
        return None
    names = device_names()
    hits = [(idx, dev) for idx, dev in enumerate(names) if needle in dev.lower()]
    if not hits:
        return None
    index = hits[0][0]
    if len(hits) > 1:
        logger.warning(
            "设备名 %r 命中多个候选: %s；取索引最小的 %d",
            name,
            ", ".join(f"{idx}:{dev}" for idx, dev in hits),
            index,
        )
    if index >= max_index:
        logger.warning("命中索引 %d 超出探测上限 %d（camera.probe_max_index）", index, max_index)
        return None
    return index
