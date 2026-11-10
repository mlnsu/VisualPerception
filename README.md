# Robot Vision System

**基于 YOLO 的实时目标检测与多目标跟踪系统**

## 1. 项目简介

Robot Vision System 是一个面向移动机器人的实时视觉感知系统，基于 YOLO 目标检测算法与 ByteTrack 多目标跟踪算法，实现对动态环境中多个目标的实时识别、定位、跟踪与运动信息提取。

项目以 Python 为主要开发语言，首先在 Windows PC 环境下完成算法验证与系统开发，后续计划引入深度相机，实现目标三维空间定位，并探索在 RDK 机器人计算平台上的部署。

本项目希望构建一套模块化、可扩展的机器人视觉感知系统，为后续机器人环境感知、动态目标分析及自主导航提供基础支持。

## 2. 项目目标

### 当前开发目标

- [x] 确定 YOLO + ByteTrack 技术路线
- [x] 实现摄像头实时图像采集
- [x] 实现 YOLO 实时目标检测
- [x] 实现多目标跟踪与 ID 分配
- [x] 实现目标运动轨迹可视化
- [x] 实现目标运动方向与速度估计
- [x] 实现检测数据记录与性能统计

### 后续开发目标

- [ ] 引入深度相机，实现目标距离估计
- [ ] 实现相机坐标系下的三维目标定位
- [ ] 建立机器人坐标系转换
- [ ] 探索 ROS 2 通信接口
- [ ] 部署至 RDK 机器人计算平台
- [ ] 为机器人导航与动态避障提供感知数据

## 3. 系统架构

```text
             Camera / Video Stream
                      |
                      v
              Image Acquisition
                      |
                      v
               YOLO Detection
                      |
                      v
              ByteTrack Tracking
                      |
                      v
              Target Information
                      |
           +----------+----------+
           |          |          |
           v          v          v
        Object     Trajectory   Motion
       Detection    Tracking   Analysis
           |          |          |
           +----------+----------+
                      |
                      v
               Visualization
                      |
                      v
              Perception Output
                      |
                      v
           Future: 3D Spatial Perception
                      |
                      v
                 ROS 2 / RDK
```

## 4. 技术栈

| 技术          | 用途       |
| ----------- | -------- |
| Python 3.11 | 主要开发语言   |
| PyTorch     | 深度学习推理框架 |
| YOLO11n     | 实时目标检测   |
| ByteTrack   | 多目标跟踪    |
| OpenCV      | 图像采集与处理  |
| NumPy       | 数值计算     |
| Pandas      | 数据记录与分析  |
| Git         | 版本管理     |

### 核心算法

**YOLO11n**

负责从输入图像中检测目标，输出目标类别、置信度及边界框坐标。

**ByteTrack**

负责关联连续视频帧中的检测结果，为目标分配 Track ID，实现多目标连续跟踪。

**OpenCV**

负责摄像头采集、图像处理、检测结果绘制及实时窗口显示。

## 5. 项目结构

```text
VisualPerception/
│
├── configs/
│   └── config.yaml          # 全部阈值/路径/设备配置（无硬编码）
├── models/                  # YOLO 权重目录（.pt 不入库）
│
├── src/
│   ├── config.py            # 配置加载 + 默认值合并 + 路径解析
│   ├── datatypes.py         # Detection / Track / TrackHistory / FrameResult
│   ├── camera.py            # FrameSource（含实时线程化采集）
│   ├── detector.py          # YOLO11n 检测封装（只出框）
│   ├── tracker.py           # 独立 sv.ByteTrack 跟踪封装
│   ├── trajectory.py        # 轨迹历史（deque + TTL 淘汰）
│   ├── motion.py            # 方向/速度估计（最小二乘，px/s）
│   ├── visualizer.py        # 框/轨迹/标签/HUD 绘制
│   ├── csv_logger.py        # CSV 记录（Pandas 可直读）
│   ├── pipeline.py          # PerceptionPipeline 逐帧编排
│   └── main.py              # CLI 入口
│
├── data/
│   ├── images/
│   └── videos/
│
├── outputs/
│   ├── videos/
│   └── logs/                # detections.csv / baseline.txt / perf.txt
│
├── tests/                   # 6 个测试文件，不依赖网络/摄像头/权重
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 6. 环境配置

### 运行环境

- OS：Windows 10 / Windows 11
- Python：3.11+（V1 在 3.13 上开发验证）
- RAM：建议 16GB
- GPU：支持 CPU 推理；如有 NVIDIA GPU，可配置 CUDA 加速
- Camera：USB 摄像头或其他视频输入设备（无摄像头时可用 `--source <视频文件>` 替代）

### 创建虚拟环境

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv\Scripts\activate
```

### 安装依赖

```bash
python -m pip install -r requirements.txt
```

依赖已带版本上限固定。注意 `supervision` 上限为 `<0.26`（更新版本移除了 `sv.ByteTrack`）。

## 7. 快速开始

### 7.1 下载项目

```bash
git clone https://github.com/你的用户名/VisualPerception.git

cd VisualPerception
```

### 7.2 下载模型

首次运行时，Ultralytics 可以自动下载 YOLO11n 预训练权重。

也可以提前下载模型，并放置在 `models/` 目录。

### 7.3 运行目标检测

```bash
python src/main.py
```

默认使用编号为 0 的摄像头。

按 `Q` 键退出实时检测窗口。

### 7.4 视频文件输入与无窗口模式

```bash
# 视频文件 + headless（无窗口，服务器/自动化自检）
python src/main.py --source data/videos/test.mp4 --headless --max-frames 300

# 其他常用参数
#   --save-video     把标注画面写为 outputs/videos/run_*.mp4
#   --no-log         禁用 CSV 记录
#   --append         CSV 追加模式（默认覆盖）
#   --device cpu     覆盖推理设备
#   --log-level DEBUG
```

也可以在配置文件中切换视频输入：

```yaml
camera:
  source: "data/videos/test.mp4"
```

运行结束后输出摘要（处理帧数 / 平均 FPS / 平均检测与跟踪耗时 / 最大 Track ID / CSV 路径），
逐帧明细写入 `outputs/logs/detections.csv`（Pandas 可直读）。

## 8. 核心功能

| 功能     | 说明                        | 状态   |
| ------ | ------------------------- | ---- |
| 实时图像采集 | 摄像头/视频统一读帧，线程化低延迟采集    | ✅ 已实现 |
| 目标检测   | YOLO11n 识别目标（阈值双轨：跟踪 0.10 / 显示 0.30） | ✅ 已实现 |
| 多目标跟踪  | 独立 sv.ByteTrack 分配稳定 ID | ✅ 已实现 |
| 轨迹绘制   | 定长 deque 轨迹 + TTL 回收，颜色按 ID 恒定 | ✅ 已实现 |
| 速度估计   | 最小二乘拟合，图像平面 px/s（无标定不输出 m/s） | ✅ 已实现 |
| 数据记录   | CSV 逐帧记录（Pandas 直读）+ FPS/耗时 HUD | ✅ 已实现 |
| 深度感知   | 获取目标深度信息                  | 后续开发 |
| 三维定位   | 转换至相机坐标系                  | 后续开发 |
| RDK 部署 | 边缘端模型推理                   | 后续开发 |

## 9. 性能评估

项目计划从检测精度、跟踪效果和运行性能三个方面进行评估。

### 目标检测指标

- Precision
- Recall
- mAP50
- mAP50-95

### 多目标跟踪指标

- MOTA
- IDF1
- HOTA
- ID Switches

### 系统性能指标

- FPS：每秒处理帧数
- Latency：单帧处理延迟
- CPU Usage：CPU 使用率
- Memory Usage：内存占用
- Stability：长时间运行稳定性

## 10. 技术路线规划

| 版本   | 开发内容            | 目标     |
| ---- | --------------- | ------ |
| V1.0 | YOLO 实时目标检测     | 完成基础检测 |
| V1.1 | ByteTrack 多目标跟踪 | 实现连续跟踪 |
| V1.2 | 轨迹分析与速度估计       | 提取运动信息 |
| V1.3 | 模块化重构与性能优化      | 提高工程质量 |
| V2.0 | RGB-D 深度感知      | 实现三维定位 |
| V2.1 | 相机与机器人坐标变换      | 建立空间感知 |
| V3.0 | ROS 2 / RDK 部署  | 机器人端运行 |

## 11. 后续研究方向

项目后续将重点探索以下方向：

1. **视觉感知：** 提升复杂环境下的目标检测和跟踪能力。
2. **空间感知：** 融合 RGB 图像与深度信息，实现三维目标定位。
3. **多传感器融合：** 探索视觉、深度相机及其他传感器的数据融合。
4. **机器人系统集成：** 研究 ROS 2 通信机制及机器人感知接口。
5. **边缘计算部署：** 探索模型轻量化、推理加速及 RDK 平台部署。

## 12. 参考资料

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [Ultralytics Tracking Documentation](https://docs.ultralytics.com/modes/track/)
- [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)
- [OpenCV Documentation](https://docs.opencv.org/4.x/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## 13. 项目说明

本项目为个人学习与机器人视觉技术探索项目，目前处于开发阶段。

项目重点关注计算机视觉、目标检测、多目标跟踪及机器人空间感知技术，后续将持续完善算法实现、系统架构与边缘端部署。

**Project:** Robot Vision System
**Development Language:** Python
**Core Algorithms:** YOLO + ByteTrack
**Application:** Robotics Visual Perception

---

*Building a visual perception system for intelligent robots.*
