# LaserMatch - 高精度激光线双目匹配与三维重建系统

![Language](https://img.shields.io/badge/Language-C++17-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Linux-orange.svg)
![Framework](https://img.shields.io/badge/Framework-OpenCV%20|%20CUDA-green.svg)

## 1. 项目简介

`LaserCalib` 是一个专为结构光系统设计的高精度激光线提取、匹配与三维重建平台。系统基于双目极线校正技术，结合光曲面模型（Quad Surface）和创新的“切片匹配（Slice Matching）”算法，实现了在复杂环境下对激光条纹的鲁棒识别与亚像素级匹配。

### 核心功能：
- **双目极线校正**：基于 OpenCV 实现高速图像校正，统一左右图极线。
- **激光线提取**：两阶段提取方案（Two-Pass 连通域 + 质心法亚像素提取）。
- **光曲面建模**：支持多个二次曲面（Quad Surface）光场建模，有效补偿镜头畸变与非平面光场。
- **切片匹配算法**：独创的局部贪心切片匹配（Match5），针对激光线的不连续性和噪声具有极高稳定性。
- **高性能计算**：深度集成 OpenCV CUDA 模块与 Intel TBB，实现多路并行加速。

---

## 2. 目录结构

```bash
.
├── src/                # C++ 源代码 (.cpp)
│   ├── main.cpp        # 程序入口，处理命令行参数与主流程
│   ├── laser.cpp       # 激光中心提取与曲面求交逻辑
│   └── match.cpp       # 核心匹配算法实现
├── include/            # 头文件 (.h)
│   ├── configer.h      # 配置管理与全局路径
│   ├── laser.h         # 激光处理类定义
│   ├── match.h         # 匹配处理器定义
│   └── type.h          # 数据结构定义 (LaserLine, QuadSurface 等)
├── python_view/        # Python 分析与可视化工具
│   ├── pts3_fliter_3method.py  # 3D 点云多级滤波 (双边/中值/SG滤波)
│   └── edge_check.py           # 激光边缘质量检查
├── CMakeLists.txt      # CMake 编译配置文件
└── README.md           # 项目文档
```

---

## 3. 环境依赖

| 依赖项 | 推荐版本 | 说明 |
| :--- | :--- | :--- |
| **OpenCV** | 4.x + | 需包含 `contrib` 模块且开启 `CUDA` 支持 |
| **CUDA Toolkit** | 11.x + | 用于图像预处理与计算加速 |
| **Intel TBB** | Latest | 用于多线程并行提取 |
| **fmt** | Latest | 高性能日志格式化 |
| **CMake** | 3.26 + | 构建工具 |

---

## 4. 编译指南

在 Linux 环境下，使用以下命令进行编译：

```bash
cmake --build build -j 8
```

---

## 5. 使用说明

### 5.1 数据准备
运行前需准备测试文件夹（例如 `laser_0926`），目录内应包含：
- `left/` 和 `right/`：存放左右相机同步拍摄的激光图像。
- `stereo_calib.yml`：双目相机标定参数。
- `quad_surface.yml`：光曲面系数文件。

### 5.2 运行命令
```bash
# 基础运行模式
./LaserCalib -f <数据目录>

# 开启 Debug 模式（生成可视化图与匹配详情）
./LaserCalib -f <数据目录> -d

# 检查特定激光线 ID 的重投影一致性（需指定左线 ID）
./LaserCalib -f <数据目录> -d -i 3
```

### 5.3 命令行参数
- `-f <path>`: 指定测试目录路径（必填）。
- `-d`: 开启调试模式，结果保存至 `debug_img/`。
- `-i <id>`: 对指定的左激光线绘制所有光曲面重投影，用于验证标定精度。

---

## 6. 算法详解（手下交接参考）

### 6.1 切片匹配 (Slice Matching)
核心函数 `MatchProcessor::match`。不同于传统的极线约束点对点匹配，本项目将激光线拆分为多个 Y 轴切片。
1. **重投影约束**：利用光曲面方程将左图点投影至三维空间，再反投影至右图极线。
2. **多特征评分**：综合 **Census 变换**、**距离众数 (Mode distance)**、**标准差 (StdDev)** 等指标。
3. **局部优化**：利用局部平滑性约束（Dynamic Programming 思想）过滤掉跳变的匹配结果。

### 6.2 后处理滤波 (`python_view`)
激光线连通区域是通过边缘定位的，边缘可能因为噪声而定位不准确，该python代码用于测试和查看滤波对边缘平滑的效果。
- **自适应中值滤波**：去除脉冲噪声。
- **一维双边滤波**：在保持深度跳变边缘的同时平滑激光面。
- **SG 滤波 (Savitzky-Golay)**：高阶平滑，提升点云曲率表现。