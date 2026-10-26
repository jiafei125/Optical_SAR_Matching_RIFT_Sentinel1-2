# 光学-SAR 遥感影像配准预处理流程

基于 Python 的光学-SAR 异源遥感影像配准与平台定位项目，实现了从原始影像到配准完成的完整处理链路。核心流程包括光学影像六步预处理、SAR 影像对齐与辐射归一化、基于相位一致性的异源特征配准（RIFT），以及基于斜距交会的 SAR 平台定位解算。

## 项目结构

```
Synthesize/
├── CompleteProcess.py           # 光学预处理完整流程入口（M1→M6）
├── common.py                    # 通用工具（TIF 读写、坐标转换、质量评价等）
│
├── m1_geometric.py              # M1: 几何校正（去旋转、可选正射校正）
├── m2_radiometric.py            # M2: 辐射校正（DN → 表观反射率）
├── m3_mask.py                   # M3: 掩膜 + 云检测（OTSU 自适应 + 形态学）
├── m4_filter.py                 # M4: 滤波（双边/高斯/中值/非局部均值）
├── m5_resample.py               # M5: 工作网格重采样（双线性插值 + 掩膜最近邻）
├── m6_normalize.py              # M6: 灰度归一化（百分位拉伸 / CLAHE）
│
├── Pre_SAR_Align_Optical.py     # SAR 预处理: 投影对齐到光学基准网格 + 共同掩膜生成
├── Pre_SAR_HistogramMatch.py    # SAR 预处理: 直方图匹配 / 对数拉伸
│
├── Registration.py              # 光学-SAR 异源配准（RIFT: 相位一致性 + MIM 描述子）
├── Position.py                  # SAR 平台定位（斜距交会 + 两步法稳健估计）
│
├── data/                        # 原始输入数据
│   ├── optical_v3.tif           # 光学影像（DN 值）
│   ├── sar_vv_v2.tif            # SAR 影像（VV 极化）
│   ├── dem_v2.tif               # DEM 高程数据
│   ├── sar_mask_v2.tif          # SAR 有效区域掩膜
│   └── slant_range_v2.tif       # SAR 斜距波段
│
└── output/                      # 各模块中间结果与最终输出
```

## 处理流程

### 一、光学影像预处理（M1 → M6）

通过 `CompleteProcess.py` 一键执行，也可单独运行各模块。

| 模块 | 功能 | 输入 | 输出 |
|------|------|------|------|
| M1 | 几何校正：去旋转（可选正射校正），生成工作网格 | `data/optical_v3.tif` | `m1_optical_geo.tif`, `m1_valid_mask.npy`, `work_grid.json` |
| M2 | 辐射校正：DN → 标准表观反射率（支持 L2A/L1C/Landsat 等多种产品） | M1 输出 | `m2_optical_refl.tif` |
| M3 | 掩膜 + 云检测：OTSU 自适应阈值 + 形态学开闭运算（可选阴影检测） | M1, M2 输出 | `m3_valid_mask.npy/tif` |
| M4 | 滤波：双边滤波为主，腐蚀掩膜避免边界污染 | M2, M3 输出 | `m4_optical_filtered.tif`, `m4_eroded_mask.npy` |
| M5 | 工作网格对齐：影像双线性重采样 + 掩膜最近邻重采样，一致性校验 | M4 输出, `work_grid.json` | `m5_optical_resampled.tif`, `m5_valid_mask.npy/tif` |
| M6 | 灰度归一化：百分位拉伸到 [0,1]，输出质量评价指标 | M5 输出 | `m6_optical_final.tif` |

### 二、SAR 影像预处理

| 模块 | 功能 | 说明 |
|------|------|------|
| `Pre_SAR_Align_Optical.py` | 投影对齐：将 SAR/DEM/斜距重投影到光学网格 | GDAL Warp 流式处理，支持双线性/最近邻插值；自动生成 SAR-光学共同掩膜 |
| `Pre_SAR_HistogramMatch.py` | 辐射归一化：直方图匹配（CDF 映射）或对数拉伸 | 分块处理避免内存溢出；KS 检验定量评估匹配效果 |

### 三、异源配准（RIFT）

`Registration.py` 实现了基于相位一致性（Phase Congruency）的光学-SAR 配准：

1. **预处理**：百分位归一化、掩膜腐蚀、FFT 填充
2. **Log-Gabor 滤波**：4 尺度 × 6 方向 = 24 个频域滤波器
3. **相位一致性计算**：分方向 PC 图 → 矩特征（Mmax/Mmin）→ MIM（最大索引图）
4. **特征点检测**：Mmax + Mmin 叠加图上的 Harris 角点 + NMS
5. **描述子生成**：基于 MIM 的 6×6 子块方向直方图，128 维（6×6×6=216 维），旋转不变
6. **特征匹配**：双重 Lowe 比率阈值的 FSC 策略（严格集采样 → 宽松集评估 → 全部内点精化）

输出：`registration.json`（含匹配点对、仿射变换矩阵、RMSE 等）

### 四、SAR 平台定位

`Position.py` 基于斜距交会实现 SAR 卫星平台位置解算：

1. 读取已对齐的 DEM 和斜距波段
2. 斜距单位自动识别与归一化（纳秒 → 米）
3. 提取匹配点的地面坐标 (X, Y) + 高程 Z + 斜距 R
4. 两步法稳健定位：L-M 非线性最小二乘 → 剔除离群点 → 迭代精化
5. 坐标系统：WGS84 椭球，ECEF ↔ ENU ↔ 成像坐标系全链路转换

## 环境依赖

```
Python >= 3.8
numpy
scipy
opencv-python (cv2)
rasterio
gdal (osgeo)
pyproj
Pillow
```

安装：

```bash
pip install numpy scipy opencv-python rasterio pyproj Pillow
# GDAL 建议通过 conda 安装
conda install -c conda-forge gdal
```

## 快速开始

### 1. 光学预处理

```bash
python CompleteProcess.py
```

### 2. SAR 对齐到光学网格

```python
from Pre_SAR_Align_Optical import align_to_optical, combine_common_mask

# SAR 影像对齐
align_to_optical(
    src_tif="data/sar_vv_v2.tif",
    optical_tif="output/m6_optical_final.tif",
    out_tif="output/sar_aligned.tif",
    sar_mask_tif="data/sar_mask_v2.tif",
    out_mask_tif="output/sar_aligned_mask.tif"
)

# 生成共同掩膜
combine_common_mask(
    "output/sar_aligned_mask.tif",
    "output/m3_valid_mask.tif",
    "output/common_mask.tif"
)
```

### 3. 直方图匹配 / 对数拉伸

```python
from Pre_SAR_HistogramMatch import histogram_match, logarithmic_stretch

# 方式 A：直方图匹配（CDF 映射）
histogram_match(
    sar_tif="output/sar_aligned.tif",
    optical_tif="output/m6_optical_final.tif",
    out_tif="output/sar_histogram.tif",
    sar_mask_tif="output/common_mask.tif"
)

# 方式 B：对数拉伸（适用于强度图）
logarithmic_stretch(
    sar_tif="output/sar_aligned.tif",
    optical_tif="output/m6_optical_final.tif",
    out_tif="output/sar_logarithmic.tif",
    sar_mask_tif="output/common_mask.tif",
    amplitude_mode=False  # 强度图用 False，幅度图用 True
)
```

### 4. 异源配准

```bash
python Registration.py
```

### 5. 平台定位

```bash
python Position.py
```

## 关键参数说明

### 辐射校正（M2）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `product_type` | 产品类型：`L2A_REFLECTANCE` / `L1C_REFLECTANCE` / `RADIANCE` / `DN_WITH_GAIN` / `ALREADY_REFLECTANCE` / `SKIP` | `L2A_REFLECTANCE` |
| `scale_factor` | DN 缩放系数（Sentinel-2 为 10000） | `10000.0` |

### 云检测（M3）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `cloud_threshold` | 云的最低亮度阈值 | `0.40` |
| `use_shadow_detect` | 是否启用阴影检测 | `False` |

### 滤波（M4）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `method` | 滤波方法：`bilateral` / `gaussian` / `median` / `nlmeans` | `bilateral` |

### 配准（Registration）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `nscale` | Log-Gabor 滤波器尺度数 | `4` |
| `norient` | Log-Gabor 滤波器方向数 | `6` |
| `noise_threshold` | PC 噪声阈值 | `0.05` |
| `ratio_dh` | FSC 严格集比率阈值 | `0.75` |
| `ratio_dl` | FSC 宽松集比率阈值 | `0.95` |
| `reproj_thresh` | RANSAC 重投影误差阈值 | `3.0` |
| `max_size` | 配准前图像缩放最大尺寸 | `512` |

## 坐标转换工具

`common.py` 内置了完整的坐标转换链路：

- `geodetic_to_ecef` / `ecef_to_geodetic`：经纬高 ↔ ECEF
- `ecef_to_enu` / `enu_to_ecef`：ECEF ↔ ENU
- `enu_to_imaging` / `imaging_to_enu`：ENU ↔ 成像坐标系

基于 WGS84 椭球参数（a = 6378137.0 m, f = 1/298.257223563）。

## 图像质量评价

`common.py` 中的 `evaluate_image_quality` 提供分块计算的图像质量指标：

- 对比度（Contrast）：有效像素标准差
- 信噪比（SNR）：均值 / 标准差
- 有效覆盖率（Coverage）：有效像素占比
- 结构信息量（Entropy）：归一化直方图的信息熵
- 异常区域比例（AbnormalRatio）：超出 3σ 范围的像素占比

## 注意事项

1. **执行顺序**：先运行 `CompleteProcess.py`（光学预处理），再运行 SAR 对齐和直方图匹配，最后运行配准和定位。
2. **内存管理**：大尺寸影像的直方图匹配和滤波均采用分块处理（默认 2048 行一块），避免内存溢出。
3. **NoData 约定**：全项目统一使用 `-9999.0` 作为 NoData 值，掩膜使用 `0`（无效）/ `1`（有效）。
4. **工作网格**：M1 生成的 `work_grid.json` 定义了光学预处理后的统一网格，M5 重采样和 SAR 对齐均以此为基准。
5. **m1_geometric.py**：该文件当前仅存在于 git 历史中，如需运行光学预处理流程，请从 git 中恢复。
