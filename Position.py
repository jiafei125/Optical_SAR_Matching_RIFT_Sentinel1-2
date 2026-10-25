# geo_localization.py
"""
基于斜距交会的SAR平台定位解算

前提：
    SAR图、DEM、斜距波段都已由 m7 对齐到光学网格（10980×10980，EPSG:32651，10m）
    registration.json 中的匹配点坐标是同一网格下的像素坐标

流程：
    1. 读取已对齐的 DEM 和斜距
    2. 斜距单位归一化
    3. 读取匹配点（直接使用像素坐标）
    4. 提取每个匹配点的 (X, Y, Z, R)
    5. 两步法稳健定位
    6. 输出平台位置 + 匹配点数据
"""
import time
start_time = time.perf_counter()

import os
import json
import numpy as np
import rasterio
from scipy.optimize import least_squares
from pyproj import Transformer
from rasterio.transform import xy
from common import geodetic_to_ecef, ecef_to_geodetic, log_info


# 常量
C_LIGHT = 299792458.0           # 光速（m/s），斜距单位归一化时使用（时间 → 距离）
SENTINEL1_ORBIT_H = 700000.0    # Sentinel-1 轨道高度（m），惯导初值自动判定用


# 读取 m7 对齐后的 DEM 和斜距
"""
直接读取 m7 对齐后的 DEM 和斜距（已与光学/SAR 同网格）
返回: dict 包含 transform, crs, shape, dem, slant
"""
def read_aligned_products(dem_tif, slant_tif):
    log_info("读取已对齐的 DEM 和斜距:")

    # ----------DEM----------
    with rasterio.open(dem_tif) as src:
        dem_arr = src.read(1).astype(np.float64)
        dem_nodata = src.nodata
        transform = src.transform
        crs = src.crs
        shape = (src.height, src.width)
    # 将 NoData 转为 NaN ，后续用 np.isfinite 判断有效性，比 == -9999 更通用
    if dem_nodata is not None:
        dem_arr[dem_arr == dem_nodata] = np.nan
    dem_arr[~np.isfinite(dem_arr)] = np.nan

    log_info(f"DEM: {shape}, 有效像素 {np.isfinite(dem_arr).sum()}/{dem_arr.size}")
    log_info(f"     范围: [{np.nanmin(dem_arr):.1f}, {np.nanmax(dem_arr):.1f}] m")


    # ----------斜距----------
    with rasterio.open(slant_tif) as src:
        slant_arr = src.read(1).astype(np.float64)
        slant_nodata = src.nodata
        assert (src.height, src.width) == shape, "❌ 斜距与DEM不同网格"
        assert src.crs == crs, "❌ 斜距与DEM CRS不同"
    # 将 NoData 转为 NaN ，后续用 np.isfinite 判断有效性，比 == -9999 更通用
    if slant_nodata is not None:
        slant_arr[slant_arr == slant_nodata] = np.nan
    slant_arr[~np.isfinite(slant_arr)] = np.nan

    log_info(f"斜距: {shape}, 有效像素 {np.isfinite(slant_arr).sum()}/{slant_arr.size}")
    log_info(f"     范围: [{np.nanmin(slant_arr):.4e}, {np.nanmax(slant_arr):.4e}]")


    return {
        "transform": transform,     # 仿射矩阵，实现 像素坐标 (col, row) ↔ 地理投影坐标 (x,y) 的互相转换
        "crs": crs,                 # 用于投影反算
        "shape": shape,             # (H, W)
        "dem": dem_arr,             # 高程数组
        "slant": slant_arr,         # 斜距数组
    }

# 双线性插值
"""
坐标变换之后，得到的位置不一定是整数索引，就要用这个函数插值
双线性插值核心公式：四个点分别乘权重，加权求和。权重离该点越近，权重越大。
"""
def bilinear(arr, x, y):
    h, w = arr.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    dx, dy = x - x0, y - y0
    return float(
        arr[y0, x0] * (1 - dx) * (1 - dy) +
        arr[y0, x1] * dx * (1 - dy) +
        arr[y1, x0] * (1 - dx) * dy +
        arr[y1, x1] * dx * dy
    )

# 斜距单位归一化
"""
自动识别斜距数组的单位（纳秒/米/其他），统一转换为米
"""
def normalize_slant(slant_arr):
    valid = slant_arr[np.isfinite(slant_arr) & (slant_arr > 0)]
    if valid.size == 0:
        raise ValueError("❌ 斜距波段无有效像素")
    median = np.median(valid)
    log_info(f"\n原始斜距中位数: {median:.4e}米")

    # 分支：如果是纳秒存储的时间值（一般几千以内）→转距离
    if median < 2000:
        log_info("检测为斜距时间（纳秒），转换为距离米：R = c * t /2")
        slant_arr = slant_arr * 1e-9 * C_LIGHT / 2.0
    # Sentinel-1 IW正常斜距：800~1000km
    elif 8e5 <= median <= 1e6:
        log_info("Sentinel-1 IW斜距，单位已经是米，无需缩放")
    else:
        raise ValueError(f"❌ 无法识别斜距数值: median={median:.4e}")

    valid2 = slant_arr[np.isfinite(slant_arr) & (slant_arr > 0)]
    log_info(f"转换后斜距范围: [{valid2.min():.1f}, {valid2.max():.1f}] 米")
    return slant_arr

# 读取匹配点（直接使用像素坐标）
"""
从 registration.json 中读取匹配点对，过滤无效值，去重（重复点会让最小二乘解过度加权）。
由于 SAR、光学、DEM、斜距都已对齐到同一网格，
匹配点坐标可以直接使用，无需转换。
"""
def load_match_pairs(reg_json, shape):

    log_info("读取匹配点:")

    with open(reg_json, "r", encoding="utf-8") as f:
        reg = json.load(f)

    new_h, new_w = shape
    log_info(f"配准方法: {reg.get('method', 'unknown')}")
    log_info(f"cps 总数: {len(reg.get('cps', []))}")

    # 遍历 cps，做边界检查和去重
    pairs = []
    seen = set()
    for cp in reg["cps"]:
        sar_col, sar_row = cp["sar_xy_ori"]
        opt_col, opt_row = cp["opt_xy_ori"]
        # 边界检查
        if not (0 <= sar_col < new_w and 0 <= sar_row < new_h):
            continue
        if not (0 <= opt_col < new_w and 0 <= opt_row < new_h):
            continue
        # 去重
        key = (round(sar_col, 3), round(sar_row, 3),
               round(opt_col, 3), round(opt_row, 3))
        if key in seen:
            continue
        seen.add(key)

        pairs.append((sar_col, sar_row, opt_col, opt_row))

    log_info(f"有效匹配点: {len(pairs)}")
    return pairs


# 提取 (X, Y, Z, R)
"""
对每个匹配点提取地面坐标 (X, Y)、高程 Z、斜距 R
"""
def extract_ground_points(pairs, transform, dem_arr, slant_arr):

    X_list, Y_list, Z_list, R_list = [], [], [], []
    for sar_x, sar_y, opt_x, opt_y in pairs:

        # 光学像素坐标 -> 地理坐标
        # opt_x=col, opt_y=row，取像素中心地理XY
        X, Y = xy(transform, opt_y, opt_x, offset="center")
        # DEM 采样高程
        Z = bilinear(dem_arr, opt_x, opt_y)
        # 斜距波段采样斜距
        R = bilinear(slant_arr, sar_x, sar_y)

        X_list.append(X)
        Y_list.append(Y)
        Z_list.append(Z)
        R_list.append(R)

    X_arr = np.array(X_list)
    Y_arr = np.array(Y_list)
    Z_arr = np.array(Z_list)
    R_arr = np.array(R_list)

    return X_arr, Y_arr, Z_arr, R_arr


# 两步法稳健定位
"""
斜距交会定位的核心，用两步迭代剔除离群点。
两步法：先定位 → 剔除离群点 → 再定位

threshold=200.0:表示残差 > 200 米的点被剔除
"""
def localization(points_ecef, slant_ranges, init_ecef,
                 threshold=200.0, max_iter=3):

    log_info("斜距交会定位:")

    valid_idx = np.arange(len(points_ecef))     # 初始：全部点有效
    platform = init_ecef.copy()                 # 卫星位置初值
    rmse = None

    for iteration in range(max_iter):
        pts = points_ecef[valid_idx]
        rs = slant_ranges[valid_idx]

        # 残差函数，残差 = 卫星到地面点的欧氏距离 - 观测斜距
        def residual(p):
            return np.linalg.norm(p[None, :] - pts, axis=1) - rs

        # L-M非线性最小二乘求解
        # platform：初值p，卫星 ECEF 初始猜测坐标
        # method='lm':LM = 高斯牛顿 + 梯度下降混合算法
        # max_nfev=500:最大残差函数调用次数（不是迭代次数
        # ftol=1e-12:函数收敛阈值,残差平方和的相对变化 f < ftol 时停止迭代
        # xtol=1e-12:待求参数（卫星坐标）的收敛阈值,待求向量p更新步长足够小，参数几乎不再变化，则停止
        result = least_squares(
            residual, platform,
            method='lm',
            max_nfev=500,
            ftol=1e-12, xtol=1e-12
        )

        # result.x：最优解，就是更新后的卫星坐标
        platform = result.x         # 更新卫星位置
        res = residual(platform)    # 各个内点残差
        rmse = float(np.sqrt(np.mean(res ** 2)))
        log_info(f"  迭代 {iteration+1}: {len(valid_idx)} 点, RMSE={rmse:.2f} m")

        # 剔除离群点
        new_valid = np.abs(res) < threshold

        if new_valid.all():
            break   # 所有点都通过，无需再剔除
        valid_idx = valid_idx[new_valid]
        if len(valid_idx) < 4:
            log_info("  剩余点不足4个，停止迭代")
            break

    return platform, rmse, valid_idx

