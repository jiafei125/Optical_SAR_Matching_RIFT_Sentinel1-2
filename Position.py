# geo_localization.py
"""
基于斜距交会的SAR平台定位解算【通用版：支持星载/机载/无人机SAR】
前提：
    SAR图、DEM、斜距波段都已由 m7 对齐到同一光学网格
    registration.json 中的匹配点坐标是同一网格下的像素坐标
流程：
    1. 读取已对齐的 DEM 和斜距
    2. 斜距单位转换（不再自动猜测，由参数指定单位模式）
    3. 读取匹配点（直接使用像素坐标）
    4. 提取每个匹配点的 (X, Y, Z, R)，自动剔除DEM/斜距采样NaN点
    5. 两步法稳健定位（参数化残差阈值，适配不同SAR）
    6. 输出平台ECEF位置 + 诊断信息（残差、保留/剔除点索引）

入参说明：
    slant_unit: {"meters", "time_ns"}
        "meters"：斜距已经是米；
        "time_ns"：斜距存储为往返时间纳秒，执行 R = c*t/2
    init_platform_ecef: 平台ECEF初始猜测坐标，**机载/无人机必须外部传入，星载S‑1也建议传入**
    residual_threshold: 残差阈值(m)，星载150~250；机载20~80；无人机5~30
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


def read_aligned_products(dem_tif, slant_tif):
    """
    读取 m7 对齐后的 DEM 和斜距（已与光学/SAR同网格）
    返回: dict 包含 transform, crs, shape, dem, slant
    """
    log_info("读取已对齐的 DEM 和斜距:")

    # ----------DEM----------
    with rasterio.open(dem_tif) as src:
        dem_arr = src.read(1).astype(np.float64)
        dem_nodata = src.nodata
        transform = src.transform
        crs = src.crs
        shape = (src.height, src.width)
    # 将 NoData 转为 NaN
    if dem_nodata is not None:
        dem_arr[dem_arr == dem_nodata] = np.nan
    dem_arr[~np.isfinite(dem_arr)] = np.nan

    valid_dem = np.isfinite(dem_arr).sum()
    log_info(f"DEM: {shape}, 有效像素 {valid_dem}/{dem_arr.size}")
    if valid_dem > 0:
        log_info(f"     范围: [{np.nanmin(dem_arr):.1f}, {np.nanmax(dem_arr):.1f}] m")
    else:
        raise RuntimeError("DEM无有效像素")

    # ----------斜距----------
    with rasterio.open(slant_tif) as src:
        slant_arr = src.read(1).astype(np.float64)
        slant_nodata = src.nodata
        assert (src.height, src.width) == shape, "❌ 斜距与DEM不同网格"
        assert src.crs == crs, "❌ 斜距与DEM CRS不同"
    if slant_nodata is not None:
        slant_arr[slant_arr == slant_nodata] = np.nan
    slant_arr[~np.isfinite(slant_arr)] = np.nan

    valid_slant = np.isfinite(slant_arr).sum()
    log_info(f"斜距: {shape}, 有效像素 {valid_slant}/{slant_arr.size}")
    if valid_slant > 0:
        log_info(f"     范围: [{np.nanmin(slant_arr):.4e}, {np.nanmax(slant_arr):.4e}]")
    else:
        raise RuntimeError("斜距波段无有效像素")

    return {
        "transform": transform,
        "crs": crs,
        "shape": shape,
        "dem": dem_arr,
        "slant": slant_arr,
    }


def bilinear(arr, x, y):
    """
    双线性插值，x=col，y=row
    """
    h, w = arr.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    dx, dy = x - x0, y - y0
    val = (
        arr[y0, x0] * (1 - dx) * (1 - dy)
        + arr[y0, x1] * dx * (1 - dy)
        + arr[y1, x0] * (1 - dx) * dy
        + arr[y1, x1] * dx * dy
    )
    return float(val)


def normalize_slant(slant_arr, slant_unit="meters"):
    """
    斜距单位转换【通用版本，删除自动猜测】
    :param slant_arr: 原始斜距数组
    :param slant_unit: "meters" 已经是米; "time_ns" 往返时间纳秒 R = c*t/2
    :return slant_arr_m: 斜距，单位米
    """
    valid = slant_arr[np.isfinite(slant_arr) & (slant_arr > 0)]
    if valid.size == 0:
        raise ValueError("❌ 斜距波段无有效像素")
    median = np.median(valid)
    log_info(f"\n原始斜距中位数: {median:.4e}")

    if slant_unit == "time_ns":
        log_info("斜距输入为往返时间(纳秒)，执行转换 R = c * t / 2")
        slant_arr_m = slant_arr * 1e-9 * C_LIGHT / 2.0
    elif slant_unit == "meters":
        log_info("斜距输入单位已经是米，不做缩放")
        slant_arr_m = slant_arr.copy()
    else:
        raise ValueError(f"slant_unit 仅支持 'meters' / 'time_ns'，输入={slant_unit}")

    valid2 = slant_arr_m[np.isfinite(slant_arr_m) & (slant_arr_m > 0)]
    log_info(f"转换后斜距范围: [{valid2.min():.1f}, {valid2.max():.1f}] 米")
    return slant_arr_m


def load_match_pairs(reg_json, shape):
    """
    从 registration.json 中读取匹配点对，边界检查、去重
    """
    log_info("读取匹配点:")
    with open(reg_json, "r", encoding="utf-8") as f:
        reg = json.load(f)

    new_h, new_w = shape
    log_info(f"配准方法: {reg.get('method', 'unknown')}")
    cps_list = reg.get("cps", [])
    log_info(f"cps 总数: {len(cps_list)}")

    pairs = []
    seen = set()
    for cp in cps_list:
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

    log_info(f"边界过滤+去重后有效匹配点: {len(pairs)}")
    return pairs


def extract_ground_points(pairs, transform, dem_arr, slant_arr):
    """
    对每个匹配点提取地面投影坐标(X,Y)、高程Z、斜距R
    自动剔除 DEM/Z 或者斜距R为NaN的点
    返回 X_arr,Y_arr,Z_arr,R_arr，以及原始有效点索引
    """
    X_list, Y_list, Z_list, R_list = [], [], [], []
    keep_idx = []
    for idx, (sar_x, sar_y, opt_x, opt_y) in enumerate(pairs):
        # 像素中心转投影地理XY
        X, Y = xy(transform, opt_y, opt_x, offset="center")
        Z = bilinear(dem_arr, opt_x, opt_y)
        R = bilinear(slant_arr, sar_x, sar_y)
        # 过滤插值得到NaN（DEM或者斜距无效）
        if np.isfinite(Z) and np.isfinite(R) and R > 1.0:
            X_list.append(X)
            Y_list.append(Y)
            Z_list.append(Z)
            R_list.append(R)
            keep_idx.append(idx)

    X_arr = np.array(X_list, dtype=np.float64)
    Y_arr = np.array(Y_list, dtype=np.float64)
    Z_arr = np.array(Z_list, dtype=np.float64)
    R_arr = np.array(R_list, dtype=np.float64)
    log_info(f"提取地面点，剔除NaN后剩余 {len(X_arr)} / {len(pairs)}")
    return X_arr, Y_arr, Z_arr, R_arr, keep_idx


def localization(points_ecef, slant_ranges, init_ecef,
                 residual_threshold=200.0, max_iter=3):
    """
    两步法稳健斜距交会定位【通用，星载/机载/无人机】
    :param points_ecef: N×3 地面点ECEF坐标
    :param slant_ranges: N 观测斜距（米）
    :param init_ecef: 平台ECEF初始猜测 [x,y,z]
    :param residual_threshold: 残差阈值，米；星载150‑250，机载20‑80，无人机5‑30
    :param max_iter: 迭代剔除离群点最大轮数
    :return: platform_ecef, final_rmse_m, valid_mask_idx, residuals_all
    """
    log_info(f"斜距交会定位，残差阈值={residual_threshold:.1f} m")
    n_total = len(points_ecef)
    if n_total < 4:
        raise ValueError(f"定位需要至少4个点，当前仅{n_total}")

    valid_idx = np.arange(n_total)
    platform = np.array(init_ecef, dtype=np.float64).copy()
    residuals_all = np.zeros(n_total, dtype=np.float64)
    final_rmse = None

    for iteration in range(max_iter):
        pts = points_ecef[valid_idx]
        rs = slant_ranges[valid_idx]

        def residual(p):
            return np.linalg.norm(p[None, :] - pts, axis=1) - rs

        result = least_squares(
            residual, platform,
            method='lm',
            max_nfev=500,
            ftol=1e-12, xtol=1e-12
        )
        platform = result.x
        res = residual(platform)
        rmse = float(np.sqrt(np.mean(res ** 2)))
        log_info(f"  迭代 {iteration+1}: {len(valid_idx)} 点, RMSE={rmse:.2f} m")

        # 保存本轮全部点残差
        residuals_all[valid_idx] = res
        # 剔除离群
        new_valid = np.abs(res) < residual_threshold
        if new_valid.all():
            break
        valid_idx = valid_idx[new_valid]
        if len(valid_idx) < 4:
            log_info("  剩余点不足4个，停止迭代剔除离群点")
            break

    # 用最终有效点重新计算一次残差与RMSE
    final_residuals = residual(platform)
    final_rmse = float(np.sqrt(np.mean(final_residuals ** 2)))
    residuals_all[valid_idx] = final_residuals
    log_info(f"定位完成，最终内点 {len(valid_idx)}，最终RMSE={final_rmse:.2f} m")
    return platform, final_rmse, valid_idx, residuals_all


# ----------------示例调用参考----------------
if __name__ == "__main__":
    dem_tif = r"output/dem_aligned.tif"
    slant_tif = r"output/slant_aligned.tif"
    reg_json = r"output/registration.json"

    # ==========参数，根据SAR类型修改==========
    # slant_unit: "meters" / "time_ns"
    slant_unit = "meters"
    # ！！机载/无人机必须填入真实GPS/IMU得到的ECEF初值；星载S‑1也建议填入轨道初值
    init_ecef = np.array([4200000.0, 1200000.0, 5400000.0])
    # 残差阈值：星载150‑250；机载20‑80；无人机5‑30
    res_thresh = 200.0

    prod = read_aligned_products(dem_tif, slant_tif)
    slant_m = normalize_slant(prod["slant"], slant_unit=slant_unit)

    pairs = load_match_pairs(reg_json, prod["shape"])
    X_arr, Y_arr, Z_arr, R_arr, keep_idx = extract_ground_points(
        pairs, prod["transform"], prod["dem"], slant_m
    )
    if len(X_arr) < 4:
        raise RuntimeError("可用地面点不足4个，无法定位")

    # 投影地理坐标(U‑TM X,Y,Z) → ECEF大地地心坐标
    transformer = Transformer.from_crs(prod["crs"], "EPSG:4978", always_xy=True)
    x_ecef, y_ecef, z_ecef = transformer.transform(X_arr, Y_arr, Z_arr)
    points_ecef = np.column_stack([x_ecef, y_ecef, z_ecef])

    platform_ecef, rmse_m, valid_idx, res_all = localization(
        points_ecef, R_arr, init_ecef,
        residual_threshold=res_thresh, max_iter=3
    )
    lat, lon, alt = ecef_to_geodetic(*platform_ecef)
    log_info(f"解算平台ECEF XYZ: {platform_ecef.round(2)}")
    log_info(f"解算平台地理(LLA): lat={lat:.4f}°, lon={lon:.4f}°, alt={alt:.2f} m")
    log_info(f"定位RMSE = {rmse_m:.2f} m")
