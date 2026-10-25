"""
通用工具
"""

import os
import json
import time
import numpy as np
import rasterio
import cv2

NODATA = -9999.0
# WORK_CRS = "EPSG:32650"
# WORK_RES = 10.0

# 控制是否使用print打印信息
def log_info(msg, log_flag= True):
    if log_flag:
        print(msg)

# ============ .tif 读写 ============
def save_tif(path, array, reference_tif=None, geotrans=None, crs=None,
             dtype='float32'):
    import rasterio
    from rasterio.transform import from_origin
    array = array.astype(dtype)
    h, w = array.shape

    if reference_tif is not None:
        with rasterio.open(reference_tif) as ref:
            transform = ref.transform
            crs = ref.crs
    elif geotrans is not None:
        transform = from_origin(geotrans[0], geotrans[3],
                                geotrans[1], abs(geotrans[5]))
    else:
        transform = from_origin(0, 0, 1, 1)

    with rasterio.open(path, 'w', driver='GTiff',
                       height=h, width=w, count=1, dtype=dtype,
                       crs=crs, transform=transform) as dst:
        dst.write(array, 1)


def load_tif(path):
    import rasterio
    with rasterio.open(path) as src:
        return src.read(1), src.profile


def get_tif_info(path):
    import rasterio
    with rasterio.open(path) as src:
        return {
            'shape': (src.height, src.width),
            'crs': str(src.crs),
            'transform': src.transform,
            'bounds': src.bounds,
            'dtype': src.dtypes[0],
            'nodata': src.nodata,
            'count': src.count
        }

def read_band(path, band=1):
    with rasterio.open(path) as src:
        info = {
            "arr": src.read(band).astype(np.float32),
            "profile": src.profile.copy(),
            "transform": src.transform,
            "crs": src.crs,
            "nodata": src.nodata,
            "bounds": src.bounds,
            "width": src.width,
            "height": src.height,
            "path": path,
        }
    return info

def write_band(path, arr, profile, nodata=NODATA):
    profile = profile.copy()
    profile.update(dtype="float32", count=1, nodata=nodata)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)

def describe(res, name="img"):
    arr = res["arr"]
    nodata = res["nodata"] if res["nodata"] is not None else NODATA
    mask = (arr == nodata) | (~np.isfinite(arr))
    valid = arr[~mask]
    if valid.size == 0:
        min_val = max_val = mean_val = std_val = float(nodata)
        in_range = None
    else:
        min_val = float(valid.min())
        max_val = float(valid.max())
        mean_val = float(valid.mean())
        std_val = float(valid.std())
        in_range = bool((valid.min() >= 0) & (valid.max() <= 1))

    trans = res["profile"]["transform"]
    res_x, res_y = trans.a, abs(trans.e)

    return (f"{name}: shape={arr.shape}, crs={res['crs']}, res=({res_x:.2f},{res_y:.2f})\n "
            f"bounds={res['bounds']}\n "
            f"range=[{min_val:.4f},{max_val:.4f}], mean={mean_val:.4f}, "
            f"std={std_val:.4f}, in_range={in_range}")


def save_work_grid(profile, path="data/work_grid.json"):
    """
    保存工作网格，供后续 M5 重采样对齐使用

    参数:
        profile: rasterio profile 字典（含 crs/transform/width/height/nodata）
        path:    输出 json 路径
    """
    import json
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)

    transform = profile["transform"]
    grid = {
        "crs": str(profile["crs"]),
        "transform": list(transform)[:6],
        "width": profile["width"],
        "height": profile["height"],
        "nodata": profile.get("nodata"),
        "res": [transform.a, abs(transform.e)],
    }
    with open(path, "w") as f:
        json.dump(grid, f, indent=2)
    return grid


def load_work_grid(path="data/work_grid.json"):
    """读取工作网格定义"""
    import json
    with open(path) as f:
        return json.load(f)

# ============ 复数数据 ============
def save_complex_npy(path, complex_array):
    np.savez(path, real=complex_array.real, imag=complex_array.imag)


def load_complex_npy(path):
    data = np.load(path)
    return data['real'] + 1j * data['imag']



# coord_transform.py
"""
坐标转换：地心坐标系(ECEF) <-> 东北天坐标系(ENU) <-> 成像坐标系
斜距交会定位在 ECEF 下计算最简单，是标准的欧氏距离
"""
import numpy as np

# WGS84 椭球参数
WGS84_A = 6378137.0  # 地球长半轴（米）
WGS84_F = 1.0 / 298.257223563  # 扁率
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2  # 从扁率 f 计算第一偏心率平方的标准公式

"""
经纬高 -> 地心坐标系（ECEF）

公式：
    N = a/√(1 - e^2 * (sinM)^2)
    x = (N + h) * cosM * cosL
    y = (N + h) * cosM * sinL
    z = (N * (1-e^2) + h) * sinM
其中：
    N为卯酉圈曲率半径
    M为纬度，L为经度
    (x,y,z)为该点在地球三维空间中的绝对坐标
"""


def geodetic_to_ecef(lat_deg, lon_deg, h):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    x = (N + h) * np.cos(lat) * np.cos(lon)
    y = (N + h) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - WGS84_E2) + h) * np.sin(lat)
    return np.array([x, y, z])


"""
地心坐标系（ECEF） -> 经纬高
从 ECEF 到经纬度没有解析解，需要迭代
"""


def ecef_to_geodetic(x, y, z, max_iter=10, tol=1e-9):
    lon = np.arctan2(y, x)

    # 给定初值后，反复用新算出的N和h更新纬度，直到收敛。
    p = np.sqrt(x ** 2 + y ** 2)
    lat = np.arctan2(z, p * (1 - WGS84_E2))
    h = 0.0
    for _ in range(max_iter):
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat_new = np.arctan2(z, p * (1 - WGS84_E2 * N / (N + h)))
        if abs(lat_new - lat) < tol:
            lat = lat_new
            break
        lat = lat_new
    return np.rad2deg(lat), np.rad2deg(lon), h


"""
地心坐标系（ECEF） -> 东北天坐标系（ENU） 的旋转矩阵
公式：
    R = [-sinL        ,cosL          ,  0]
        [-cosL * sinM ,-sinL * sinM ,cosM]
        [cosL * cosM  ,cosM * sinL  ,sinM]
其中：
    M为纬度，L为经度
"""


def ecef_to_enu_matrix(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    R = np.array([
        [-np.sin(lon), np.cos(lon), 0],
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    ])
    return R


"""
地心坐标系（ECEF） -> 东北天坐标系（ENU）
先平移，再旋转
"""


def ecef_to_enu(point_ecef, origin_lat, origin_lon, origin_h):
    origin_ecef = geodetic_to_ecef(origin_lat, origin_lon, origin_h)
    R = ecef_to_enu_matrix(origin_lat, origin_lon)
    return R @ (point_ecef - origin_ecef)


"""
东北天坐标系（ENU） -> 地心坐标系（ECEF）
先逆旋转（旋转矩阵是正交矩阵，转置就是逆），再逆平移
"""


def enu_to_ecef(point_enu, origin_lat, origin_lon, origin_h):
    origin_ecef = geodetic_to_ecef(origin_lat, origin_lon, origin_h)
    R = ecef_to_enu_matrix(origin_lat, origin_lon)
    return R.T @ point_enu + origin_ecef


"""
东北天坐标系（ENU） -> 成像坐标系
把 ENU 点旋转到以平台航向为 X 轴的成像坐标系
公式：
    R = [cosγ ,sinγ ,0]
        [-sinγ,cosγ ,0]
        [0    ,0    ,1]    
"""


def enu_to_imaging(point_enu, heading_deg):
    gamma = np.deg2rad(heading_deg)
    R = np.array([
        [np.cos(gamma), np.sin(gamma), 0],
        [-np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])
    return R @ point_enu


"""
成像坐标系 -> 东北天坐标系（ENU）
逆旋转(旋转矩阵转置即可实现)
"""


def imaging_to_enu(point_xyz, heading_deg):
    gamma = np.deg2rad(heading_deg)
    R = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])
    return R @ point_xyz


# 对数据进行统一整理并转换成后续定位模块能够直接读取的标准 JSON 格式。
def build_unified_result(
    method, direction, kp_sar, kp_opt, inlier_matches, good_count,
    H, sar_scale, opt_scale, transform, crs, sar_shape, opt_shape, crop=None
):
    pts_sar_ds = np.float32([kp_sar[m.queryIdx].pt for m in inlier_matches]).reshape(-1, 2)
    pts_opt_ds = np.float32([kp_opt[m.trainIdx].pt for m in inlier_matches]).reshape(-1, 2)

    # 主位移：内点平均偏移，统一为 opt - sar
    dx_avg_ds = float(np.mean(pts_opt_ds[:, 0] - pts_sar_ds[:, 0]))
    dy_avg_ds = float(np.mean(pts_opt_ds[:, 1] - pts_sar_ds[:, 1]))
    dx_avg_ori = dx_avg_ds / sar_scale
    dy_avg_ori = dy_avg_ds / sar_scale

    # 辅助位移：中心点经 H 投影
    h_sar, w_sar = sar_shape
    h_opt, w_opt = opt_shape
    center = np.array([[[w_sar / 2.0, h_sar / 2.0]]], dtype=np.float32)
    center_proj = cv2.perspectiveTransform(center, H)
    dx_center_ds = float(center_proj[0, 0, 0] - w_opt / 2.0)
    dy_center_ds = float(center_proj[0, 0, 1] - h_opt / 2.0)

    # 统一 RMSE：全部用 cv2.transform
    pts_sar_h = pts_sar_ds.reshape(-1, 1, 2)
    pts_proj = cv2.transform(pts_sar_h, H[0:2, :]).reshape(-1, 2)
    residual = pts_opt_ds - pts_proj
    rmse_ds = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
    rmse_ori = rmse_ds / sar_scale

    # 统一地理偏移
    geo_east = dx_avg_ori * transform.a
    geo_north = dy_avg_ori * transform.e
    geo_mag = float(np.hypot(geo_east, geo_north))

    inlier_ratio = float(len(inlier_matches) / good_count) if good_count > 0 else 0.0

    return {
        "method": method,
        "direction": direction,
        "crop": crop,
        "scale": {"sar": float(sar_scale), "opt": float(opt_scale)},
        "pixel_offset_avg_ds": {"dx": dx_avg_ds, "dy": dy_avg_ds},
        "pixel_offset_avg_ori": {"dx": dx_avg_ori, "dy": dy_avg_ori},
        "pixel_offset_center_ds": {"dx": dx_center_ds, "dy": dy_center_ds},
        "geo_offset_meter": {
            "east": float(geo_east),
            "north": float(geo_north),
            "magnitude": geo_mag
        },
        "rmse_pixel_ds": rmse_ds,
        "rmse_pixel_ori": rmse_ori,
        "good_match_count": int(good_count),
        "inlier_count": int(len(inlier_matches)),
        "inlier_ratio": inlier_ratio,
        "confidence": inlier_ratio,
        "matrix_ds": H.tolist(),
        "crs": str(crs),
        "cps": [
            {
                "sar_xy_ds": [float(kp_sar[m.queryIdx].pt[0]), float(kp_sar[m.queryIdx].pt[1])],
                "opt_xy_ds": [float(kp_opt[m.trainIdx].pt[0]), float(kp_opt[m.trainIdx].pt[1])],
                "sar_xy_ori": [float(kp_sar[m.queryIdx].pt[0] / sar_scale), float(kp_sar[m.queryIdx].pt[1] / sar_scale)],
                "opt_xy_ori": [float(kp_opt[m.trainIdx].pt[0] / opt_scale), float(kp_opt[m.trainIdx].pt[1] / opt_scale)]
            }
            for m in inlier_matches
        ]
    }

def evaluate_image_quality(img, mask=None, nodata=-9999.0, chunk_size=2048):
    """
    计算图像质量评价指标（分块处理，避免大图内存溢出）

    参数:
        img: 2D numpy array (float32)
        mask: 2D bool array (True=有效区域)
        nodata: 无效值
        chunk_size: 分块大小

    返回:
        dict: 包含各项指标的字典
    """
    h, w = img.shape

    # 全局统计变量
    valid_pixels = 0
    sum_val = 0.0
    sum_sq_val = 0.0
    min_val = np.inf
    max_val = -np.inf

    # 用于直方图熵计算（假设值范围在0~1或0~255，这里用分位数动态范围）
    hist_bins = 256
    global_hist = np.zeros(hist_bins, dtype=np.int64)

    # 用于异常检测（基于均值和标准差）
    # 先分块统计，第二遍再算异常

    # ---------- 第一遍：计算全局均值、标准差、极值 ----------
    for r0 in range(0, h, chunk_size):
        r1 = min(r0 + chunk_size, h)
        for c0 in range(0, w, chunk_size):
            c1 = min(c0 + chunk_size, w)

            block = img[r0:r1, c0:c1]
            if mask is not None:
                block_mask = mask[r0:r1, c0:c1]
                valid = block_mask & np.isfinite(block) & (block != nodata)
            else:
                valid = np.isfinite(block) & (block != nodata)

            if valid.sum() == 0:
                continue

            vals = block[valid].astype(np.float64)
            valid_pixels += vals.size
            sum_val += vals.sum()
            sum_sq_val += (vals ** 2).sum()
            min_val = min(min_val, vals.min())
            max_val = max(max_val, vals.max())

    if valid_pixels == 0:
        return {"error": "无有效像素"}

    mean_val = sum_val / valid_pixels
    var_val = (sum_sq_val / valid_pixels) - (mean_val ** 2)
    std_val = np.sqrt(max(var_val, 0))

    # ---------- 第二遍：计算直方图、清晰度、异常比例 ----------
    # 清晰度：拉普拉斯方差需要图像梯度，我们用Sobel近似
    lap_var = 0.0
    lap_count = 0

    # 异常像素计数（均值 ± 3倍标准差）
    abnormal_count = 0
    z_thresh = 3.0

    # 分块计算
    for r0 in range(0, h, chunk_size):
        r1 = min(r0 + chunk_size, h)
        for c0 in range(0, w, chunk_size):
            c1 = min(c0 + chunk_size, w)

            block = img[r0:r1, c0:c1].astype(np.float32)
            if mask is not None:
                block_mask = mask[r0:r1, c0:c1]
                valid = block_mask & np.isfinite(block) & (block != nodata)
            else:
                valid = np.isfinite(block) & (block != nodata)

            if valid.sum() == 0:
                continue

            vals = block[valid]

            # 直方图统计（归一化到0~1后分箱）
            v_norm = (vals - min_val) / (max_val - min_val + 1e-10)
            hist, _ = np.histogram(v_norm, bins=hist_bins, range=(0, 1))
            global_hist += hist.astype(np.int64)


            # 异常像素（均值 ± 3σ）
            if std_val > 1e-6:
                z_score = np.abs(vals - mean_val) / std_val
                abnormal_count += int((z_score > z_thresh).sum())


    # 对比度（全局标准差）
    contrast = std_val

    # 信噪比（SNR = 均值 / 标准差）
    snr = mean_val / (std_val + 1e-10)

    # 有效覆盖率
    total_pixels = h * w
    coverage = valid_pixels / total_pixels

    # 结构信息量（信息熵）
    p = global_hist / (global_hist.sum() + 1e-10)
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p))

    # 异常区域比例
    abnormal_ratio = abnormal_count / valid_pixels


    return {
        "对比度(Contrast)": float(contrast),
        "信噪比(SNR)": float(snr),
        "有效覆盖率(Coverage)": float(coverage),
        "结构信息量(Entropy)": float(entropy),
        "异常区域比例(AbnormalRatio)": float(abnormal_ratio),
        "有效像素数": int(valid_pixels),
        "均值": float(mean_val),
        "标准差": float(std_val),
        "最小值": float(min_val),
        "最大值": float(max_val),
    }
