# m5_resample.py
"""
模块5：工作网格对齐（影像 + 掩膜 统一处理）

输入:
    data/m4_optical_filtered.tif    # M4 滤波后影像
    data/m3_valid_mask.npy          # M3 综合掩膜（源网格）
    data/m3_valid_mask.tif          # M3 掩膜 GeoTIFF（供 rasterio 读取 transform）
    data/work_grid.json             # 工作网格定义（可选）
    ref_path                        # 参考影像路径（可选，优先）

输出:
    data/m5_optical_resampled.tif   # 重采样后影像
    data/m5_valid_mask.npy          # 重采样后掩膜（numpy）
    data/m5_valid_mask.tif          # 重采样后掩膜（GeoTIFF）

职责:
    1. 获取工作网格（优先参考影像 > work_grid.json）
    2. 影像用双线性插值重采样
    3. 掩膜用最近邻插值重采样（保持二值）
    4. 校验两者尺寸/CRS/transform/bounds 完全一致
"""
import os
import json
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from common import read_band, write_band, describe, NODATA


# ============================================================
# 1. 获取工作网格
# ============================================================
def get_work_grid(ref_path=None, grid_path="output/work_grid.json"):
    """
    优先级：参考影像 > work_grid.json > 报错
    """
    # 优先级1：参考影像
    if ref_path and os.path.exists(ref_path):
        with rasterio.open(ref_path) as s:
            grid = {
                "crs": str(s.crs),
                "transform": list(s.transform)[:6],
                "width": s.width,
                "height": s.height,
                "res": [s.transform.a, abs(s.transform.e)],
                "bounds": [s.bounds.left, s.bounds.bottom,
                           s.bounds.right, s.bounds.top],
                "source": f"ref: {ref_path}"
            }
        return grid

    # 优先级2：work_grid.json
    if os.path.exists(grid_path):
        with open(grid_path) as f:
            grid = json.load(f)
        grid["source"] = f"json: {grid_path}"
        return grid

    raise FileNotFoundError(
        f"未找到参考影像 {ref_path}，也未找到 {grid_path}\n"
        f"请先定义工作网格"
    )


# ============================================================
# 2. 重采样影像（双线性）
# ============================================================
def resample_image(src_path, dst_path, grid):
    """影像重采样：双线性插值"""
    tf = Affine(*grid["transform"][:6])

    with rasterio.open(src_path) as s:
        prof = s.profile.copy()
        prof.update(
            crs=grid["crs"],
            transform=tf,
            width=grid["width"],
            height=grid["height"],
            dtype="float32",
            count=1,
            nodata=grid["nodata"]
        )
        with rasterio.open(dst_path, "w", **prof) as d:
            reproject(
                source=rasterio.band(s, 1),
                destination=rasterio.band(d, 1),
                src_transform=s.transform, src_crs=s.crs,
                dst_transform=tf, dst_crs=grid["crs"],
                src_nodata=s.nodata, dst_nodata=NODATA,
                resampling=Resampling.bilinear
            )


# ============================================================
# 3. 重采样掩膜（最近邻）
# ============================================================
def resample_mask(mask_src, src_mask_tif, dst_npy, dst_tif, grid):
    """
    掩膜重采样：最近邻插值（保持二值）

    参数:
        mask_src:     源掩膜数组（numpy bool）
        src_mask_tif: 源掩膜 GeoTIFF 路径（提供 transform/crs）
        dst_npy:      输出 npy 路径
        dst_tif:      输出 tif 路径
        grid:         目标网格
    """
    tf = Affine(*grid["transform"][:6])

    # 读取源掩膜的 transform/crs
    with rasterio.open(src_mask_tif) as s:
        src_transform = s.transform
        src_crs = s.crs
        # 校验形状
        assert mask_src.shape == (s.height, s.width), \
            f"掩膜数组 shape={mask_src.shape} 与 {src_mask_tif} 尺寸=({s.height},{s.width}) 不匹配"

    # 加波段维度
    mask_uint8 = mask_src.astype(np.uint8)[np.newaxis, ...]
    dst_uint8 = np.zeros((1, grid["height"], grid["width"]), dtype=np.uint8)

    # 最近邻重采样
    reproject(
        source=mask_uint8,
        destination=dst_uint8,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=tf, dst_crs=grid["crs"],
        resampling=Resampling.nearest,
        src_nodata=None,
        dst_nodata=0
    )
    dst_uint8 = dst_uint8[0, ...]
    mask_resampled = (dst_uint8 == 1)

    # 保存 npy
    np.save(dst_npy, mask_resampled)

    # 保存 GeoTIFF
    prof = {
        "driver": "GTiff", "dtype": "uint8", "count": 1,
        "crs": grid["crs"], "transform": tf,
        "width": grid["width"], "height": grid["height"],
        "nodata": 0,
    }
    with rasterio.open(dst_tif, "w", **prof) as d:
        d.write(dst_uint8, 1)

    return mask_resampled


# ============================================================
# 4. 一致性校验
# ============================================================
def validate_consistency(img_info, mask_info, grid):
    """校验影像和掩膜的网格一致性"""
    checks = {
        "img_size": (img_info["width"], img_info["height"]),
        "mask_size": (mask_info["width"], mask_info["height"]),
        "grid_size": (grid["width"], grid["height"]),
    }

    size_match = (img_info["width"] == mask_info["width"] == grid["width"] and
                  img_info["height"] == mask_info["height"] == grid["height"])

    # transform 比较
    img_tf = img_info["transform"]
    mask_tf = mask_info["transform"]
    tf_match = all(abs(a - b) < 1e-8 for a, b in zip(img_tf, mask_tf))

    # CRS 比较
    crs_match = (str(img_info["crs"]) == str(mask_info["crs"]))

    return {
        "size_match": bool(size_match),
        "transform_match": bool(tf_match),
        "crs_match": bool(crs_match),
        "all_match": bool(size_match and tf_match and crs_match),
        **checks
    }


# ============================================================
# 5. 主流程
# ============================================================
def Implement5(m4_path="output/m4_optical_filtered.tif",
               m3_mask_npy="output/m3_valid_mask.npy",
               m3_mask_tif="output/m3_valid_mask.tif",
               ref_path=None,
               grid_path="output/work_grid.json"):
    """
    M5: 工作网格对齐（影像 + 掩膜 统一处理）

    返回:
        (m5_tif_path, m5_mask_npy_path)
    """
    # ---- 输入检查 ----
    if not os.path.exists(m4_path):
        raise FileNotFoundError(f"输入影像不存在: {m4_path}")
    if not os.path.exists(m3_mask_npy):
        raise FileNotFoundError(f"输入掩膜不存在: {m3_mask_npy}")
    if not os.path.exists(m3_mask_tif):
        raise FileNotFoundError(f"输入掩膜tif不存在: {m3_mask_tif}")

    inp = read_band(m4_path)
    print("[输入] " + describe(inp, "滤波后"))

    # ---- 1. 获取工作网格 ----
    grid = get_work_grid(ref_path, grid_path)
    print(f"[工作网格] 来源={grid.get('source', 'unknown')}")
    print(f"[工作网格] CRS={grid['crs']}, 分辨率={grid['res']}, "
          f"尺寸={grid['width']}x{grid['height']}")

    # ---- 2. 重采样影像 ----
    out_tif = "output/m5_optical_resampled.tif"
    resample_image(m4_path, out_tif, grid)
    print(f"[重采样影像] {out_tif}")

    # ---- 3. 重采样掩膜 ----
    mask_src = np.load(m3_mask_npy)
    out_mask_npy = "output/m5_valid_mask.npy"
    out_mask_tif = "output/m5_valid_mask.tif"
    mask_resampled = resample_mask(
        mask_src, m3_mask_tif, out_mask_npy, out_mask_tif, grid
    )
    print(f"[重采样掩膜] {out_mask_tif}, 有效率={mask_resampled.mean()*100:.2f}%")

    # ---- 4. 一致性校验 ----
    out_info = read_band(out_tif)
    mask_info = read_band(out_mask_tif)
    valid = validate_consistency(out_info, mask_info, grid)
    print(f"[一致性校验] {valid}")

    if not valid["all_match"]:
        raise RuntimeError(f"M5 影像与掩膜不一致！{valid}")

    print("[输出] " + describe(out_info, "重采样后"))

    return out_tif, out_mask_npy


if __name__ == "__main__":
    Implement5()
