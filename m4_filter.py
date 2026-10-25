# m4_filter.py
"""
模块4：滤波
输入:  data/m2_optical_refl.tif + data/m3_valid_mask.npy
输出:  data/m4_optical_filtered.tif
       data/m4_eroded_mask.npy（供检查模块用）

策略:
    1. 用腐蚀掩膜避免边界污染（滤波计算时）
    2. 输出时用原始掩膜保留全部有效区域
    3. 输出边界像素用原值填充（不滤波）
"""
import os
import numpy as np
import cv2
from common import read_band, write_band, describe, NODATA


def filter_optical(arr, method="bilateral", valid_mask=None):
    """只对有效区域滤波，边界保留原值"""
    arr_f = arr.astype(np.float32)
    mask = valid_mask.copy() if valid_mask is not None else np.ones_like(arr_f, dtype=bool)

    # 有效值填充（避免NODATA污染滤波）
    roi_all = arr_f[mask]
    roi_clean = roi_all[np.isfinite(roi_all) & (roi_all != NODATA)]
    if roi_clean.size == 0:
        raise RuntimeError("有效像素为空")
    fill_val = float(np.mean(roi_clean))

    img_in = arr_f.copy()
    img_in[~mask] = fill_val

    # 百分位拉伸
    vmin, vmax = np.percentile(roi_clean, [2, 98])
    arr_norm = np.clip((img_in - vmin) / (vmax - vmin + 1e-8), 0, 1)

    # 滤波
    if method == "bilateral":
        arr_u8 = (arr_norm * 255).astype(np.uint8)
        out_u8 = cv2.bilateralFilter(arr_u8, d=7, sigmaColor=8, sigmaSpace=7)
        out = out_u8.astype(np.float32) / 255.0
    elif method == "gaussian":
        out = cv2.GaussianBlur(arr_norm, (5, 5), 0)
    elif method == "median":
        out = cv2.medianBlur((arr_norm * 255).astype(np.uint8), 5)
        out = out.astype(np.float32) / 255.0
    elif method == "nlmeans":
        arr_u8 = (arr_norm * 255).astype(np.uint8)
        out = cv2.fastNlMeansDenoising(arr_u8, None, h=10,
                                        templateWindowSize=7,
                                        searchWindowSize=21)
        out = out.astype(np.float32) / 255.0
    else:
        raise ValueError(f"不支持的滤波方法: {method}")

    # 反拉伸
    res = out * (vmax - vmin) + vmin
    return res.astype(np.float32)


def Implement4(m2_path="output/m2_optical_refl.tif",
               m3_mask_path="output/m3_valid_mask.npy"):
    """滤波主流程"""
    if not os.path.exists(m2_path):
        raise FileNotFoundError(f"输入影像不存在: {m2_path}")

    inp = read_band(m2_path)
    valid = np.load(m3_mask_path)
    print("[输入] " + describe(inp, "表观反射率"))
    print(f"[输入] 掩膜有效率={valid.mean()*100:.2f}%")

    # 腐蚀掩膜：滤波计算时用（避免边界污染）
    kernel = np.ones((5, 5), np.uint8)
    valid_eroded = cv2.erode(valid.astype(np.uint8), kernel, iterations=1).astype(bool)
    np.save("output/m4_eroded_mask.npy", valid_eroded)

    # 滤波（只在腐蚀掩膜内计算）
    filtered_eroded = filter_optical(inp["arr"], method="bilateral",
                                      valid_mask=valid_eroded)


    # 腐蚀掩膜内的像素用滤波值，边界像素保留原值
    filtered = inp["arr"].copy()
    filtered[valid_eroded] = filtered_eroded[valid_eroded]
    filtered[~valid] = NODATA

    out_path = "output/m4_optical_filtered.tif"
    write_band(out_path, filtered, inp["profile"], nodata=NODATA)

    out = read_band(out_path)
    print("[输出] " + describe(out, "滤波后"))

    return out_path, "output/m4_eroded_mask.npy"


if __name__ == "__main__":
    Implement4()