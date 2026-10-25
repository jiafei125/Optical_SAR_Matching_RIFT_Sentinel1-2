# m3_mask.py
"""
模块3：掩膜 + 云检测
输入:  data/m2_optical_refl.tif + data/m1_valid_mask.npy
输出:  data/m3_valid_mask.npy
       data/m3_valid_mask.tif

云检测策略（单波段场景）:
    1. 亮度阈值：反射率 > 云亮度阈值
    2. OTSU 自适应阈值：自动找到亮/暗分界
    3. 形态学开闭：去除小噪声，填充云内部空洞
    4. 阴影检测（可选）：低于周围均值的暗区
"""
import os
import numpy as np
import cv2
from common import read_band, write_band, describe, NODATA


def detect_cloud_otsu(arr, base_mask,
                      min_cloud_threshold=0.35,
                      morph_kernel_size=5):
    """
    基于 OTSU + 形态学 的云检测

    参数:
        arr: 反射率数组
        base_mask: M1 基础有效掩膜
        min_cloud_threshold: 云的最低亮度（避免把暗区误判为云）
        morph_kernel_size: 形态学核大小
    """
    cloud_mask = np.zeros_like(arr, dtype=bool)

    # 只在有效区域做检测
    roi = arr[base_mask]
    if roi.size == 0:
        return cloud_mask

    # 1. 归一化到 [0,255]
    vmin, vmax = np.percentile(roi, [2, 98])
    if vmax - vmin < 1e-8:
        return cloud_mask
    roi_norm = np.clip((roi - vmin) / (vmax - vmin), 0, 1) * 255
    roi_norm = roi_norm.astype(np.uint8)

    # 2. OTSU 自适应阈值
    otsu_thresh, _ = cv2.threshold(
        roi_norm, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 3. 双重条件：OTSU阈值 + 最低亮度约束
    min_thresh_norm = (min_cloud_threshold - vmin) / (vmax - vmin) * 255
    final_thresh = max(otsu_thresh, min_thresh_norm)

    # 4. 亮度检测
    cloud_roi = roi_norm > final_thresh

    # 5. 形态学处理（在 ROI 上）
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )

    # 将 ROI 展开成与 arr 同形状
    cloud_full = np.zeros_like(arr, dtype=np.uint8)
    cloud_full[base_mask] = cloud_roi.astype(np.uint8) * 255

    # 开运算：去小噪点
    cloud_open = cv2.morphologyEx(cloud_full, cv2.MORPH_OPEN, kernel)
    # 闭运算：填充云内部空洞
    cloud_close = cv2.morphologyEx(cloud_open, cv2.MORPH_CLOSE, kernel)

    cloud_mask = cloud_close > 127

    print(f"[云检测] OTSU阈值={otsu_thresh:.1f}, 最终阈值={final_thresh:.1f}, "
          f"云像素占比={cloud_mask.mean()*100:.2f}%")

    return cloud_mask


def detect_shadow(arr, base_mask, cloud_mask,
                  shadow_ratio=0.6,
                  morph_kernel_size=3):
    """
    阴影检测（可选）：云的邻域内暗区
    简化策略：低于局部均值的暗区，且靠近云
    """
    if cloud_mask.sum() == 0:
        return np.zeros_like(arr, dtype=bool)

    # 云膨胀，得到云的邻域
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size * 3, morph_kernel_size * 3)
    )
    cloud_dilated = cv2.dilate(cloud_mask.astype(np.uint8), kernel, iterations=2) > 0

    # 暗区：低于全局均值的 shadow_ratio 倍
    valid_vals = arr[base_mask]
    global_mean = np.mean(valid_vals)
    dark_mask = (arr < global_mean * shadow_ratio) & base_mask

    # 阴影 = 暗区 ∩ 云邻域
    shadow_mask = dark_mask & cloud_dilated & (~cloud_mask)

    print(f"[阴影检测] 阴影像素占比={shadow_mask.mean()*100:.2f}%")

    return shadow_mask


def build_mask(arr, base_mask, nodata=NODATA,
               cloud_threshold=0.35,
               use_shadow_detect=False):
    """综合掩膜构建"""
    valid = base_mask.copy()

    # 1. 剔除 NaN/Inf
    valid &= np.isfinite(arr)

    # 2. 剔除 NODATA
    if nodata is not None:
        valid &= (arr != nodata)

    # 3. 反射率合理范围（放宽，云可能>1）
    valid &= (arr >= 0.0)

    # 4. 云检测
    cloud_mask = detect_cloud_otsu(arr, valid,
                                    min_cloud_threshold=cloud_threshold)
    valid &= (~cloud_mask)

    # 5. 阴影检测（可选）
    if use_shadow_detect:
        shadow_mask = detect_shadow(arr, base_mask, cloud_mask)
        valid &= (~shadow_mask)

    return valid


def Implement3(m2_path="output/m2_optical_refl.tif",
               m1_mask_path="output/m1_valid_mask.npy",
               cloud_threshold=0.35,
               use_shadow_detect=False):
    """掩膜 + 云检测"""
    if not os.path.exists(m2_path):
        raise FileNotFoundError(f"输入影像不存在: {m2_path}")

    inp = read_band(m2_path)
    print("[输入] " + describe(inp, "表观反射率"))

    base_mask = np.load(m1_mask_path)
    if base_mask.shape != inp["arr"].shape:
        raise ValueError(f"掩膜尺寸不匹配")

    valid = build_mask(
        inp["arr"], base_mask,
        nodata=inp["nodata"],
        cloud_threshold=cloud_threshold,
        use_shadow_detect=use_shadow_detect
    )

    # 输出 npy
    np.save("output/m3_valid_mask.npy", valid)

    # 输出 GeoTIFF
    mask_uint8 = valid.astype(np.uint8)
    write_band("output/m3_valid_mask.tif", mask_uint8,
               inp["profile"], nodata=0)

    print(f"[输出] 有效率={valid.mean()*100:.2f}%")

    return "output/m3_valid_mask.tif", "output/m3_valid_mask.npy", valid


if __name__ == "__main__":
    Implement3()