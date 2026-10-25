# m6_normalize.py
"""
模块6：灰度归一化
输入:  data/m5_optical_resampled.tif + data/m5_valid_mask.npy
输出:  data/m6_optical_final.tif

修复:
    1. 下限改为 0.0（不再用0.05，避免误判真值0）
    2. NODATA保持-9999
"""
import os
import numpy as np
import cv2
from common import read_band, write_band, describe, NODATA, evaluate_image_quality


def normalize(arr, method="percentile", low=2, high=98, valid_mask=None):
    """归一化到 [0, 1]"""
    vals = arr[valid_mask]
    vals = vals[np.isfinite(vals) & (vals != NODATA)]
    if vals.size == 0:
        raise RuntimeError("有效像素为空")

    out_arr = np.full_like(arr, NODATA, dtype=np.float32)

    if method == "percentile":
        p_low, p_high = np.percentile(vals, [low, high])
        # 【修复】下限改为 0.0
        res = np.clip((arr[valid_mask] - p_low) / (p_high - p_low + 1e-8),
                      0.0, 1.0).astype(np.float32)
        out_arr[valid_mask] = res

    elif method == "mean_std":
        mu, sigma = np.mean(vals), np.std(vals)
        z = (arr[valid_mask] - mu) / (sigma + 1e-8)
        res = np.clip((z - z.min()) / (z.max() - z.min() + 1e-8),
                      0.0, 1.0).astype(np.float32)
        out_arr[valid_mask] = res

    elif method == "clahe":
        valid_data = arr[valid_mask]
        v8 = cv2.normalize(valid_data, None, 0, 255,
                           cv2.NORM_MINMAX).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v8_eq = clahe.apply(v8.reshape(-1, 1))
        res = v8_eq.ravel().astype(np.float32) / 255.0
        out_arr[valid_mask] = res

    else:
        raise ValueError(f"不支持的方法: {method}")

    return out_arr

def evaluate_optical_result(m6_path="output/m6_optical_final.tif",
                            mask_path="output/m5_valid_mask.npy"):
    """对归一化后的光学图进行质量评价"""
    # 读取图像
    inp = read_band(m6_path)
    arr = inp["arr"]  # float32 数组

    # 获取有效掩膜
    if os.path.exists(mask_path):
        mask = np.load(mask_path)
        if mask.shape != arr.shape:
            raise ValueError(f"掩膜尺寸不匹配: {mask.shape} vs {arr.shape}")
    else:
        # 若无掩膜文件，则根据 NoData 生成
        mask = (arr != NODATA) & np.isfinite(arr)

    # 调用评价函数
    metrics = evaluate_image_quality(arr, mask, nodata=NODATA)
    return metrics

def Implement6(m5_path="output/m5_optical_resampled.tif",
               m5_mask_path="output/m5_valid_mask.npy"):
    """归一化主流程"""
    if not os.path.exists(m5_path):
        raise FileNotFoundError(f"输入影像不存在: {m5_path}")

    inp = read_band(m5_path)
    print("[输入] " + describe(inp, "重采样后"))

    valid = np.load(m5_mask_path)
    if valid.shape != inp["arr"].shape:
        raise ValueError(f"掩膜尺寸不匹配: {valid.shape} vs {inp['arr'].shape}")

    norm = normalize(inp["arr"], method="percentile",
                     low=2, high=98, valid_mask=valid)

    out_path = "output/m6_optical_final.tif"
    write_band(out_path, norm, inp["profile"], nodata=NODATA)

    out = read_band(out_path)
    print("[输出] " + describe(out, "归一化后"))

    # ---- 新增：质量评价 ----
    valid_mask = np.load(m5_mask_path)  # 复用 m5 掩膜
    if valid_mask.shape != out["arr"].shape:
        raise ValueError(f"掩膜尺寸不匹配: {valid_mask.shape} vs {out['arr'].shape}")

    quality = evaluate_image_quality(out["arr"], valid_mask, nodata=NODATA)
    print("\n[光学图质量评价]")
    for k, v in quality.items():
        if isinstance(v, float):
            print(f"  {k:30s} = {v:.4f}")
        else:
            print(f"  {k:30s} = {v}")

    return out_path


if __name__ == "__main__":
    Implement6()