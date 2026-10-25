# m2_radiometric.py
import numpy as np
from common import *
import os

def convert_to_reflectance(arr, valid_mask, product_type,
                           scale_factor=10000.0, offset=0.0,
                           gain=1.0, esun=1512.06,
                           sun_zenith_deg=45.0,
                           earth_sun_distance=1.0,
                           nodata=NODATA):
    """
    通用 DN → 反射率转换

    支持的产品类型：
        - 'L2A_REFLECTANCE': DN 已经是反射率 × scale（如 Sentinel-2 L2A、MODIS）
        - 'L1C_REFLECTANCE': DN 是大气顶反射率 × scale（如 Sentinel-2 L1C）
        - 'RADIANCE': DN 是辐射亮度（如 Landsat L1）
        - 'DN_WITH_GAIN': DN 需要 gain/offset 定标（如 GF-1、SPOT）
        - 'ALREADY_REFLECTANCE': 已经是反射率 [0,1]（如无人机、处理后的产品）
        - 'SKIP': 跳过，直接返回原值（如 RGB）
    """
    mask = (~valid_mask) | (arr == nodata) | (~np.isfinite(arr))
    out = np.full_like(arr, nodata, dtype=np.float32)
    arr_valid = arr[~mask]

    if arr_valid.size == 0:
        raise RuntimeError("有效像素为空")

    # ---------- 按产品类型分发 ----------
    if product_type == "L2A_REFLECTANCE":
        # Sentinel-2 L2A, MODIS 等
        # DN → 地表反射率
        refl = arr_valid / scale_factor

    elif product_type == "L1C_REFLECTANCE":
        # Sentinel-2 L1C
        # DN → 大气顶反射率
        refl = arr_valid / scale_factor

    elif product_type == "RADIANCE":
        # Landsat L1 等
        # DN → 辐射亮度 → 反射率
        radiance = gain * arr_valid + offset
        cos_zenith = np.cos(np.deg2rad(sun_zenith_deg))
        refl = (np.pi * radiance * earth_sun_distance ** 2
                / (esun * cos_zenith + 1e-8))

    elif product_type == "DN_WITH_GAIN":
        # GF-1, SPOT 等，DN 需要 gain/offset 定标
        # 有些产品直接给反射率定标系数
        refl = gain * arr_valid + offset

    elif product_type == "ALREADY_REFLECTANCE":
        # 已经是反射率
        refl = arr_valid

    elif product_type == "SKIP":
        # 跳过辐射校正
        return arr.astype(np.float32)

    else:
        raise ValueError(f"未知产品类型: {product_type}")

    # 物理裁剪（放宽到 [0, 2]，允许云>1）
    refl = np.clip(refl, 0.0, 2.0)
    out[~mask] = refl
    return out.astype(np.float32)


def Implement2(m1_path="output/m1_optical_geo.tif",
               mask_path="output/m1_valid_mask.npy",
               product_type="L2A_REFLECTANCE",  # 【关键参数】
               scale_factor=10000.0,
               **kwargs):
    """
    M2 主流程

    参数:
        product_type: 见 convert_to_reflectance 说明
        scale_factor: 缩放系数（Sentinel-2 为 10000）
        **kwargs:     gain, offset, esun, sun_zenith_deg 等
    """
    if not os.path.exists(m1_path):
        raise FileNotFoundError(f"输入影像不存在: {m1_path}")

    inp = read_band(m1_path)
    print("[输入] " + describe(inp, "DN"))
    print(f"[辐射校正] 产品类型={product_type}, scale_factor={scale_factor}")

    valid_mask = np.load(mask_path)
    if valid_mask.shape != inp["arr"].shape:
        raise ValueError("掩膜尺寸不匹配")

    refl = convert_to_reflectance(
        inp["arr"], valid_mask,
        product_type=product_type,
        scale_factor=scale_factor,
        nodata=NODATA,
        **kwargs
    )

    out_path = "output/m2_optical_refl.tif"
    write_band(out_path, refl, inp["profile"], nodata=NODATA)
    print("[输出] " + describe(read_band(out_path), "反射率"))

    return out_path


if __name__ == "__main__":
    Implement2()