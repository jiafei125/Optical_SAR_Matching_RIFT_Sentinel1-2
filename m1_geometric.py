# m1_geometric.py
"""
模块1：几何校正（基础）
输入:  data/optical_ori.tif
输出:  data/m1_optical_geo.tif
       data/m1_valid_mask.npy

职责:
    1. 正射校正（可选，需DEM/RPC）
    2. 去旋转（保证图像正北栅格）
    3. 保持原始CRS/分辨率（重投影在M5做）

不职责:
    - 不做重投影（留给M5）
    - 不做AOI裁剪（留给M5）
"""
import os
import numpy as np
import rasterio
from rasterio.transform import Affine
from common import read_band, write_band, describe, NODATA, save_work_grid

def orthorectify(src, dst, dem=None, dst_crs=None):
    """正射校正（可选）"""
    try:
        from osgeo import gdal
        opts = gdal.WarpOptions(
            format="GTiff", dstSRS=dst_crs,
            rpc=True, transformerOptions=[f"RPC_DEM={dem}"] if dem else [],
            resampleAlg="bilinear", dstNodata=NODATA,
            creationOptions=["COMPRESS=LZW", "TILED=YES"])
        gdal.Warp(dst, src, options=opts)
    except Exception as e:
        raise RuntimeError(f"正射校正失败: {e}")


def remove_rotation(src, dst):
    """去旋转：将图像转为标准正北栅格（transform.b=d=0）"""
    with rasterio.open(src) as s:
        transform = s.transform
        # 检查是否有旋转
        if abs(transform.b) < 1e-9 and abs(transform.d) < 1e-9:
            # 无旋转，直接拷贝
            arr = s.read(1).astype(np.float32)
            prof = s.profile.copy()
            prof.update(dtype="float32", count=1, nodata=NODATA)
            with rasterio.open(dst, "w", **prof) as d:
                d.write(arr, 1)
            return False  # 无旋转

        # 有旋转，需要校正
        from rasterio.warp import reproject, Resampling, calculate_default_transform
        # 计算无旋转的 transform（保持左上角，去掉旋转项）
        new_transform = Affine(transform.a, 0, transform.c,
                               0, transform.e, transform.f)
        prof = s.profile.copy()
        prof.update(transform=new_transform, dtype="float32",
                    count=1, nodata=NODATA)
        with rasterio.open(dst, "w", **prof) as d:
            reproject(source=rasterio.band(s, 1),
                      destination=rasterio.band(d, 1),
                      src_transform=transform, src_crs=s.crs,
                      dst_transform=new_transform, dst_crs=s.crs,
                      src_nodata=s.nodata, dst_nodata=NODATA,
                      resampling=Resampling.bilinear)
        return True


def Implement1(data_path="data/optical_v3.tif", DO_ORTHO=False):
    """
    基础几何校正
    - 不做重投影（M5统一处理）
    - 不做AOI裁剪（M5统一处理）
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"输入影像不存在: {data_path}")

    os.makedirs("data", exist_ok=True)
    inp = read_band(data_path)
    print("[输入] " + describe(inp, "DN"))

    # 1.1 正射校正（可选）
    if DO_ORTHO:
        if not os.path.exists("data/dem_v2.tif"):
            raise FileNotFoundError("DEM文件不存在，无法正射校正")
        ortho = "output/m1_optical_ortho.tif"
        orthorectify(data_path, ortho, dem="data/dem_v2.tif")
        src = ortho
    else:
        src = data_path

    # 1.2 去旋转
    out_path = "output/m1_optical_geo.tif"
    had_rotation = remove_rotation(src, out_path)
    print(f"[去旋转] {'执行了去旋转' if had_rotation else '无旋转，跳过'}")

    # 1.3 输出掩膜
    out = read_band(out_path)
    print("[输出] " + describe(out, "DN"))

    valid_mask = (out["arr"] != NODATA) & np.isfinite(out["arr"])
    mask_path = "output/m1_valid_mask.npy"
    np.save(mask_path, valid_mask)



    save_work_grid(out["profile"], "output/work_grid.json")
    print(f"[保存工作网格] output/work_grid.json")

    return out_path, mask_path


if __name__ == "__main__":
    Implement1()