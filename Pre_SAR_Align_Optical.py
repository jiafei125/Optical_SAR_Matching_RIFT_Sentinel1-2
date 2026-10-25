"""
模块7: 投影对齐到光学基准图

输入: sar_geocoded.tif  (SNAP 输出的 SAR)
      sar_mask.tif      (SNAP 输出的掩膜)
      optical_ori.tif       (光学基准图)

输出: sar_aligned.tif       (与 optical_ori.tif 完全同网格)
      sar_aligned_mask.tif  (更新后的掩膜，与 optical_ori.tif 同网格)

说明:
    - SNAP 输出的 SAR 可能是 EPSG:4326 或 UTM
    - 光学图是 EPSG:32651 (UTM 51N), 10 m
    - SAR 影像用双线性，掩膜用最近邻
    - SAR 多出光学的部分自动裁剪，不足的部分填 0
    - 使用 GDAL Warp, 流式处理, 无内存问题
"""

from osgeo import gdal
import numpy as np
import os
import rasterio
from common import log_info, evaluate_image_quality


gdal.UseExceptions()


# ============================================================
# 核心函数：把任意栅格对齐到光学网格
# ============================================================
def align_to_optical(src_tif, optical_tif, out_tif,
                     sar_mask_tif=None,
                     out_mask_tif=None,
                     resample_method='bilinear',
                     src_nodata=None,
                     dst_nodata=-9999.0,
                     mask_nodata=0,
                     mask_valid_value=None):
    """
    把任意栅格（SAR影像、DEM、斜距等）重投影到光学图的精确网格。

    参数:
        src_tif:          输入栅格路径（SAR影像、DEM、斜距等）
        optical_tif:      光学基准图（网格目标）
        out_tif:          输出栅格路径
        sar_mask_tif:     源掩膜路径（可选，仅SAR影像需要）
        out_mask_tif:     输出掩膜路径（可选，与 sar_mask_tif 配套）
        resample_method:  重采样方法（nearest/bilinear/cubic/lanczos）
        src_nodata:       源 NoData 值（None=自动读取元数据）
        dst_nodata:       输出 NoData 值
        mask_nodata:      掩膜输出 NoData（固定 0）
        mask_valid_value: 掩膜有效值（None=自动判断）

    返回:
        out_tif（无掩膜）或 (out_tif, out_mask_tif)（有掩膜）
    """
    log_info(f"投影对齐: {os.path.basename(src_tif)} → 光学网格")

    # ============================================================
    # 1. 读取光学图网格
    # ============================================================
    opt = gdal.Open(optical_tif)
    if opt is None:
        raise FileNotFoundError(f"无法打开光学图: {optical_tif}")

    opt_gt = opt.GetGeoTransform()
    opt_proj = opt.GetProjection()
    opt_w = opt.RasterXSize
    opt_h = opt.RasterYSize

    opt_x_min = opt_gt[0]
    opt_y_max = opt_gt[3]
    opt_x_max = opt_x_min + opt_w * opt_gt[1]
    opt_y_min = opt_y_max + opt_h * opt_gt[5]

    log_info(f"  光学图网格: {opt_w} × {opt_h}, "
          f"分辨率 ({opt_gt[1]:.3f}, {abs(opt_gt[5]):.3f}) m")

    # ============================================================
    # 2. 打开源数据，确定 src_nodata
    # ============================================================
    src = gdal.Open(src_tif)
    if src is None:
        raise FileNotFoundError(f"无法打开源数据: {src_tif}")

    src_band = src.GetRasterBand(1)
    if src_nodata is None:
        src_nodata = src_band.GetNoDataValue()
        if src_nodata is None:
            log_info(f"  源数据元数据未定义 NoData，将不启用 srcNodata 过滤")
        else:
            log_info(f"  自动读取 src_nodata = {src_nodata}")
    else:
        log_info(f"  使用指定 src_nodata = {src_nodata}")

    log_info(f"  源数据: {src.RasterXSize} × {src.RasterYSize}, "
          f"波段数 {src.RasterCount}")

    # ============================================================
    # 3. 重采样方法
    # ============================================================
    method_map = {
        'nearest':  gdal.GRA_NearestNeighbour,
        'bilinear': gdal.GRA_Bilinear,
        'cubic':    gdal.GRA_Cubic,
        'lanczos':  gdal.GRA_Lanczos,
        'average':  gdal.GRA_Average,
    }
    alg = method_map.get(resample_method, gdal.GRA_Bilinear)

    # ============================================================
    # 4. 公共 WarpOptions
    # ============================================================
    common_opts = dict(
        format='GTiff',
        dstSRS=opt_proj,
        width=opt_w,
        height=opt_h,
        outputBounds=[opt_x_min, opt_y_min, opt_x_max, opt_y_max],
        multithread=True,
        warpMemoryLimit=512,
        creationOptions=[
            'TILED=YES',
            'COMPRESS=LZW',
            'BIGTIFF=IF_SAFER',
            'BLOCKXSIZE=512',
            'BLOCKYSIZE=512',
        ]
    )

    # ============================================================
    # 5. Warp 源影像（SAR 图 / DEM / 斜距，统一走这里）
    # ============================================================
    warp_options = gdal.WarpOptions(
        srcNodata=src_nodata,
        resampleAlg=alg,
        dstNodata=dst_nodata,
        **common_opts
    )
    gdal.Warp(out_tif, src, options=warp_options)
    log_info(f"  ✓ 影像对齐完成: {out_tif}")

    # ============================================================
    # 6. Warp 掩膜（仅当同时提供 sar_mask_tif 和 out_mask_tif）
    # ============================================================
    if sar_mask_tif is not None and out_mask_tif is not None:
        log_info(f"\n  处理掩膜...")

        mask = gdal.Open(sar_mask_tif)
        if mask is None:
            raise FileNotFoundError(f"无法打开掩膜: {sar_mask_tif}")

        mask_array = mask.GetRasterBand(1).ReadAsArray()
        mask_src_nodata = mask.GetRasterBand(1).GetNoDataValue()

        # ---- 6.1 剔除 NaN/Inf ----
        mask_clean = mask_array.copy().astype(np.float64)
        invalid = ~np.isfinite(mask_clean)
        if invalid.sum() > 0:
            log_info(f"    剔除 NaN/Inf 像素: {invalid.sum()}")
            mask_clean[invalid] = 0

        # ---- 6.2 剔除源 nodata ----
        if mask_src_nodata is not None and np.isfinite(mask_src_nodata):
            nodata_count = int((mask_clean == mask_src_nodata).sum())
            if nodata_count > 0:
                log_info(f"    剔除源 nodata({mask_src_nodata}) 像素: {nodata_count}")
                mask_clean[mask_clean == mask_src_nodata] = 0

        # ---- 6.3 判断有效值 ----
        unique_vals = np.unique(mask_clean)
        if mask_valid_value is None:
            non_zero_vals = [v for v in unique_vals if v != 0]
            if len(non_zero_vals) == 0:
                raise ValueError("掩膜全为 0，无有效区域")
            if len(non_zero_vals) == 1:
                mask_valid_value = float(non_zero_vals[0])
            else:
                counts = {float(v): int((mask_clean == v).sum())
                          for v in non_zero_vals}
                mask_valid_value = max(counts, key=counts.get)
            log_info(f"    自动判断有效值: {mask_valid_value}")

        # ---- 6.4 二值化到 0/1 ----
        mask_binary = (mask_clean == mask_valid_value).astype(np.uint8)
        log_info(f"    二值化后有效: {mask_binary.sum()}/{mask_binary.size} "
              f"({mask_binary.sum() / mask_binary.size * 100:.2f}%)")

        # ---- 6.5 存临时 GTiff ----
        tmp_mask_tif = out_mask_tif.replace('.tif', '_tmp.tif')
        driver = gdal.GetDriverByName('GTiff')
        tmp_ds = driver.Create(
            tmp_mask_tif,
            mask.RasterXSize, mask.RasterYSize,
            1, gdal.GDT_Byte,
            options=['COMPRESS=LZW']
        )
        tmp_ds.SetGeoTransform(mask.GetGeoTransform())
        tmp_ds.SetProjection(mask.GetProjection())
        tmp_ds.GetRasterBand(1).WriteArray(mask_binary)
        tmp_ds.GetRasterBand(1).SetNoDataValue(0)
        tmp_ds = None

        # ---- 6.6 Warp 掩膜 ----
        mask_warp_options = gdal.WarpOptions(
            resampleAlg=gdal.GRA_NearestNeighbour,
            dstNodata=mask_nodata,
            **common_opts
        )
        gdal.Warp(out_mask_tif, tmp_mask_tif, options=mask_warp_options)

        try:
            os.remove(tmp_mask_tif)
        except OSError:
            pass

        # ---- 6.7 与影像有效区取交集 ----
        _update_mask_with_image_valid(out_mask_tif, out_tif, dst_nodata)
        log_info(f"  ✓ 掩膜对齐完成: {out_mask_tif}")

    # ============================================================
    # 7. 验证输出
    # ============================================================
    with gdal.Open(out_tif) as aligned:
        assert aligned.RasterXSize == opt_w, "输出宽度不匹配"
        assert aligned.RasterYSize == opt_h, "输出高度不匹配"
        from osgeo import osr
        srs_aligned = osr.SpatialReference(wkt=aligned.GetProjection())
        srs_opt = osr.SpatialReference(wkt=opt_proj)
        assert srs_aligned.IsSame(srs_opt), "输出 CRS 不匹配"

    if out_mask_tif is not None and os.path.exists(out_mask_tif):
        with gdal.Open(out_mask_tif) as aligned_mask:
            assert aligned_mask.RasterXSize == opt_w, "掩膜宽度不匹配"
            assert aligned_mask.RasterYSize == opt_h, "掩膜高度不匹配"

    # ============================================================
    # 8. 返回
    # ============================================================
    if sar_mask_tif is not None and out_mask_tif is not None:
        return out_tif, out_mask_tif
    return out_tif

# ============================================================
# 辅助函数：掩膜与 SAR 影像有效区取交集
# ============================================================
def _update_mask_with_image_valid(mask_tif, image_tif, image_nodata):
    """
    更新掩膜：把 SAR 影像中为 NODATA 的位置，掩膜也置 0

    这一步确保：
        mask = 1  ⟺  SAR 影像有效
    """
    log_info(f"\n    更新掩膜：与 SAR 影像有效区取交集...")

    ds_img = gdal.Open(image_tif, gdal.GA_ReadOnly)
    ds_msk = gdal.Open(mask_tif, gdal.GA_Update)

    img_band = ds_img.GetRasterBand(1)
    msk_band = ds_msk.GetRasterBand(1)

    W, H = ds_img.RasterXSize, ds_img.RasterYSize

    # 逐块处理，避免内存问题
    block_size = 2048
    total_removed = 0

    for y in range(0, H, block_size):
        h = min(block_size, H - y)
        for x in range(0, W, block_size):
            w = min(block_size, W - x)

            # 读 SAR 影像块
            img_block = img_band.ReadAsArray(x, y, w, h)

            # 读掩膜块
            msk_block = msk_band.ReadAsArray(x, y, w, h)

            # 掩膜有效 且 影像非 NODATA
            if image_nodata is not None:
                valid = (
                    (img_block != image_nodata) &
                    np.isfinite(img_block) &
                    (msk_block == 1)
                )
            else:
                valid = np.isfinite(img_block) & (msk_block == 1)

            # 更新掩膜
            new_msk = valid.astype(np.uint8)
            removed = int((msk_block == 1).sum() - new_msk.sum())
            total_removed += removed

            msk_band.WriteArray(new_msk, x, y)

    ds_msk = None  # 刷新
    ds_img = None

    # print(f"      过滤掉的无效像素: {total_removed}")


def combine_common_mask(sar_aligned_mask_tif, optical_mask_tif, out_common_mask_tif):
    """
    生成共同掩膜：common_mask = sar_mask & optical_mask
    sar_mask=1：投影后的SAR有效像素
    optical_mask=1：光学图有效像素（去云、无nodata）
    输出common_mask.tif: 1=两者都有效，0=任意一方无效
    """

    log_info("合并掩模：SAR掩模 & 光学掩模，生成共同有效区域掩模")

    ds_sar_msk = gdal.Open(sar_aligned_mask_tif, gdal.GA_ReadOnly)
    ds_opt_msk = gdal.Open(optical_mask_tif, gdal.GA_ReadOnly)
    if ds_sar_msk is None:
        raise FileNotFoundError(f"找不到sar对齐掩模：{sar_aligned_mask_tif}")
    if ds_opt_msk is None:
        raise FileNotFoundError(f"找不到光学掩模：{optical_mask_tif}")

    # 校验尺寸投影是否一致（光学掩模必须和optical.tif同网格）
    w, h = ds_sar_msk.RasterXSize, ds_sar_msk.RasterYSize
    assert ds_opt_msk.RasterXSize == w and ds_opt_msk.RasterYSize == h, "光学掩模尺寸必须和光学基准图一致！"
    assert ds_sar_msk.GetProjection() == ds_opt_msk.GetProjection(), "CRS不一致，无法合并掩模"

    # 创建输出共同掩模
    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.Create(
        out_common_mask_tif, w, h, 1, gdal.GDT_Byte,
        options=['TILED=YES', 'COMPRESS=LZW', 'BIGTIFF=IF_SAFER']
    )
    ds_out.SetGeoTransform(ds_sar_msk.GetGeoTransform())
    ds_out.SetProjection(ds_sar_msk.GetProjection())
    out_band = ds_out.GetRasterBand(1)
    out_band.SetNoDataValue(0)

    block_size = 2048
    total_valid = 0
    # 分块读取合并，避免内存溢出
    for y in range(0, h, block_size):
        bh = min(block_size, h - y)
        for x in range(0, w, block_size):
            bw = min(block_size, w - x)
            sar_block = ds_sar_msk.GetRasterBand(1).ReadAsArray(x, y, bw, bh)
            opt_block = ds_opt_msk.GetRasterBand(1).ReadAsArray(x, y, bw, bh)
            # 二值按位与：两者都等于1才保留为1
            common_block = ((sar_block == 1) & (opt_block == 1)).astype(np.uint8)
            total_valid += int(common_block.sum())
            out_band.WriteArray(common_block, x, y)

    ds_sar_msk = None
    ds_opt_msk = None
    ds_out = None
    log_info(f"共同掩模生成完成：{out_common_mask_tif}")
    log_info(f"共同有效像素总数：{total_valid}")
    return out_common_mask_tif


# ============================================================
# 兼容旧接口
# ============================================================
def align_to_grid(sar_tif, grid_tif, out_tif, resample_method='bilinear'):
    """通用接口: 对齐到任意参考网格（仅影像）"""
    return align_to_optical(sar_tif, grid_tif, out_tif,
                            resample_method=resample_method) # <--- 必须改成关键字传参

