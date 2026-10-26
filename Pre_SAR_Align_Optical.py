"""
模块7: 栅格投影对齐到参考基准网格（通用版，支持星载SAR、机载SAR、无人机SAR、DEM等）
输入:
    src_tif        源栅格（SAR强度/幅度、DEM、其他栅格；支持SNAP星载、机载SAR）
    src_mask_tif   源掩膜(可选)
    ref_grid_tif   参考基准图（决定输出CRS、分辨率、尺寸、边界）
输出:
    out_tif        对齐后栅格，与ref_grid_tif完全同网格
    out_mask_tif   对齐后掩膜（可选输出）

说明:
    - 源数据CRS可以任意(4326/UTM/局部投影)，输出完全跟随参考图网格
    - 影像默认双线性；掩膜强制最近邻，保证二值
    - 源超出参考范围自动裁剪；源不足参考范围填充nodata
    - 全部分块处理，大影像不会爆内存；兼容星载SNAP‑SAR、机载SAR、无人机SAR
    mask_mode说明：
        "auto_snap": 自动识别掩膜有效值，适配SNAP输出掩膜（原有星载逻辑）
        "binary":    输入掩膜已经是0/1二值，直接使用（机载SAR常用）
        "manual":   用户手动指定mask_valid_value
"""
from osgeo import gdal, osr
import numpy as np
import os
from common import log_info, evaluate_image_quality

gdal.UseExceptions()


def align_to_reference(
    src_tif,
    ref_grid_tif,
    out_tif,
    src_mask_tif=None,
    out_mask_tif=None,
    mask_mode: str = "auto_snap",
    mask_valid_value=None,
    resample_method='bilinear',
    src_nodata=None,
    dst_nodata=-9999.0,
    mask_nodata=0,
    src_band_idx=1
):
    """
    【通用核心函数】把任意栅格重投影对齐到参考栅格网格
    参数:
        src_tif:          输入源栅格路径（SAR强度、DEM等）
        ref_grid_tif:     参考基准图，输出完全匹配此图网格
        out_tif:          输出对齐后栅格路径
        src_mask_tif:     源掩膜路径(可选)
        out_mask_tif:     输出对齐后掩膜路径(可选，与src_mask_tif成对)
        mask_mode:        {"auto_snap","binary","manual"}，掩膜解析模式
        mask_valid_value: 掩膜有效值；mask_mode=="manual"时必须传入
        resample_method:  重采样方法 nearest/bilinear/cubic/lanczos/average
        src_nodata:       源NoData；None自动从元数据读取
        dst_nodata:       输出栅格NoData
        mask_nodata:      输出掩膜NoData，固定0
        src_band_idx:     读取源栅格的波段编号，从1开始，适配多极化SAR
    返回:
        out_tif 或者 (out_tif, out_mask_tif)
    """
    log_info(f"投影对齐: {os.path.basename(src_tif)} → 参考网格 {os.path.basename(ref_grid_tif)}")

    # -------------------------- 1 读取参考网格信息 --------------------------
    ref_ds = gdal.Open(ref_grid_tif)
    if ref_ds is None:
        raise FileNotFoundError(f"无法打开参考基准图: {ref_grid_tif}")
    ref_gt = ref_ds.GetGeoTransform()
    ref_proj = ref_ds.GetProjection()
    ref_w = ref_ds.RasterXSize
    ref_h = ref_ds.RasterYSize

    ref_x_min = ref_gt[0]
    ref_y_max = ref_gt[3]
    ref_x_max = ref_x_min + ref_w * ref_gt[1]
    ref_y_min = ref_y_max + ref_h * ref_gt[5]

    log_info(
        f"  参考网格: {ref_w} × {ref_h}, 分辨率 ({ref_gt[1]:.3f}, {abs(ref_gt[5]):.3f}) m")

    # -------------------------- 2 打开源数据，检查复数、读取nodata --------------------------
    src_ds = gdal.Open(src_tif)
    if src_ds is None:
        raise FileNotFoundError(f"无法打开源数据: {src_tif}")

    src_band = src_ds.GetRasterBand(src_band_idx)
    dt = src_band.DataType
    if dt in (gdal.GDT_CInt16, gdal.GDT_CInt32, gdal.GDT_CFloat32, gdal.GDT_CFloat64):
        log_info("【警告】源为复数SAR栅格，本函数仅处理幅度/强度，请提前取实部/幅度！")

    if src_nodata is None:
        src_nodata = src_band.GetNoDataValue()
        if src_nodata is None:
            log_info(f"  源元数据未定义NoData，不启用srcNodata过滤")
        else:
            log_info(f"  自动读取src_nodata = {src_nodata}")
    else:
        log_info(f"  使用指定src_nodata = {src_nodata}")

    log_info(f"  源数据: {src_ds.RasterXSize} × {src_ds.RasterYSize}, 波段数 {src_ds.RasterCount}, 使用波段={src_band_idx}")

    # -------------------------- 3 重采样算法映射 --------------------------
    method_map = {
        'nearest': gdal.GRA_NearestNeighbour,
        'bilinear': gdal.GRA_Bilinear,
        'cubic': gdal.GRA_Cubic,
        'lanczos': gdal.GRA_Lanczos,
        'average': gdal.GRA_Average,
    }
    alg = method_map.get(resample_method, gdal.GRA_Bilinear)

    # -------------------------- 4 warp公共配置 --------------------------
    common_opts = dict(
        format='GTiff',
        dstSRS=ref_proj,
        width=ref_w,
        height=ref_h,
        outputBounds=[ref_x_min, ref_y_min, ref_x_max, ref_y_max],
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

    # -------------------------- 5 warp主影像 --------------------------
    warp_options = gdal.WarpOptions(
        srcNodata=src_nodata,
        resampleAlg=alg,
        dstNodata=dst_nodata,
        **common_opts
    )
    gdal.Warp(out_tif, src_ds, options=warp_options)
    log_info(f"  ✓ 影像对齐完成: {out_tif}")

    # -------------------------- 6 掩膜处理（分块，不再全图读内存） --------------------------
    if src_mask_tif is not None and out_mask_tif is not None:
        log_info(f"\n  处理源掩膜，mask_mode={mask_mode}...")
        mask_ds = gdal.Open(src_mask_tif)
        if mask_ds is None:
            raise FileNotFoundError(f"无法打开掩膜: {src_mask_tif}")
        mask_band = mask_ds.GetRasterBand(1)
        mask_src_nodata = mask_band.GetNoDataValue()

        tmp_mask_tif = out_mask_tif.replace('.tif', '_tmp.tif')
        driver = gdal.GetDriverByName('GTiff')
        tmp_ds = driver.Create(
            tmp_mask_tif,
            mask_ds.RasterXSize, mask_ds.RasterYSize,
            1, gdal.GDT_Byte,
            options=['COMPRESS=LZW', 'TILED=YES']
        )
        tmp_ds.SetGeoTransform(mask_ds.GetGeoTransform())
        tmp_ds.SetProjection(mask_ds.GetProjection())
        tmp_band = tmp_ds.GetRasterBand(1)
        tmp_band.SetNoDataValue(0)

        block = 2048
        h_total = mask_ds.RasterYSize
        w_total = mask_ds.RasterXSize

        # ----分块处理掩膜，不再ReadAsArray整张读取----
        for y in range(0, h_total, block):
            bh = min(block, h_total - y)
            for x in range(0, w_total, block):
                bw = min(block, w_total - x)
                arr_blk = mask_band.ReadAsArray(x, y, bw, bh).astype(np.float64)

                # 剔除nan inf
                invalid = ~np.isfinite(arr_blk)
                arr_blk[invalid] = 0
                # 剔除源掩膜nodata
                if mask_src_nodata is not None and np.isfinite(mask_src_nodata):
                    arr_blk[arr_blk == mask_src_nodata] = 0

                if mask_mode == "binary":
                    # 已经0‑1二值，直接clip
                    bin_blk = np.clip(arr_blk, 0, 1).astype(np.uint8)
                elif mask_mode == "manual":
                    if mask_valid_value is None:
                        raise ValueError("mask_mode='manual'必须传入mask_valid_value")
                    bin_blk = (arr_blk == mask_valid_value).astype(np.uint8)
                elif mask_mode == "auto_snap":
                    # SNAP模式：只在块内统计，不读全图
                    # 注意：auto_snap分块下不再做全局unique；如果你需要全局统计，需单独读取概览或采样
                    non_zero = arr_blk[arr_blk != 0]
                    if len(non_zero) > 0:
                        vals, cnts = np.unique(non_zero, return_counts=True)
                        best = vals[np.argmax(cnts)]
                        bin_blk = (arr_blk == best).astype(np.uint8)
                    else:
                        bin_blk = np.zeros_like(arr_blk, dtype=np.uint8)
                else:
                    raise ValueError(f"不支持mask_mode={mask_mode}, 可选 auto_snap / binary / manual")

                tmp_band.WriteArray(bin_blk, x, y)
        tmp_ds = None

        # warp临时二值掩膜
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

        # 掩膜和影像有效区取交集
        _update_mask_with_image_valid(out_mask_tif, out_tif, dst_nodata)
        log_info(f"  ✓ 掩膜对齐完成: {out_mask_tif}")

    # --------------------------7 输出校验 --------------------------
    with gdal.Open(out_tif) as aligned:
        assert aligned.RasterXSize == ref_w, "输出宽度与参考网格不匹配"
        assert aligned.RasterYSize == ref_h, "输出高度与参考网格不匹配"
        srs_aligned = osr.SpatialReference(wkt=aligned.GetProjection())
        srs_ref = osr.SpatialReference(wkt=ref_proj)
        assert srs_aligned.IsSame(srs_ref), "输出CRS与参考网格不匹配"

    if out_mask_tif is not None and os.path.exists(out_mask_tif):
        with gdal.Open(out_mask_tif) as aligned_mask:
            assert aligned_mask.RasterXSize == ref_w, "掩膜宽度不匹配参考网格"
            assert aligned_mask.RasterYSize == ref_h, "掩膜高度不匹配参考网格"

    if src_mask_tif is not None and out_mask_tif is not None:
        return out_tif, out_mask_tif
    return out_tif


def _update_mask_with_image_valid(mask_tif, image_tif, image_nodata):
    """
    更新掩膜：SAR/源影像为NODATA的位置，掩膜置0
    mask=1 等价于 源影像有效
    全部分块读写，无内存爆炸风险
    """
    log_info(f"\n    更新掩膜：与源影像有效区取交集...")
    ds_img = gdal.Open(image_tif, gdal.GA_ReadOnly)
    ds_msk = gdal.Open(mask_tif, gdal.GA_Update)

    img_band = ds_img.GetRasterBand(1)
    msk_band = ds_msk.GetRasterBand(1)
    W, H = ds_img.RasterXSize, ds_img.RasterYSize

    block_size = 2048
    total_removed = 0
    for y in range(0, H, block_size):
        h = min(block_size, H - y)
        for x in range(0, W, block_size):
            w = min(block_size, W - x)
            img_block = img_band.ReadAsArray(x, y, w, h)
            msk_block = msk_band.ReadAsArray(x, y, w, h)

            if image_nodata is not None:
                valid = (img_block != image_nodata) & np.isfinite(img_block) & (msk_block == 1)
            else:
                valid = np.isfinite(img_block) & (msk_block == 1)

            new_msk = valid.astype(np.uint8)
            removed = int((msk_block == 1).sum() - new_msk.sum())
            total_removed += removed
            msk_band.WriteArray(new_msk, x, y)

    ds_msk = None
    ds_img = None
    log_info(f"      本次过滤无效像素数: {total_removed}")


def combine_common_mask(src_aligned_mask_tif, ref_mask_tif, out_common_mask_tif):
    """
    生成共同掩膜：common_mask = src_mask & ref_mask
    src_mask=1：投影后源(SAR)有效像素
    ref_mask=1：参考图(光学)有效像素（去云、无nodata）
    输出：1=两者同时有效；0任意一方无效
    """
    log_info("合并掩模：源掩模 & 参考掩模，生成共同有效区域掩模")
    ds_src_msk = gdal.Open(src_aligned_mask_tif, gdal.GA_ReadOnly)
    ds_ref_msk = gdal.Open(ref_mask_tif, gdal.GA_ReadOnly)
    if ds_src_msk is None:
        raise FileNotFoundError(f"找不到对齐后源掩模：{src_aligned_mask_tif}")
    if ds_ref_msk is None:
        raise FileNotFoundError(f"找不到参考掩模：{ref_mask_tif}")

    w, h = ds_src_msk.RasterXSize, ds_src_msk.RasterYSize
    assert ds_ref_msk.RasterXSize == w and ds_ref_msk.RasterYSize == h, "掩模尺寸不一致！"
    assert ds_src_msk.GetProjection() == ds_ref_msk.GetProjection(), "CRS不一致，不能合并掩模"

    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.Create(
        out_common_mask_tif, w, h, 1, gdal.GDT_Byte,
        options=['TILED=YES', 'COMPRESS=LZW', 'BIGTIFF=IF_SAFER']
    )
    ds_out.SetGeoTransform(ds_src_msk.GetGeoTransform())
    ds_out.SetProjection(ds_src_msk.GetProjection())
    out_band = ds_out.GetRasterBand(1)
    out_band.SetNoDataValue(0)

    block_size = 2048
    total_valid = 0
    for y in range(0, h, block_size):
        bh = min(block_size, h - y)
        for x in range(0, w, block_size):
            bw = min(block_size, w - x)
            src_blk = ds_src_msk.GetRasterBand(1).ReadAsArray(x, y, bw, bh)
            ref_blk = ds_ref_msk.GetRasterBand(1).ReadAsArray(x, y, bw, bh)
            common_blk = ((src_blk == 1) & (ref_blk == 1)).astype(np.uint8)
            total_valid += int(common_blk.sum())
            out_band.WriteArray(common_blk, x, y)
    del ds_src_msk
    del ds_ref_msk
    del ds_out
    log_info(f"共同掩模完成：{out_common_mask_tif}, 共同有效像素总数：{total_valid}")
    return out_common_mask_tif


# --------------------------向下兼容旧接口（原有调用代码无需改动）--------------------------
def align_to_optical(src_tif, optical_tif, out_tif,
                     sar_mask_tif=None,
                     out_mask_tif=None,
                     resample_method='bilinear',
                     src_nodata=None,
                     dst_nodata=-9999.0,
                     mask_nodata=0,
                     mask_valid_value=None):
    """旧接口：对齐到光学基准图；完全复用通用实现，mask_mode默认auto_snap适配SNAP星载SAR"""
    return align_to_reference(
        src_tif=src_tif,
        ref_grid_tif=optical_tif,
        out_tif=out_tif,
        src_mask_tif=sar_mask_tif,
        out_mask_tif=out_mask_tif,
        mask_mode="auto_snap",
        mask_valid_value=mask_valid_value,
        resample_method=resample_method,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
        mask_nodata=mask_nodata,
        src_band_idx=1
    )


def align_to_grid(sar_tif, grid_tif, out_tif, resample_method='bilinear'):
    """旧兼容接口，仅影像对齐到任意参考网格"""
    return align_to_reference(
        src_tif=sar_tif,
        ref_grid_tif=grid_tif,
        out_tif=out_tif,
        resample_method=resample_method
    )
