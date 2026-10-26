import os
import numpy as np
import rasterio
from rasterio.windows import Window
from common import *


def logarithmic_stretch(sar_tif, optical_tif, out_tif,
                       sar_mask_tif=None,
                       opt_mask_tif=None,
                       sar_data_type="linear_intensity",
                       eps=1e-10,
                       low_pct=2.0,
                       high_pct=98.0,
                       chunk_rows=2048,
                       sample_max=5_000_000,
                       raise_on_low_sample=True):
    """
    对数变换 + 分位数线性拉伸，适配星载/机载/无人机SAR
    sar_data_type 支持:
        "linear_intensity": 线性强度 σ⁰，K=10 (10*log10)
        "linear_amplitude": 线性幅度，K=20 (20*log10)
        "db": SAR已经是dB域，跳过log10变换，直接做分位数拉伸
    参数:
        sar_tif:        SAR对齐后影像
        optical_tif:    光学图（仅读取尺寸，不参与映射）
        out_tif:        输出路径
        sar_mask_tif:   共同掩模 common_mask.tif
        opt_mask_tif:   光学掩模，common_mask场景填None
        sar_data_type:  {"linear_intensity","linear_amplitude","db"}
        eps:            log10偏移，防止log(0)
        low_pct:        低分位数
        high_pct:       高分位数
        chunk_rows:     分块行数
        sample_max:     最大采样像素
        raise_on_low_sample: 样本不足是否抛异常
    返回 out_tif
    """
    with rasterio.open(sar_tif) as sar_src, \
            rasterio.open(optical_tif) as opt_src:
        assert sar_src.height == opt_src.height, "高度不匹配"
        assert sar_src.width == opt_src.width, "宽度不匹配"
        n_rows, n_cols = sar_src.height, sar_src.width
        sar_nodata = sar_src.nodata

        # 确定系数与是否做对数
        if sar_data_type == "linear_intensity":
            K = 10.0
            do_log = True
            log_info(f"  SAR输入：线性强度，K={K} 执行 10*log10()")
            unit_label = "dB"
        elif sar_data_type == "linear_amplitude":
            K = 20.0
            do_log = True
            log_info(f"  SAR输入：线性幅度，K={K} 执行 20*log10()")
            unit_label = "dB"
        elif sar_data_type == "db":
            K = 1.0
            do_log = False
            log_info(f"  SAR输入：已经dB域，跳过log10对数变换，直接分位数拉伸")
            unit_label = "dB"
        else:
            raise ValueError(f"不支持sar_data_type={sar_data_type}, "
                             f"可选 linear_intensity / linear_amplitude / db")

        log_info(f"  SAR 尺寸: {n_rows} × {n_cols}")
        log_info(f"  eps={eps:.2e}, 分位数: [{low_pct}%, {high_pct}%]")

        # ==========步骤1：分块采样统计分位数，掩膜只打开一次==========
        log_info(f"\n  [1/3] 统计变换后分位数...")
        total_pixels = n_rows * n_cols
        sample_stride = max(1, int(np.sqrt(total_pixels / sample_max)))
        log_info(f"    采样步长: {sample_stride}")
        trans_vals_list = []

        mask_src = None
        if sar_mask_tif is not None and os.path.exists(sar_mask_tif):
            mask_src = rasterio.open(sar_mask_tif)

        for r0 in range(0, n_rows, chunk_rows):
            r1 = min(r0 + chunk_rows, n_rows)
            win = Window(0, r0, n_cols, r1 - r0)
            sar_chunk = sar_src.read(1, window=win).astype(np.float64)

            valid = np.isfinite(sar_chunk)
            if sar_nodata is not None:
                valid &= (sar_chunk != sar_nodata)
            if do_log:
                valid &= (sar_chunk > 0)

            # 分块读取掩膜（文件只打开一次）
            if mask_src is not None:
                m_win = Window(0, r0, n_cols, r1 - r0)
                m_blk = mask_src.read(1, window=m_win)
                chunk_mask = (m_blk == 1)
                valid &= chunk_mask

            if valid.sum() == 0:
                continue
            # 采样
            sar_s = sar_chunk[::sample_stride, ::sample_stride]
            valid_s = valid[::sample_stride, ::sample_stride]
            if valid_s.sum() == 0:
                continue
            vals = sar_s[valid_s]

            if do_log:
                with np.errstate(divide='ignore', invalid='ignore'):
                    trans_vals = K * np.log10(vals + eps)
            else:
                trans_vals = vals
            trans_vals_list.append(trans_vals)

        if mask_src is not None:
            mask_src.close()

        if len(trans_vals_list) == 0:
            raise ValueError("无有效样本")
        all_trans_vals = np.concatenate(trans_vals_list)
        log_info(f"    有效样本: {len(all_trans_vals)}")
        min_sample_thresh = 100
        if len(all_trans_vals) < min_sample_thresh:
            msg = f"有效样本{len(all_trans_vals)} < {min_sample_thresh}"
            if raise_on_low_sample:
                raise ValueError(msg)
            else:
                log_info(f"【警告】{msg}，继续执行，结果不可靠！")

        low = float(np.percentile(all_trans_vals, low_pct))
        high = float(np.percentile(all_trans_vals, high_pct))
        log_info(f"    变换后范围: [{all_trans_vals.min():.2f}, {all_trans_vals.max():.2f}] {unit_label}")
        log_info(f"    分位数 [{low_pct}%, {high_pct}%]: [{low:.2f}, {high:.2f}] {unit_label}")
        if high - low < 1e-6:
            raise ValueError("分位数范围过小，无法拉伸")

        # ==========步骤2：分块处理输出==========
        log_info(f"\n  [2/3] 分块应用变换与拉伸...")
        profile = sar_src.profile.copy()
        profile.update(
            dtype='float32',
            count=1,
            nodata=sar_nodata,
            compress='lzw',
            tiled=True,
            blockxsize=512,
            blockysize=512,
        )
        n_chunks = (n_rows + chunk_rows - 1) // chunk_rows

        mask_src_write = None
        if sar_mask_tif is not None and os.path.exists(sar_mask_tif):
            mask_src_write = rasterio.open(sar_mask_tif)

        with rasterio.open(out_tif, 'w', **profile) as dst:
            for ci, r0 in enumerate(range(0, n_rows, chunk_rows)):
                r1 = min(r0 + chunk_rows, n_rows)
                win = Window(0, r0, n_cols, r1 - r0)
                sar_chunk = sar_src.read(1, window=win).astype(np.float64)
                valid = np.isfinite(sar_chunk)
                if sar_nodata is not None:
                    valid &= (sar_chunk != sar_nodata)
                if do_log:
                    valid &= (sar_chunk > 0)

                if mask_src_write is not None:
                    m_win = Window(0, r0, n_cols, r1 - r0)
                    m_blk = mask_src_write.read(1, window=m_win)
                    chunk_mask = (m_blk == 1)
                    valid &= chunk_mask

                result = np.full(sar_chunk.shape, sar_nodata, dtype=np.float32)
                if valid.any():
                    vals = sar_chunk[valid]
                    if do_log:
                        with np.errstate(divide='ignore', invalid='ignore'):
                            trans_vals = K * np.log10(vals + eps)
                    else:
                        trans_vals = vals
                    stretched = (trans_vals - low) / (high - low)
                    stretched = np.clip(stretched, 0.0, 1.0)
                    result[valid] = stretched.astype(np.float32)
                dst.write(result, 1, window=win)
                pct = (ci + 1) / n_chunks * 100
                log_info(f"      [{ci+1}/{n_chunks}] rows {r0}-{r1} ({pct:.1f}%)")

        if mask_src_write is not None:
            mask_src_write.close()

        # ==========步骤3：统计输出结果【修复：只打开一次输出文件】==========
        log_info(f"\n  [3/3] 输出统计...")
        total_valid = 0
        sum_val = 0.0
        sat_count = 0

        with rasterio.open(out_tif) as out_src:
            for r0 in range(0, n_rows, chunk_rows):
                r1 = min(r0 + chunk_rows, n_rows)
                win = Window(0, r0, n_cols, r1 - r0)
                out_chunk = out_src.read(1, window=win)
                v = out_chunk[np.isfinite(out_chunk) & (out_chunk != sar_nodata)]
                if v.size > 0:
                    total_valid += v.size
                    sum_val += v.sum()
                    sat_count += int((v >= 0.9999).sum())

        log_info(f"    有效像素: {total_valid}")
        log_info(f"    均值: {sum_val / max(total_valid, 1):.4f}")
        log_info(f"    饱和(≈1.0)像素: {sat_count} ({sat_count / max(total_valid,1) * 100:.4f}%)")

        log_info(f"\n  ✓ 对数拉伸完成: {out_tif}")
    return out_tif


if __name__ == "__main__":
    sar_tif = r"D:\Project_Pycharm\Study\Synthesize\output\sar_aligned.tif"
    optical_tif = r"D:\Project_Pycharm\Study\Synthesize\output\m6_optical_final.tif"
    out_tif = r"D:\Project_Pycharm\Study\Synthesize\output\sar_logarithmic.tif"
    common_mask = r"D:\Project_Pycharm\Study\Synthesize\output\common_mask.tif"

    # --------------------------调用示例--------------------------
    # 场景1：星载SNAP输出线性强度Sigma0（原有逻辑，兼容）
    logarithmic_stretch(
        sar_tif=sar_tif,
        optical_tif=optical_tif,
        out_tif=out_tif,
        sar_mask_tif=common_mask,
        opt_mask_tif=None,
        sar_data_type="linear_intensity",
        low_pct=2.0,
        high_pct=98.0,
    )
    # ============================================================
    # 4. 图像质量评价（新增）
    # ============================================================
    print(f"\n  [SAR图质量评价]")

    with rasterio.open(out_tif) as src:
        matched_arr = src.read(1).astype(np.float32)

    # 使用共同掩模
    mask_arr = None
    if common_mask is not None and os.path.exists(common_mask):
        with rasterio.open(common_mask) as msk_src:
            mask_arr = (msk_src.read(1) == 1)

    quality_metrics = evaluate_image_quality(matched_arr, mask_arr, nodata=-9999.0)

    for k, v in quality_metrics.items():
        if isinstance(v, float):
            print(f"    {k:30s} = {v:.4f}")
        else:
            print(f"    {k:30s} = {v}")

    print("\n全部完成！")
    print(f"输出对数变换+分位数拉伸SAR影像：{out_tif}")
