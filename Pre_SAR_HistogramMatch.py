"""
模块8: 直方图匹配
输入: sar_aligned.tif, optical_ori.tif
输出: sar_matched.tif
说明:
    - 大图分块处理, 避免内存溢出
    - 用累积分布函数 (CDF) 映射, 与 skimage.exposure.match_histograms 等价
    - 只对两图重叠且非 NoData 的像素统计
    ✅ 修改点：
        1. 支持传入预生成的common_mask共同掩模，无需同时传入SAR、光学独立掩模
        2. 【BUG修复】分块输出阶段增加掩模限制，采样区域和输出映射区域保持一致
        3. 保留兼容原有双掩模输入的能力（可选）
"""
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from common import *
from common import log_info
from scipy.stats import ks_2samp

# ============================================================
# 处理函数
# ============================================================
def histogram_match(sar_tif, optical_tif, out_tif,
                    sar_mask_tif=None,           # 推荐填入 common_mask.tif
                    opt_mask_tif=None,           # 使用common_mask时传None即可
                    bins=1024, chunk_rows=2048):
    """
    把 SAR 图像的直方图匹配到光学图

    参数:
        sar_tif:        SAR 图像路径 (已对齐)
        optical_tif:    光学图路径
        out_tif:        输出路径
        sar_mask_tif:   共同掩模 / SAR掩膜路径（强烈推荐，1=有效，0=无效）
        opt_mask_tif:   光学掩膜路径（使用共同掩模时填None）
        bins:           直方图分箱数
        chunk_rows:     分块处理的行数
    """

    # ---- 打开两图 ----
    with rasterio.open(sar_tif) as sar_src, \
            rasterio.open(optical_tif) as opt_src:

        assert sar_src.height == opt_src.height, "高度不匹配"
        assert sar_src.width == opt_src.width, "宽度不匹配"

        n_rows, n_cols = sar_src.height, sar_src.width
        sar_nodata = sar_src.nodata
        opt_nodata = opt_src.nodata

        # ============================================================
        # 【新增】加载掩膜
        # ============================================================
        log_info(f"\n  [0/3] 加载掩膜...")

        # 全局掩模数组，用于后面分块输出
        full_mask_bool = None

        # SAR 掩膜（优先放共同掩模）
        sar_mask_bool = None
        if sar_mask_tif is not None and os.path.exists(sar_mask_tif):
            with rasterio.open(sar_mask_tif) as src:
                sar_mask = src.read(1)
                # 判断掩膜有效值
                unique = np.unique(sar_mask)
                if 1 in unique:
                    sar_mask_bool = (sar_mask == 1)
                elif 255 in unique:
                    sar_mask_bool = (sar_mask == 255)
                else:
                    sar_mask_bool = (sar_mask > 0)
            full_mask_bool = sar_mask_bool  # 保存全局掩模用于输出阶段
            log_info(f"    SAR/共同掩膜: {sar_mask_bool.sum()}/{sar_mask_bool.size} "
                  f"({sar_mask_bool.sum() / sar_mask_bool.size * 100:.2f}%)")
        else:
            sar_mask_bool = None
            log_info(f"    SAR 掩膜: 未提供（只用 NODATA 过滤）")

        # 光学掩膜
        opt_mask_bool = None
        if opt_mask_tif is not None and os.path.exists(opt_mask_tif):
            with rasterio.open(opt_mask_tif) as src:
                opt_mask = src.read(1)
                unique = np.unique(opt_mask)
                if 1 in unique:
                    opt_mask_bool = (opt_mask == 1)
                elif 255 in unique:
                    opt_mask_bool = (opt_mask == 255)
                else:
                    opt_mask_bool = (opt_mask > 0)
            log_info(f"    光学掩膜: {opt_mask_bool.sum()}/{opt_mask_bool.size} "
                  f"({opt_mask_bool.sum() / opt_mask_bool.size * 100:.2f}%)")
        else:
            opt_mask_bool = None
            log_info(f"    光学掩膜: 未提供")

        # ============================================================
        # 步骤 1: 统计直方图（采样）
        # ============================================================
        log_info(f"\n  1.统计直方图...")

        target_samples = 5_000_000
        total_pixels = n_rows * n_cols
        sample_stride = max(1, int(np.sqrt(total_pixels / target_samples)))

        log_info(f"    采样步长: {sample_stride}")

        sar_vals = []
        opt_vals = []

        for r0 in range(0, n_rows, chunk_rows):
            r1 = min(r0 + chunk_rows, n_rows)
            win = Window(0, r0, n_cols, r1 - r0)
            sar_chunk = sar_src.read(1, window=win)
            opt_chunk = opt_src.read(1, window=win)

            # 采样
            sar_sample = sar_chunk[::sample_stride, ::sample_stride]
            opt_sample = opt_chunk[::sample_stride, ::sample_stride]

            # 【新增】对应的掩膜采样
            if sar_mask_bool is not None:
                sar_mask_sample = sar_mask_bool[r0:r1, :][::sample_stride, ::sample_stride]
            else:
                sar_mask_sample = None
            if opt_mask_bool is not None:
                sar_mask_sample = sar_mask_bool[r0:r1, :][::sample_stride, ::sample_stride]
            else:
                opt_mask_sample = None

            # 构建有效掩膜（NODATA + 用户掩膜）
            sar_valid = np.isfinite(sar_sample)
            if sar_nodata is not None:
                sar_valid &= (sar_sample != sar_nodata)
            if sar_mask_sample is not None:
                # 对齐形状
                if sar_mask_sample.shape == sar_sample.shape:
                    sar_valid &= sar_mask_sample

            opt_valid = np.isfinite(opt_sample)
            if opt_nodata is not None:
                opt_valid &= (opt_sample != opt_nodata)
            if opt_mask_sample is not None:
                if opt_mask_sample.shape == opt_sample.shape:
                    opt_valid &= opt_mask_sample

            # 【关键】取联合有效：两边都有效才保留
            joint_valid = sar_valid & opt_valid

            sar_vals.append(sar_sample[joint_valid])
            opt_vals.append(opt_sample[joint_valid])

        sar_vals = np.concatenate(sar_vals).astype(np.float64)
        opt_vals = np.concatenate(opt_vals).astype(np.float64)

        log_info(f"    联合有效样本: {len(sar_vals)}")

        if len(sar_vals) < 100:
            raise ValueError("联合有效样本不足，无法做直方图匹配")

        # ============================================================
        # 步骤 2: 计算 CDF 映射
        # ============================================================
        log_info(f"\n  2.计算 CDF 映射 (bins={bins})...")

        sar_sorted = np.sort(sar_vals)
        opt_sorted = np.sort(opt_vals)

        sar_min, sar_max = sar_sorted[0], sar_sorted[-1]
        opt_min, opt_max = opt_sorted[0], opt_sorted[-1]

        log_info(f"    SAR 值范围: [{sar_min:.4f}, {sar_max:.4f}]")
        log_info(f"    光学值范围: [{opt_min:.4f}, {opt_max:.4f}]")

        # 用分位映射
        sar_ds = sar_sorted[::max(1, len(sar_sorted) // 100000)]
        opt_ds = opt_sorted[::max(1, len(opt_sorted) // 100000)]

        sar_quantile_axis = np.linspace(0, 1, len(sar_ds))
        opt_quantile_axis = np.linspace(0, 1, len(opt_ds))

        def sar_value_to_optical_value(v):
            v = np.asarray(v, dtype=np.float64)
            q = np.interp(v, sar_ds, sar_quantile_axis)
            return np.interp(q, opt_quantile_axis, opt_ds)


        # =========直方图匹配评估（复用已经采样好的样本，不额外读大图）=========
        log_info("\n直方图匹配 定量评估（采样样本）:")
        # sar_vals：原始SAR采样样本；sar_value_to_optical_value映射得到匹配后的样本
        sar_matched_sample = sar_value_to_optical_value(sar_vals)

        # KS检验：匹配后SAR样本 VS 光学样本
        stat_D, p_value = ks_2samp(sar_matched_sample, opt_vals)
        log_info(f"KS D统计量: {stat_D:.4f}, p值: {p_value:.4e}")

        # 分位数对比
        quantiles = [0.02, 0.5, 0.98]
        sar_m_q = np.quantile(sar_matched_sample, quantiles)
        opt_q = np.quantile(opt_vals, quantiles)
        log_info(f"匹配后SAR分位数(2%,50%,98%): {sar_m_q.round(3)}")
        log_info(f"光学参考分位数(2%,50%,98%): {opt_q.round(3)}")

        # ============================================================
        # 步骤 3: 应用映射，分块写出
        # ============================================================
        log_info(f"\n  3.应用映射，分块写出...")

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

        with rasterio.open(out_tif, 'w', **profile) as dst:
            n_chunks = (n_rows + chunk_rows - 1) // chunk_rows

            for ci, r0 in enumerate(range(0, n_rows, chunk_rows)):
                r1 = min(r0 + chunk_rows, n_rows)
                win = Window(0, r0, n_cols, r1 - r0)

                sar_chunk = sar_src.read(1, window=win)

                # 基础nodata过滤
                valid_mask = np.isfinite(sar_chunk)
                if sar_nodata is not None:
                    valid_mask &= (sar_chunk != sar_nodata)

                # ============ 修复：叠加全局共同掩模 ============
                if full_mask_bool is not None:
                    chunk_mask = full_mask_bool[r0:r1, :]
                    valid_mask &= chunk_mask
                # ================================================

                matched = np.full_like(sar_chunk, sar_nodata, dtype=np.float32)
                if valid_mask.any():
                    vals = sar_chunk[valid_mask].astype(np.float64)
                    matched[valid_mask] = sar_value_to_optical_value(vals).astype(np.float32)

                dst.write(matched, 1, window=win)

                pct = (ci + 1) / n_chunks * 100
                log_info(f"      [{ci+1}/{n_chunks}] rows {r0}-{r1} ({pct:.1f}%)")

    log_info(f"\n  ✓ 直方图匹配完成: {out_tif}")
    return out_tif


def logarithmic_stretch(sar_tif, optical_tif, out_tif,
                        sar_mask_tif=None,
                        opt_mask_tif=None,
                        amplitude_mode=False,
                        low_pct=2.0,
                        high_pct=98.0,
                        chunk_rows=2048):
    """
    对数变换 + 分位数线性拉伸（替代全局直方图匹配）

    核心流程：
        1. 对数变换: v_log = K * log10(v + eps)，压缩动态范围
        2. 分位数统计: 计算 [low_pct, high_pct] 分位数作为拉伸上下界
        3. 线性拉伸: 映射到 [0, 1]

    参数:
        sar_tif:        SAR 图像路径（已对齐到光学网格）
        optical_tif:    光学图路径（仅用于读取尺寸，不参与映射）
        out_tif:        输出路径
        sar_mask_tif:   共同掩模（1=有效，0=无效）
        opt_mask_tif:   光学掩膜（可选，通常为None，common_mask已包含）
        amplitude_mode: True=幅度图(20*log10)，False=强度图(10*log10)
        low_pct:        低分位数（默认2.0）
        high_pct:       高分位数（默认98.0）
        chunk_rows:     分块处理行数

    返回:
        out_tif
    """
    # ---- 打开两图 ----
    with rasterio.open(sar_tif) as sar_src, \
         rasterio.open(optical_tif) as opt_src:

        assert sar_src.height == opt_src.height, "高度不匹配"
        assert sar_src.width == opt_src.width, "宽度不匹配"

        n_rows, n_cols = sar_src.height, sar_src.width
        sar_nodata = sar_src.nodata

        K = 20.0 if amplitude_mode else 10.0

        log_info(f"  SAR 尺寸: {n_rows} × {n_cols}")
        log_info(f"  Log 系数 K = {K} ({'幅度图' if amplitude_mode else '强度图'})")
        log_info(f"  分位数: [{low_pct}%, {high_pct}%]")

        # ============================================================
        # 加载掩膜
        # ============================================================
        log_info(f"\n  [0/3] 加载掩膜...")

        sar_mask_bool = None
        if sar_mask_tif is not None and os.path.exists(sar_mask_tif):
            with rasterio.open(sar_mask_tif) as src:
                m = src.read(1)
                uniq = np.unique(m)
                if 1 in uniq:
                    sar_mask_bool = (m == 1)
                elif 255 in uniq:
                    sar_mask_bool = (m == 255)
                else:
                    sar_mask_bool = (m > 0)
            log_info(f"    SAR/共同掩膜: {sar_mask_bool.sum()}/{sar_mask_bool.size} "
                     f"({sar_mask_bool.sum()/sar_mask_bool.size*100:.2f}%)")
        else:
            log_info(f"    SAR 掩膜: 未提供（只用 NODATA 过滤）")

        # 光学掩膜（可选）
        opt_mask_bool = None
        if opt_mask_tif is not None and os.path.exists(opt_mask_tif):
            with rasterio.open(opt_mask_tif) as src:
                m = src.read(1)
                uniq = np.unique(m)
                if 1 in uniq:
                    opt_mask_bool = (m == 1)
                elif 255 in uniq:
                    opt_mask_bool = (m == 255)
                else:
                    opt_mask_bool = (m > 0)
            log_info(f"    光学掩膜: {opt_mask_bool.sum()}/{opt_mask_bool.size} "
                     f"({opt_mask_bool.sum()/opt_mask_bool.size*100:.2f}%)")

        # ============================================================
        # 步骤 1: 分块统计对数变换后的分位数
        # ============================================================
        log_info(f"\n  [1/3] 统计对数变换后的分位数...")

        target_samples = 5_000_000
        total_pixels = n_rows * n_cols
        sample_stride = max(1, int(np.sqrt(total_pixels / target_samples)))
        log_info(f"    采样步长: {sample_stride}")

        log_vals_list = []

        for r0 in range(0, n_rows, chunk_rows):
            r1 = min(r0 + chunk_rows, n_rows)
            win = Window(0, r0, n_cols, r1 - r0)
            sar_chunk = sar_src.read(1, window=win).astype(np.float64)

            # 有效性判断（正值、非Nodata、掩膜内）
            valid = np.isfinite(sar_chunk)
            if sar_nodata is not None:
                valid &= (sar_chunk != sar_nodata)
            valid &= (sar_chunk > 0)  # 对数变换要求正值

            if sar_mask_bool is not None:
                valid &= sar_mask_bool[r0:r1, :]

            if valid.sum() == 0:
                continue

            # 采样
            sar_s = sar_chunk[::sample_stride, ::sample_stride]
            valid_s = valid[::sample_stride, ::sample_stride]
            if valid_s.sum() == 0:
                continue

            vals = sar_s[valid_s]
            # 对数变换
            log_vals = K * np.log10(vals + 1e-10)
            log_vals_list.append(log_vals)

        if len(log_vals_list) == 0:
            raise ValueError("无有效样本")

        all_log_vals = np.concatenate(log_vals_list)
        log_info(f"    有效样本: {len(all_log_vals)}")

        if len(all_log_vals) < 100:
            raise ValueError("有效样本不足（< 100）")

        # 计算分位数
        low = float(np.percentile(all_log_vals, low_pct))
        high = float(np.percentile(all_log_vals, high_pct))

        log_info(f"    对数变换后范围: [{all_log_vals.min():.2f}, {all_log_vals.max():.2f}] dB")
        log_info(f"    分位数 [{low_pct}%, {high_pct}%]: [{low:.2f}, {high:.2f}] dB")

        if high - low < 1e-6:
            raise ValueError("分位数范围过小，无法拉伸")

        # ============================================================
        # 步骤 2: 分块应用对数拉伸并写出
        # ============================================================
        log_info(f"\n  [2/3] 分块应用对数拉伸...")

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

        with rasterio.open(out_tif, 'w', **profile) as dst:
            n_chunks = (n_rows + chunk_rows - 1) // chunk_rows

            for ci, r0 in enumerate(range(0, n_rows, chunk_rows)):
                r1 = min(r0 + chunk_rows, n_rows)
                win = Window(0, r0, n_cols, r1 - r0)
                sar_chunk = sar_src.read(1, window=win).astype(np.float64)

                # 有效性判断
                valid = np.isfinite(sar_chunk)
                if sar_nodata is not None:
                    valid &= (sar_chunk != sar_nodata)
                valid &= (sar_chunk > 0)

                if sar_mask_bool is not None:
                    valid &= sar_mask_bool[r0:r1, :]

                # 初始化输出
                result = np.full(sar_chunk.shape, sar_nodata, dtype=np.float32)

                if valid.any():
                    vals = sar_chunk[valid]
                    log_vals = K * np.log10(vals + 1e-10)
                    stretched = (log_vals - low) / (high - low)
                    stretched = np.clip(stretched, 0.0, 1.0)
                    result[valid] = stretched.astype(np.float32)

                dst.write(result, 1, window=win)

                pct = (ci + 1) / n_chunks * 100
                log_info(f"      [{ci+1}/{n_chunks}] rows {r0}-{r1} ({pct:.1f}%)")

        # ============================================================
        # 步骤 3: 统计最终结果
        # ============================================================
        log_info(f"\n  [3/3] 输出统计...")

        # 重新分块读取，统计最终值的分布
        total_valid = 0
        sum_val = 0.0
        sat_count = 0  # 饱和（=1.0）的像素数

        for r0 in range(0, n_rows, chunk_rows):
            r1 = min(r0 + chunk_rows, n_rows)
            win = Window(0, r0, n_cols, r1 - r0)
            out_chunk = dst = None
            # 重新读
            with rasterio.open(out_tif) as src:
                out_chunk = src.read(1, window=win)

            v = out_chunk[np.isfinite(out_chunk) & (out_chunk != sar_nodata)]
            if v.size > 0:
                total_valid += v.size
                sum_val += v.sum()
                sat_count += int((v >= 0.9999).sum())

        log_info(f"    有效像素: {total_valid}")
        log_info(f"    均值: {sum_val / max(total_valid, 1):.4f}")
        log_info(f"    饱和(≈1.0)像素: {sat_count} "
                 f"({sat_count / max(total_valid,1) * 100:.4f}%)")

    log_info(f"\n  ✓ 对数拉伸完成: {out_tif}")
    return out_tif

if __name__ == "__main__":
    # 路径：注意加 r"" 原始字符串，防止 \ 转义报错
    sar_tif = r"D:\Project_Pycharm\Study\Synthesize\output\sar_aligned.tif"
    optical_tif = r"D:\Project_Pycharm\Study\Synthesize\output\m6_optical_final.tif"
    out_tif = r"D:\Project_Pycharm\Study\Synthesize\output\sar_logarithmic.tif"

    # ===== 使用【预先算好的共同掩模】=====
    common_mask = r"D:\Project_Pycharm\Study\Synthesize\output\common_mask.tif"

    logarithmic_stretch(
        sar_tif=sar_tif,
        optical_tif=optical_tif,
        out_tif=out_tif,
        sar_mask_tif=common_mask,
        opt_mask_tif=None,
        amplitude_mode=False,  # 若SAR是Sigma0（强度）用False；若是幅度图用True
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
    if common_mask is not None and os.path.exists(common_mask):
        with rasterio.open(common_mask) as msk_src:
            mask_arr = (msk_src.read(1) == 1)
    else:
        mask_arr = None

    quality_metrics = evaluate_image_quality(matched_arr, mask_arr, nodata=-9999.0)

    for k, v in quality_metrics.items():
        if isinstance(v, float):
            print(f"    {k:30s} = {v:.4f}")
        else:
            print(f"    {k:30s} = {v}")


    print("\n全部完成！")
    print(f"输出直方图匹配SAR影像：{out_tif}")





