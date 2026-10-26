"""
完整流程演示
"""
import time
from Pre_SAR_Align_Optical import *
from Pre_SAR_HistogramMatch import *
from Registration import *
from Position import *
from common import log_info


# 输入文件
sar_tif = "data/sar_vv_v2.tif"
sar_mask = "data/sar_mask_v2.tif"
dem_tif = "data/dem_v2.tif"
slant_tif = "data/slant_range_v2.tif"
# 光学处理后输出，在该模块为输出
opt_tif = "output/m6_optical_final.tif"
optical_mask = "output/m5_valid_mask.tif"


# 输出文件
common_mask = "output/common_mask.tif"
sar_aligned = "output/sar_aligned.tif"
dem_aligned = "output/dem_aligned.tif"
slant_aligned = "output/slant_range_aligned.tif"
sar_mask_aligned = "output/sar_mask_aligned.tif"

sar_histogram = "output/sar_histogram.tif"
sar_logarithmic = "output/sar_logarithmic.tif"
Match_img = "output/match_img.png"
reg_json  = "output/registration.json"
out_json = "output/platform_position.json"
OUT_DIR = r"D:\Project_Pycharm\Study\Synthesize\output"


start_time = time.perf_counter()
# ----------1.对齐到光学网络----------
t1 = time.perf_counter()

log_info("*" * 70)
log_info("对齐到光学网络")
log_info("*" * 70)
# SAR对齐
align_to_optical(
    src_tif=sar_tif,
    optical_tif=opt_tif,
    out_tif=sar_aligned,
    sar_mask_tif=sar_mask,
    out_mask_tif=sar_mask_aligned,
    resample_method='bilinear',
    dst_nodata=NODATA,      # 影像nodata=-9999
    mask_nodata=0,          # 掩膜nodata依旧0
    mask_valid_value=None,
)

# DEM对齐
align_to_optical(
    src_tif=dem_tif,
    optical_tif=opt_tif,
    out_tif=dem_aligned,
    resample_method='bilinear',
    src_nodata=None,
    dst_nodata=NODATA,
)

# 斜距对齐
align_to_optical(
    src_tif=slant_tif,
    optical_tif=opt_tif,
    out_tif=slant_aligned,
    resample_method='bilinear',
    src_nodata=None,
    dst_nodata=NODATA,
)

# 生成共同掩模，无需改动
combine_common_mask(
    src_aligned_mask_tif=sar_mask_aligned,
    ref_mask_tif=optical_mask,
    out_common_mask_tif=common_mask
)


t_model1 = time.perf_counter() - t1

# ----------2.对数变换 + 分位数拉伸----------
t2 = time.perf_counter()

log_info("*" * 70)
log_info("对数变换 + 分位数拉伸")
log_info("*" * 70)

logarithmic_stretch(
        sar_tif=sar_aligned,
        optical_tif=opt_tif,
        out_tif=sar_logarithmic,
        sar_mask_tif=common_mask,
        opt_mask_tif=None,
        sar_data_type="linear_intensity",
        low_pct=2.0,
        high_pct=98.0,
    )

t_model2 = time.perf_counter() - t2

# ----------3.异源配准----------
t3 = time.perf_counter()

log_info("*" * 70)
log_info("配准算法")
log_info("*" * 70)

# 加载影像并进行预处理
# ------------------- 预处理 -------------------
opt_img, opt_mask, opt_scale, opt_h, opt_w = load_and_preprocess(
    image_path=opt_tif,
    mask_path=common_mask,
    max_size=512,
    do_resize=True,
    is_sar=False
)

# 注意：sar_logarithmic 是m8模块输出【已经拉伸到0‑1】，不要再开启SAR特殊处理！is_sar=False
sar_img, sar_mask, sar_scale, sar_h, sar_w = load_and_preprocess(
    image_path=sar_logarithmic,
    mask_path=common_mask,
    max_size=512,
    do_resize=True,
    is_sar=False
)

# 分别计算光学图和sar图的PC、Mmax、Mmin、MIM、Op
pc_optical, Mmax_opt, Mmin_opt, MIM_opt, Op_opt = compute_pc_and_moments(
    opt_img, opt_mask, nscale=4, norient=6, noise_threshold=0.05
)
log_info(f"光学图：PC_mean: {np.mean(pc_optical):.4f}, Mmax_mean: {np.mean(Mmax_opt):.4f}, Mmin_mean: {np.mean(Mmin_opt):.4f}, Op_mean: {np.mean(Op_opt):.4f}")
log_info(f"PC >0.1 占比: {np.sum(pc_optical > 0.1) / pc_optical.size:.4f}")

pc_sar, Mmax_sar, Mmin_sar, MIM_sar, Op_sar = compute_pc_and_moments(
    sar_img, sar_mask, nscale=4, norient=6, noise_threshold=0.05
)
log_info(f"SAR图：PC_mean: {np.mean(pc_sar):.4f}, Mmax_mean: {np.mean(Mmax_sar):.4f}, Mmin_mean: {np.mean(Mmin_sar):.4f}, Op_mean: {np.mean(Op_sar):.4f}")
log_info(f"PC >0.1 占比: {np.sum(pc_sar > 0.1) / pc_sar.size:.4f}")

# 提取角点
kp_opt = detect_pc_keypoints(Mmax_opt, Mmin_opt, opt_mask, thresh_ratio=0.005, nms_win=3)
log_info(f"光学图：Harris检测到PC角点数量：{len(kp_opt)}")

kp_sar = detect_pc_keypoints(Mmax_sar, Mmin_sar, sar_mask, thresh_ratio=0.005, nms_win=3)
log_info(f"SAR图：Harris检测到PC角点数量：{len(kp_sar)}")

# 提前拦截关键点为空
if len(kp_opt) == 0 or len(kp_sar) == 0:
    log_info("关键点数量为0，直接终止配准！")
else:
    # 生成描述子
    kp_opt, des_opt = compute_pc_descriptor(MIM_opt, kp_opt, norient=6, patch_size=32, sub_regions=4)
    kp_sar, des_sar = compute_pc_descriptor(MIM_sar, kp_sar, norient=6, patch_size=32, sub_regions=4)

    # 特征匹配+FSC
    # ratio_dh:严格集质量,越小越纯但可能数量不足
    # ratio_dl:一致集规模,越大包含越多但噪声也多
    H, inlier_matches, match_img, len_good = match_and_fsc(
        kp_sar, des_sar,
        kp_opt, des_opt,
        sar_img, opt_img,
        ratio_dh=0.75,
        ratio_dl=0.9,
        reproj_thresh=3.0
    )

    log_info(f"内点的数量为{len(inlier_matches) if inlier_matches is not None else 0}")

    min_inlier = 15
    if H is not None and inlier_matches is not None and len(inlier_matches) >= min_inlier:
        log_info("变换矩阵H（降采样图，3×3）：")
        log_info(H)

        cv2.imwrite(Match_img, match_img)

        from common import build_unified_result
        import rasterio

        with rasterio.open(opt_tif) as src:
            transform = src.transform
            crs = src.crs

        reg_result = build_unified_result(
            method="BasedOnPC",
            direction="SAR->Optical",
            kp_sar=kp_sar,
            kp_opt=kp_opt,
            inlier_matches=inlier_matches,
            good_count=len_good,
            H=H,
            sar_scale=sar_scale,
            opt_scale=opt_scale,
            transform=transform,
            crs=crs,
            sar_shape=sar_img.shape,
            opt_shape=opt_img.shape,
            crop=None   # 没有做裁剪，传None
        )

        out_reg_json = os.path.join(OUT_DIR, "registration.json")
        with open(out_reg_json, "w", encoding="utf-8") as f:
            json.dump(reg_result, f, indent=2, ensure_ascii=False)

        log_info(f"降采样图平均像素偏移 dx={reg_result['pixel_offset_avg_ds']['dx']:.3f}, dy={reg_result['pixel_offset_avg_ds']['dy']:.3f}")
        log_info(f"RMSE = {reg_result['rmse_pixel_ds']:.3f} 像素")
        log_info(f"配准结果已写入{out_reg_json}")
    else:
        log_info(f"配准失败，有效内点不足({0 if inlier_matches is None else len(inlier_matches)} / 需要≥{min_inlier})，无法保证变换矩阵稳定！")

t_model3 = time.perf_counter() - t3

# ----------4.定位----------
t4 = time.perf_counter()

log_info("*" * 70)
log_info("开始定位")
log_info("*" * 70)

# ============ 配置参数（根据SAR类型修改）============
# slant_unit："meters"斜距米 / "time_ns"纳秒往返时间
slant_unit = "meters"
# 初值模式
# auto_height: 场景中心+估计高度(仅星载S‑1调试用，机载不要用)
# external: 外部传入GPS/IMU得到的ECEF，机载/无人机必须选这个
init_mode = "auto_height"
# 当init_mode="external"，填入平台ECEF初值
init_ecef_external = np.array([0.0, 0.0, 0.0])
# 残差阈值：星载150~250；机载20~80；无人机5~30
residual_threshold = 150.0
# auto_height模式下：星载S‑1轨道高度；仅init_mode=auto_height生效
SENTINEL1_ORBIT_H = 700000.0
airborne_guess_h = 5000.0  # 机载猜测高度，auto_height模式仅做调试用
# =====================================================

products = read_aligned_products(dem_aligned, slant_aligned)
transform = products["transform"]
crs = products["crs"]
shape = products["shape"]
dem_arr = products["dem"]
slant_arr = products["slant"]

# 斜距归一化，【必须指定slant_unit】
slant_arr = normalize_slant(slant_arr, slant_unit=slant_unit)

# 读取匹配点
pairs = load_match_pairs(reg_json, shape)
if len(pairs) < 4:
    raise ValueError(f"匹配点不足：只有 {len(pairs)} 个，至少需要 4 个")

# extract_ground_points 返回：X,Y,Z,R, keep_idx
X, Y, Z, R, keep_idx = extract_ground_points(pairs, transform, dem_arr, slant_arr)

if len(X) < 4:
    raise ValueError(f"有效点少于4个，当前有效点：{len(X)}")

# --------坐标转换：UTM 直接 → ECEF(EPSG:4978)，省去中间4326一步--------
transformer = Transformer.from_crs(crs, "EPSG:4978", always_xy=True)
x_ecef, y_ecef, z_ecef = transformer.transform(X, Y, Z)
points_ecef = np.column_stack([x_ecef, y_ecef, z_ecef])

# --------构造LM最小二乘初值 init_ecef --------
if init_mode == "external":
    # 机载/无人机优先走这里，外部传入GPS/IMU的ECEF
    init_ecef = init_ecef_external.copy()
    log_info(f"使用外部提供平台ECEF初值：{init_ecef.round(2)}")
elif init_mode == "auto_height":
    # 原有逻辑：场景中心点+猜测高度，仅用于星载调试，机载不建议依赖
    # 先求场景中心经纬度
    transformer_llh = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon_scene, lat_scene = transformer_llh.transform(np.mean(X), np.mean(Y))
    slant_median = float(np.median(R))
    if slant_median > 10e3:
        guess_h = SENTINEL1_ORBIT_H
    else:
        guess_h = airborne_guess_h
    init_ecef = geodetic_to_ecef(lat_scene, lon_scene, guess_h)
    log_info(f"【自动初猜模式】场景中心 lat={lat_scene:.4f}°, lon={lon_scene:.4f}°, guess_h={guess_h:.0f} m")
    log_info(f"自动生成init_ecef = {init_ecef.round(2)}")
else:
    raise ValueError('init_mode只能选 "external" / "auto_height"')

# =========定位解算，注意返回值现在4个 =========
# 返回：platform_ecef, final_rmse, valid_idx, residuals_all
platform_ecef, rmse, inlier_idx, residuals_all = localization(
    points_ecef,
    R,
    init_ecef,
    residual_threshold=residual_threshold,
    max_iter=3
)

lat, lon, h = ecef_to_geodetic(*platform_ecef)

log_info("平台位置:")
log_info(f"  经度: {lon:.6f}°")
log_info(f"  纬度: {lat:.6f}°")
log_info(f"  高度: {h:.1f} m")
log_info(f"  定位 RMSE: {rmse:.3f} m")

# 保存平台位置
result = {
    "method": "slant_range_intersection",
    "match_point_count": int(len(X)),
    "inlier_count": int(len(inlier_idx)),
    "rmse_meter": float(rmse),
    "residual_threshold_used": residual_threshold,
    "platform": {
        "lat": float(lat),
        "lon": float(lon),
        "h": float(h),
        "ecef_x": float(platform_ecef[0]),
        "ecef_y": float(platform_ecef[1]),
        "ecef_z": float(platform_ecef[2]),
    },
}

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

log_info(f"\n定位结果已保存: {out_json}")



t_model4 = time.perf_counter() - t4

# 输出整个项目的相关指标
end_time = time.perf_counter()
total_time = end_time - start_time

log_info(f"对齐到光学网络 : {t_model1:.2f} s", True)
log_info(f"对数变形+分位数拉伸 : {t_model2:.2f} s", True)
log_info(f"异源配准 : {t_model3:.2f} s", True)
log_info(f"定位 : {t_model4:.2f} s", True)
log_info(f"\n项目总运行时间：{total_time:.4f} s", True)
