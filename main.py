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
# sar对齐到光学网络，同时将光学掩模对齐
align_to_optical(
    src_tif=sar_tif,
    optical_tif=opt_tif,
    out_tif=sar_aligned,
    sar_mask_tif=sar_mask,
    out_mask_tif=sar_mask_aligned,
    resample_method='bilinear',
    dst_nodata=0,
    mask_nodata=0,
    mask_valid_value=None,
)

# DEM对齐到光学网格
align_to_optical(
    src_tif=dem_tif,
    optical_tif=opt_tif,
    out_tif=dem_aligned,
    resample_method='bilinear',
    src_nodata=None,  # 自动读取；若 DEM 的 nodata 是 -32767，会自动识别
    dst_nodata=0,
)

# 斜距对齐到光学网格
align_to_optical(
    src_tif=slant_tif,
    optical_tif=opt_tif,
    out_tif=slant_aligned,
    resample_method='bilinear',
    src_nodata=None,  # 自动读取
    dst_nodata=0,
)

# 生成共同掩模
combine_common_mask(
    sar_aligned_mask_tif=sar_mask_aligned,
    optical_mask_tif=optical_mask,
    out_common_mask_tif=common_mask
)


t_model1 = time.perf_counter() - t1

# ----------2.对数变换 + 分位数拉伸----------
t2 = time.perf_counter()

log_info("*" * 70)
log_info("对数变换 + 分位数拉伸")
log_info("*" * 70)
# histogram_match(
#     sar_tif=sar_aligned,
#     optical_tif=opt_tif,
#     out_tif=sar_histogram,
#     sar_mask_tif=common_mask,
#     opt_mask_tif=None  # 共同掩模已经包含光学有效区，这里填None
# )

logarithmic_stretch(
        sar_tif=sar_aligned,
        optical_tif=opt_tif,
        out_tif=sar_logarithmic,
        sar_mask_tif=common_mask,
        opt_mask_tif=None,
        amplitude_mode=False,  # 若SAR是Sigma0（强度）用False；若是幅度图用True
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
opt_img, opt_mask, opt_scale, opt_h, opt_w = load_and_preprocess(opt_tif, common_mask, max_size=512, do_resize=True)
sar_img, sar_mask, sar_scale, sar_h, sar_w = load_and_preprocess(sar_logarithmic, common_mask, max_size=512, do_resize=True)

# 分别计算光学图和sar图的PC、Mmax、Mmin、MIM、Op
pc_optical, Mmax_opt, Mmin_opt, MIM_opt, Op_opt = compute_pc_and_moments(opt_img, opt_mask, nscale=4, norient=6, noise_threshold=0.05)
# PC mean = 整张图像所有像素 PC 的全局平均值，偏低：说明图像大部分区域是平坦区域，只有少量像素存在稳定边缘
# Mmax：结构张量最大特征值，Mmax 均值很小：强边缘像素只占很少一部分，大部分区域没有明显单向边缘
# Mmin：结构张量最小特征值，均值很小：角点像素在整张图里是非常稀疏的
# 如果 Op mean 接近 45°，说明结构分布比较均匀，或者以对角线结构为主
log_info(f"光学图：PC_mean: {np.mean(pc_optical):.4f}, Mmax_mean: {np.mean(Mmax_opt):.4f}, Mmin_mean: {np.mean(Mmin_opt):.4f}, Op_mean: {np.mean(Op_opt):.4f}")
# PC>0.1：判定为有显著相位一致性（有效边缘像素）
# 这个比例是用来调参noise_threshold、Log-Gabor 尺度方向的重要指标，如果占比过高（>0.4）：噪声被当成边缘，杂点太多；如果占比过低（<0.05）：大量真实边缘被抑制，提取特征太少
log_info(f"PC >0.1 占比: {np.sum(pc_optical > 0.1) / pc_optical.size:.4f}")

pc_sar, Mmax_sar, Mmin_sar, MIM_sar, Op_sar = compute_pc_and_moments(sar_img, sar_mask, nscale=4, norient=6, noise_threshold=0.05)
log_info(f"SAR图：PC_mean: {np.mean(pc_sar):.4f}, Mmax_mean: {np.mean(Mmax_sar):.4f}, Mmin_mean: {np.mean(Mmin_sar):.4f}, Op_mean: {np.mean(Op_sar):.4f}")
log_info(f"PC >0.1 占比: {np.sum(pc_sar > 0.1) / pc_sar.size:.4f}")

# 提取角点
kp_opt = detect_pc_keypoints(Mmax_opt, Mmin_opt, opt_mask, thresh_ratio=0.005, nms_win=3)
log_info(f"光学图：Harris检测到PC角点数量：{len(kp_opt)}")
kp_sar = detect_pc_keypoints(Mmax_sar, Mmin_sar, sar_mask, thresh_ratio=0.005, nms_win=3)
log_info(f"SAR图：Harris检测到PC角点数量：{len(kp_sar)}")

# 生成描述子
kp_opt, des_opt = compute_pc_descriptor(MIM_opt, kp_opt, norient=6, patch_size=32, sub_regions=4)
kp_sar, des_sar = compute_pc_descriptor(MIM_sar, kp_sar, norient=6, patch_size=32, sub_regions=4)

# 特征匹配+FSC
# ratio_dh:严格集质量,越小越纯但可能数量不足
# ratio_dl:一致集规模,越大包含越多但噪声也多
H, inlier_matches, match_img, len_good = match_and_fsc(kp_sar, des_sar, kp_opt, des_opt, sar_img, opt_img,  ratio_dh=0.75, ratio_dl=0.9,  reproj_thresh=3.0)  # SAR 图像坐标系 → 光学图像坐标系 的相似变换

log_info(f"内点的数量为{len(inlier_matches)}")
if H is not None and len(inlier_matches) >= 15:
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
        crop={"size": int(max(opt_img.shape[:2]))}
    )

    out_reg_json = os.path.join(OUT_DIR, "registration.json")
    with open(out_reg_json, "w", encoding="utf-8") as f:
        json.dump(reg_result, f, indent=2, ensure_ascii=False)

    log_info(f"降采样图平均像素偏移 dx={reg_result['pixel_offset_avg_ds']['dx']:.3f}, dy={reg_result['pixel_offset_avg_ds']['dy']:.3f}")
    log_info(f"RMSE = {reg_result['rmse_pixel_ds']:.3f} 像素")
    log_info(f"配准结果已写入{out_reg_json}")
else:
    log_info("配准失败，RANSAC内点不足，无法保证仿射矩阵稳定！")


t_model3 = time.perf_counter() - t3

# ----------4.定位----------
t4 = time.perf_counter()

log_info("*" * 70)
log_info("开始定位")
log_info("*" * 70)
# 读取 DEM 和斜距
products = read_aligned_products(dem_aligned, slant_aligned)
transform = products["transform"]
crs = products["crs"]
shape = products["shape"]
dem_arr = products["dem"]
slant_arr = products["slant"]
# 斜距单位归一化
slant_arr = normalize_slant(slant_arr)

# 读取匹配点
pairs = load_match_pairs(reg_json, shape)
if len(pairs) < 4:
    raise ValueError(f"匹配点不足：只有 {len(pairs)} 个，至少需要 4 个")

# 提取 (X, Y, Z, R)
X, Y, Z, R = extract_ground_points(pairs, transform, dem_arr, slant_arr)

# 过滤无效值
valid = np.isfinite(Z) & np.isfinite(R)
if not valid.all():
    log_info(f"   剔除无效点: {(~valid).sum()} 个")
    X, Y, Z, R = X[valid], Y[valid], Z[valid], R[valid]
    if len(X) < 4:
        raise ValueError("有效点少于4个")

# 平面投影坐标系UTM → 经纬度 → 地心坐标系ECEF
# 提取 EPSG 编号
crs_str = str(crs)
epsg = crs_str.split(":")[-1] if "EPSG:" in crs_str else "32651"
log_info(f"\n   CRS: {crs_str}")
# 坐标转换器，强制输入顺序是(东向X,北向Y)，输出顺序(lon, lat)（经度在前，纬度在后）
transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326",
                                        always_xy=True)

points_ecef = np.zeros((len(X), 3))
scene_lats, scene_lons = [], []
# 循环逐点转换
for i, (xi, yi, zi) in enumerate(zip(X, Y, Z)):
    lon, lat = transformer.transform(xi, yi)
    points_ecef[i] = geodetic_to_ecef(lat, lon, zi)
    scene_lats.append(lat); scene_lons.append(lon)

# 求所有有效点经纬度平均值，得到场景中心。
# 后续用来生成卫星 ECEF 初值`init_ecef`
scene_lat = float(np.mean(scene_lats))
scene_lon = float(np.mean(scene_lons))
log_info(f"   场景中心: lat={scene_lat:.4f}°, lon={scene_lon:.4f}°")


# 惯导初值
slant_median = float(np.median(R))  # 取斜距中位数
ins_h = SENTINEL1_ORBIT_H if slant_median > 10e3 else 8000.0
# 假设卫星就在场景中心点的正上方，把这个假想的 “卫星经纬度 + 高度” 转为 ECEF 地心坐标，作为 LM 的初始猜测init_ecef
init_ecef = geodetic_to_ecef(scene_lat, scene_lon, ins_h)
log_info(f"   惯导初值: lat={scene_lat:.4f}°, lon={scene_lon:.4f}°, h={ins_h:.0f} m")


# 定位解算
platform_ecef, rmse, inlier_idx = localization(points_ecef, R, init_ecef, threshold=15.0, max_iter=3)

# 输出
lat, lon, h = ecef_to_geodetic(*platform_ecef)

log_info("平台位置:",True)
log_info(f"  经度: {lon:.6f}°", True)
log_info(f"  纬度: {lat:.6f}°", True)
log_info(f"  高度: {h:.1f} m", True)
log_info(f"  定位 RMSE: {rmse:.3f} m", True)

# 保存平台位置
result = {
    "method": "slant_range_intersection",
    "match_point_count": int(len(X)),
    "inlier_count": int(len(inlier_idx)),
    "rmse_meter": float(rmse),
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
