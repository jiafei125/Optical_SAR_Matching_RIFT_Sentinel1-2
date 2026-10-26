# -*- coding: utf-8 -*-
"""
基于相位一致性(PC)的光学‑SAR异源图像配准【通用版，支持星载/机载/无人机SAR】
修复：图像归一化到0~1、Log‑Gabor滤波器中心频率修正、噪声阈值生效
更新：描述子基于MIM，获得旋转不变性，RIFT流程
新增适配点：
    1. sar_data_type: 区分 linear_intensity / linear_amplitude / db （与m8模块对齐）
    2. 斑点噪声平滑开关 speckle_smooth，机载SAR开启抑制斑点
    3. 全部关键超参向外暴露，不再硬编码
    4. dB域SAR做值域偏移保护，防止负数输入到Log‑Gabor
    5. 增加中间结果nan/inf清理，鲁棒性增强
输出：PC可视化图、Mmax/Mmin/MIM图、匹配效果图、registration.json
"""
import cv2
import numpy as np
import os
from scipy.fft import fft2, ifft2, ifftshift
from PIL import Image
import rasterio
import json
from common import log_info


def percentile_norm_rs(img, nodata=-9999, low_pct=0.5, high_pct=99.5):
    """
    遥感图像百分位归一化
    作用：遥感影像专用的分位数拉伸，裁剪超出分位数的像素，拉伸到 0~1，无效区域保持0，剔除 Nodata 极值、亮异常值
    """
    valid_mask = img != nodata
    valid_data = img[valid_mask]
    if len(valid_data) == 0:
        return np.zeros_like(img, dtype=np.float32)

    low = np.percentile(valid_data, low_pct)
    high = np.percentile(valid_data, high_pct)

    img_clipped = np.clip(img, low, high)
    out = np.zeros_like(img, dtype=np.float32)
    out[valid_mask] = (img_clipped[valid_mask] - low) / (high - low + 1e-8)
    return out


def load_and_preprocess(image_path,
                        mask_path=None,
                        max_size=512,
                        nodata=-9999,
                        do_resize=True,
                        is_sar=False,
                        sar_data_type="linear_intensity",
                        speckle_smooth=False):
    """
    图像预处理【通用适配星载/机载SAR】
    :param image_path: 影像路径
    :param mask_path: 共同掩模路径
    :param max_size: 缩放最大边长，可配置
    :param nodata: 无效值
    :param do_resize: 是否执行缩放
    :param is_sar: True=SAR图像，False=光学图像
    :param sar_data_type: 仅is_sar=True生效：linear_intensity / linear_amplitude / db
    :param speckle_smooth: SAR是否做斑点预平滑（机载建议True；星载默认False）
    :return: img_norm, mask, scale, h_ori, w_ori
    """
    with rasterio.open(image_path) as src:
        img = src.read(1)

    h_ori, w_ori = img.shape
    img = np.where(img == nodata, np.nan, img)

    mask = None
    if mask_path and os.path.exists(mask_path):
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.uint8)
        if mask.shape != img.shape:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8) * 255
        if np.sum(mask) == 0:
            log_info(f"掩膜全黑，忽略mask {os.path.basename(mask_path)}")
            mask = None
    if mask is None:
        mask = (~np.isnan(img)).astype(np.uint8) * 255

    valid_pixels = img[mask == 255]
    mean_val = np.nanmean(valid_pixels) if len(valid_pixels) > 0 else 0
    img_filled = np.nan_to_num(img, nan=mean_val)

    # ========= SAR 分支处理，适配dB域（dB存在负数，移位保证非负） =========
    if is_sar:
        if sar_data_type == "db":
            min_valid = np.nanmin(img_filled[mask == 255])
            shift = max(0.0, -min_valid + 1e-6)
            img_filled = img_filled + shift
            log_info(f"[预处理SAR-dB] 执行值域移位 +{shift:.3f}，保证输入非负")
        # 机载SAR：斑点预平滑
        if speckle_smooth:
            img_filled = cv2.GaussianBlur(img_filled, (5, 5), sigmaX=1.2)
            log_info(f"[预处理SAR] 开启斑点预平滑(Gaussian 5x5)")

    scale = 1.0
    if do_resize:
        h, w = img_filled.shape
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_filled = cv2.resize(img_filled, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, erode_kernel, iterations=1)

    img_norm = percentile_norm_rs(img_filled, nodata=nodata)
    # 清除nan inf
    img_norm = np.nan_to_num(img_norm, nan=0.0, posinf=1.0, neginf=0.0)
    return img_norm, mask, scale, h_ori, w_ori


def build_log_gabor_filters(rows, cols, nscale=4, norient=6, sigma_r=0.65):
    """Log‑Gabor滤波器构建，数学逻辑不变"""
    x = np.linspace(-0.5, 0.5, cols)
    y = np.linspace(-0.5, 0.5, rows)
    u, v = np.meshgrid(x, y)
    radius = np.sqrt(u ** 2 + v ** 2)
    radius = np.maximum(radius, 1e-10)
    theta = np.arctan2(v, u)

    min_wavelength = 3.0
    filters = []
    for s in range(nscale):
        wavelength = min_wavelength * (2 ** s)
        f0 = 1.0 / wavelength
        log_gabor = np.exp(-(np.log(radius / f0)) ** 2 / (2 * (np.log(sigma_r) ** 2)))
        log_gabor[0, 0] = 0

        for o in range(norient):
            angle = o * np.pi / norient
            dtheta = np.pi / (2 * norient)
            dt = theta - angle
            spread = np.exp(-(np.arctan2(np.sin(dt), np.cos(dt)) ** 2) / (2 * dtheta ** 2))
            filt = log_gabor * spread
            filt = ifftshift(filt)
            filters.append((filt, angle))
    return filters


def compute_pc_and_moments(image, mask=None, nscale=4, norient=6, noise_threshold=0.05):
    """PC计算与矩特征提取；参数向外暴露，适配不同SAR噪声等级"""
    rows, cols = image.shape
    img_fft = fft2(image)
    filters = build_log_gabor_filters(rows, cols, nscale, norient)

    PC_theta = np.zeros((norient, rows, cols), dtype=np.float32)
    odd_amplitudes = np.zeros((norient, rows, cols), dtype=np.float32)
    theta_o = np.zeros(norient, dtype=np.float32)

    for o in range(norient):
        sum_even_o = np.zeros_like(image)
        sum_odd_o = np.zeros_like(image)
        sum_amp_o = np.zeros_like(image)
        for s in range(nscale):
            filt, angle = filters[s * norient + o]
            resp_fft = img_fft * filt
            resp = ifft2(resp_fft)
            even = np.real(resp)
            odd = np.imag(resp)
            amp = np.sqrt(even ** 2 + odd ** 2)
            sum_even_o += even
            sum_odd_o += odd
            sum_amp_o += amp

        mean_phase = np.arctan2(sum_odd_o, sum_even_o)
        sum_phase_dev_o = np.zeros_like(image)
        for s in range(nscale):
            filt, angle = filters[s * norient + o]
            resp_fft = img_fft * filt
            resp = ifft2(resp_fft)
            even = np.real(resp)
            odd = np.imag(resp)
            amp = np.sqrt(even ** 2 + odd ** 2)
            phase = np.arctan2(odd, even)
            delta_phi = np.cos(phase - mean_phase) - np.abs(np.sin(phase - mean_phase))
            sum_phase_dev_o += amp * delta_phi

        sum_phase_dev_o = np.maximum(sum_phase_dev_o - noise_threshold, 0.0)
        PC_theta[o] = sum_phase_dev_o / (sum_amp_o + 1e-8)
        PC_theta[o] = np.clip(PC_theta[o], 0, 1)
        odd_amplitudes[o] = sum_odd_o
        theta_o[o] = angle

    pc_map = np.max(PC_theta, axis=0)

    cos_theta = np.cos(theta_o).reshape(-1, 1, 1)
    sin_theta = np.sin(theta_o).reshape(-1, 1, 1)
    a = np.sum((PC_theta * cos_theta) ** 2, axis=0)
    b = 2 * np.sum((PC_theta * cos_theta) * (PC_theta * sin_theta), axis=0)
    c = np.sum((PC_theta * sin_theta) ** 2, axis=0)
    trace = a + c
    det = a * c - (b / 2) ** 2
    det[det < 0] = 0
    sqrt_term = np.sqrt(trace ** 2 - 4 * det + 1e-8)

    Mmax = (trace + sqrt_term) / 2
    Mmin = (trace - sqrt_term) / 2

    Mmax = (Mmax - np.min(Mmax)) / (np.max(Mmax) - np.min(Mmax) + 1e-8)
    Mmin = (Mmin - np.min(Mmin)) / (np.max(Mmin) - np.min(Mmin) + 1e-8)

    MIM = np.argmax(np.abs(odd_amplitudes), axis=0).astype(np.float32)
    Ox = np.sum(odd_amplitudes * cos_theta, axis=0)
    Oy = np.sum(odd_amplitudes * sin_theta, axis=0)
    Op = np.arctan2(Oy, Ox)
    Op_deg = (np.degrees(Op) + 360) % 180

    # 清理nan inf
    pc_map = np.nan_to_num(pc_map, nan=0.0)
    Mmax = np.nan_to_num(Mmax, nan=0.0)
    Mmin = np.nan_to_num(Mmin, nan=0.0)
    MIM = np.nan_to_num(MIM, nan=0.0)
    Op_deg = np.nan_to_num(Op_deg, nan=0.0)

    if mask is not None:
        pc_map[mask == 0] = 0
        Mmax[mask == 0] = 0
        Mmin[mask == 0] = 0
        MIM[mask == 0] = 0
        Op_deg[mask == 0] = 0

    return pc_map, Mmax, Mmin, MIM, Op_deg


def detect_pc_keypoints(Mmax, Mmin, mask, thresh_ratio=0.01, nms_win=3):
    """特征点检测，参数可配置"""
    M_sum = Mmax + Mmin
    M_sum = cv2.GaussianBlur(M_sum, (3, 3), 1.0)
    M_sum_norm = cv2.normalize(M_sum, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    harris_resp = cv2.cornerHarris(M_sum_norm, blockSize=2, ksize=3, k=0.04)
    kernel = np.ones((nms_win, nms_win), np.uint8)
    local_max = cv2.dilate(harris_resp, kernel)
    peak_thresh = np.max(harris_resp) * thresh_ratio
    is_peak = (harris_resp == local_max) & (harris_resp > peak_thresh) & (mask > 0)

    border = 5
    is_peak[:border, :] = False
    is_peak[-border:, :] = False
    is_peak[:, :border] = False
    is_peak[:, -border:] = False

    y_coords, x_coords = np.where(is_peak)
    kps = []
    for y, x in zip(y_coords, x_coords):
        kp = cv2.KeyPoint(x=float(x), y=float(y), size=8, response=harris_resp[y, x])
        kps.append(kp)
    return kps


def compute_pc_descriptor(MIM, keypoints, norient=6, patch_size=48, sub_regions=6):
    """描述子生成，逻辑不变"""
    h, w = MIM.shape
    des_list = []
    valid_kps = []
    half_patch = patch_size // 2
    sub_block = patch_size // sub_regions

    for kp in keypoints:
        x = int(round(kp.pt[0]))
        y = int(round(kp.pt[1]))
        if (x - half_patch < 0) or (x + half_patch >= w) or (y - half_patch < 0) or (y + half_patch >= h):
            continue
        patch_MIM = MIM[y - half_patch:y + half_patch, x - half_patch:x + half_patch].astype(np.int32)
        hist_main, _ = np.histogram(patch_MIM, bins=norient, range=(0, norient))
        main_idx = np.argmax(hist_main)
        relative_MIM = (patch_MIM - main_idx + norient) % norient

        desc = np.zeros((sub_regions, sub_regions, norient), dtype=np.float32)
        for i in range(sub_regions):
            for j in range(sub_regions):
                sb_MIM = relative_MIM[i * sub_block:(i + 1) * sub_block, j * sub_block:(j + 1) * sub_block]
                for b in range(norient):
                    desc[i, j, b] = np.sum(sb_MIM == b)
        desc_flat = desc.reshape(-1)
        norm = np.linalg.norm(desc_flat) + 1e-8
        desc_flat /= norm
        des_list.append(desc_flat)
        valid_kps.append(kp)

    des = np.array(des_list, dtype=np.float32) if des_list else np.empty((0, norient * sub_regions * sub_regions), dtype=np.float32)
    return valid_kps, des


def match_and_fsc(kp1, des1, kp2, des2, img1, img2,
                  ratio_dh=0.75,
                  ratio_dl=0.95,
                  reproj_thresh=3.0):
    """FSC特征匹配与几何验证，逻辑不变"""
    if len(des1) == 0 or len(des2) == 0:
        log_info("描述子为空，无法匹配")
        return None, None, None, 0

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    good_strict = []
    good_loose = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if n.distance <= 0:
            continue
        ratio = m.distance / n.distance
        if ratio < ratio_dl:
            good_loose.append(m)
        if ratio < ratio_dh:
            good_strict.append(m)

    log_info(f"FSC: 严格集 Ch 数量 = {len(good_strict)}")
    log_info(f"FSC: 宽松集 C  数量 = {len(good_loose)}")

    if len(good_strict) < 3:
        log_info("FSC: 严格集 Ch 点数不足 ( < 3 )，无法拟合")
        return None, None, None, len(good_loose)
    if len(good_loose) < 4:
        log_info("FSC: 宽松集 C 点数不足 ( < 4 )")
        return None, None, None, len(good_loose)

    pts1_strict = np.float32([kp1[m.queryIdx].pt for m in good_strict]).reshape(-1, 1, 2)
    pts2_strict = np.float32([kp2[m.trainIdx].pt for m in good_strict]).reshape(-1, 1, 2)

    H_rough, _ = cv2.estimateAffinePartial2D(
        pts1_strict, pts2_strict,
        method=cv2.RANSAC,
        ransacReprojThreshold=reproj_thresh,
        maxIters=1000,
        confidence=0.99
    )
    if H_rough is None:
        log_info("FSC: 严格集拟合模型失败")
        return None, None, None, len(good_loose)

    pts1_loose = np.float32([kp1[m.queryIdx].pt for m in good_loose]).reshape(-1, 1, 2)
    pts2_loose = np.float32([kp2[m.trainIdx].pt for m in good_loose]).reshape(-1, 1, 2)
    pts1_proj = cv2.transform(pts1_loose, H_rough).reshape(-1, 2)
    pts2_flat = pts2_loose.reshape(-1, 2)
    residuals = np.linalg.norm(pts2_flat - pts1_proj, axis=1)
    inlier_mask = residuals < reproj_thresh
    n_inliers = int(inlier_mask.sum())
    log_info(f"FSC: 在宽松集上评估，内点 = {n_inliers} / {len(good_loose)}")

    if n_inliers < 4:
        log_info("FSC: 内点数量不足 ( < 4 )")
        return None, None, None, len(good_loose)

    inlier_pts1 = pts1_loose[inlier_mask].reshape(-1, 1, 2)
    inlier_pts2 = pts2_loose[inlier_mask].reshape(-1, 1, 2)
    H_refined, _ = cv2.estimateAffinePartial2D(
        inlier_pts1, inlier_pts2,
        method=cv2.LMEDS
    )
    if H_refined is not None:
        H = np.eye(3, dtype=np.float64)
        H[0:2, :] = H_refined
    else:
        H = np.eye(3, dtype=np.float64)
        H[0:2, :] = H_rough

    inlier_matches = [good_loose[i] for i in range(len(good_loose)) if inlier_mask[i]]
    inlier_ratio = len(inlier_matches) / len(good_loose)
    log_info(f"FSC: 最终内点数 = {len(inlier_matches)}")
    log_info(f"FSC: 最终内点率 = {inlier_ratio:.3f}")

    draw_matches = inlier_matches
    img1_draw = (img1 * 255).astype(np.uint8)
    img2_draw = (img2 * 255).astype(np.uint8)
    match_img = cv2.drawMatches(
        img1_draw, kp1, img2_draw, kp2, draw_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return H, inlier_matches, match_img, len(good_loose)


def save_float2png(arr, save_path):
    arr_uint8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr_uint8).save(save_path)
