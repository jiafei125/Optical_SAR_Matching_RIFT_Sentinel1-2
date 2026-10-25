# -*- coding: utf-8 -*-
"""
基于相位一致性(PC)的光学-SAR异源图像配准
修复：图像归一化到0~1、Log-Gabor滤波器中心频率修正、噪声阈值生效
更新：描述子基于MIM，并且获得旋转不变性，RIFT流程
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

# BASE_DIR = r"D:\Project_Pycharm\Study\MatchingAlgorithm\data"
# OPTICAL_PATH = os.path.join(BASE_DIR, 'm6_optical_final.tif')
# SAR_PATH = os.path.join(BASE_DIR, 'sar_histogram.tif')
# MASK_PATH = os.path.join(BASE_DIR, 'common_mask.tif')
#
# OUT_DIR = r"D:\Project_Pycharm\Study\Synthesize\output"


# 遥感图像百分位归一化
"""
    作用：遥感影像专用的分位数拉伸，裁剪超出分位数的像素，拉伸到 0~255，无效区域保持0，剔除 Nodata 极值、亮异常值
"""
def percentile_norm_rs(img, nodata=-9999, low_pct=0.5, high_pct=99.5):
    # 找出有效data
    valid_mask = img != nodata
    valid_data = img[valid_mask]
    # 如果整幅图都无效，则整幅图全置0
    if len(valid_data) == 0:
        return np.zeros_like(img, dtype=np.uint8)

    # 计算有效数据的0.5%分位数和99.5%分位数
    low = np.percentile(valid_data, low_pct)
    high = np.percentile(valid_data, high_pct)


    img_clipped = np.clip(img, low, high)   # 将像素值截取到[low， high]之间
    out = np.zeros_like(img, dtype=np.float32)  #创建一个和原图 img 尺寸完全一样、但数据类型被强制指定为 32 位浮点数（float32）、且所有像素值都初始化为 0 的空白矩阵
    out[valid_mask] = (img_clipped[valid_mask] - low) / (high - low + 1e-8) # 对有效像素执行线性映射：(原值 - low) / (high - low) 将值域映射到 [0, 1]

    return out


# 图像预处理
"""
    处理无效值：将 -9999 替换为 np.nan，防止NoData极值干扰归一化
    掩膜处理：读取 common_mask.tif。如果掩膜全黑，会自动回退到“非NaN区域”作为掩膜。对掩膜进行3×3腐蚀，防止边缘无效点被提取
    填充：计算有效区域均值，用均值填充NaN，防止FFT截断效应（这是PC计算的关键）。
    降采样与归一化：将大图缩放到 max_size=512，并将图像归一化到 0~1 浮点数
"""
def load_and_preprocess(image_path, mask_path=None, max_size=512, nodata=-9999, do_resize=True):
    # 读取 TIF 的第一个波段
    with rasterio.open(image_path) as src:
        img = src.read(1)

    h_ori, w_ori = img.shape    # 保存原始高宽 h_ori, w_ori 以便后续坐标还原
    img = np.where(img == nodata, np.nan, img)  # 将无效值 -9999 替换为 np.nan,让 numpy 的 nan 系列函数可以自动跳过无效像素做统计

    mask = None
    # 如果mask_path是非空字符串，则判定为 True
    # os.path.exists(路径)：判断这个路径对应的文件/文件夹在电脑上是否真实存在
    if mask_path and os.path.exists(mask_path):
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.uint8)
        if mask.shape != img.shape: # 把掩膜缩放到 和 img 完全一样的宽高
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8) * 255    # 把掩膜转成标准二值掩膜：有效像素 = 255，无效像素 = 0
        if np.sum(mask) == 0:
            log_info(f"掩膜全黑，忽略mask {os.path.basename(mask_path)}")
            mask = None
    #如果生成的common_mask.tif全黑，程序不会报错，而是自动回退，用“非 NaN 区域”生成一个临时掩膜
    if mask is None:
        mask = (~np.isnan(img)).astype(np.uint8) * 255

    # 计算有效像素的均值，用该均值填充 NaN（黑边），防 FFT 截断效应
    valid_pixels = img[mask == 255]
    mean_val = np.nanmean(valid_pixels) if len(valid_pixels) > 0 else 0
    img_filled = np.nan_to_num(img, nan=mean_val)

    # 如果图像过大，按比例缩放至 max_size
    scale = 1.0
    if do_resize:
        h, w = img_filled.shape
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_filled = cv2.resize(img_filled, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # 掩膜腐蚀，将有效区域的边界向内收缩，剔除掩模边界这些“假特征点”
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, erode_kernel, iterations=1)

    # 调用百分位截断归一化
    img_norm = percentile_norm_rs(img_filled, nodata=nodata)

    return img_norm, mask, scale, h_ori, w_ori


# Log-Gabor滤波器构建
"""
    修正中心频率：将空间波长转换为归一化中心频率 f0 = 1.0 / wavelength
    多尺度多方向：nscale=4，norient=6，构建了24个频域滤波器
    方向扩散：使用角度差的高斯函数控制方向带宽，确保滤波器对特定方向敏感
    sigma_r 每个中心频率附近，它能覆盖多大一段频率范围，sigma_r越大，单个滤波器覆盖频率范围越宽，较少的中心频率就可以覆盖整个频域
"""
def build_log_gabor_filters(rows, cols, nscale=4, norient=6, sigma_r=0.65):
    # 频域坐标网格构建，生成归一化的频域坐标
    # radius 决定滤波器“看多细（频率）”，theta 决定“看哪个方向”
    x = np.linspace(-0.5, 0.5, cols)
    y = np.linspace(-0.5, 0.5, rows)
    u, v = np.meshgrid(x, y)
    radius = np.sqrt(u ** 2 + v ** 2)
    radius = np.maximum(radius, 1e-10)
    theta = np.arctan2(v, u)

    # 多尺度径向滤波器（频率选择项）
    min_wavelength = 3.0    # 空间波长：一个完整明暗纹理周期占多少个像素，代表这个滤波器敏感的纹理周期，波长越长，滤波器捕捉越大尺度粗纹理
    filters = []
    for s in range(nscale):
        wavelength = min_wavelength * (2 ** s)  # 对应4个尺度（由细到粗）
        f0 = 1.0 / wavelength   # 将“空间像素波长”转化为“归一化频域中心频率”
        log_gabor = np.exp(-(np.log(radius / f0)) ** 2 / (2 * (np.log(sigma_r) ** 2)))  # Log-Gabor函数
        log_gabor[0, 0] = 0  # 去除直流分量

        # 多方向角度滤波器（角度选择项）
        for o in range(norient):
            angle = o * np.pi / norient # norient=6，生成 6 个方向：0°, 30°, 60°, 90°, 120°, 150°
            dtheta = np.pi / (2 * norient)  # dtheta 控制方向带宽
            dt = theta - angle  # dt 计算角度差
            spread = np.exp(-(np.arctan2(np.sin(dt), np.cos(dt)) ** 2) / (2 * dtheta ** 2)) # 将角度差强制归一化到 [-pi, pi]，解决了角度周期性跳变问题

            # 滤波器合成与输出
            filt = log_gabor * spread   # 将径向滤波器和方向滤波器相乘，得到最终的二维频域滤波器
            filt = ifftshift(filt)
            filters.append((filt, angle))   # filt 是一个复数滤波器，angle 被保留，方便后续计算相位一致性方向图

    return filters


# PC计算与矩特征提取
"""
    频域滤波：对图像做FFT，分别与24个滤波器相乘，逆变换回空域，得到每个方向的偶对称响应（even）和奇对称响应（odd）。
    PC计算：累加偶、奇响应求局部能量，减去 noise_threshold=0.05 后除以幅度和，得到该方向的PC值 PC_theta
    矩计算：修复了核心逻辑，使用每个方向的 PC_theta 计算协方差矩阵，提取出：最大矩Mmax（边缘响应强），最小矩Mmin（角点响应强）
    计算MIM与Op_deg,MIM：最大幅值索引图（记录每个像素在哪个方向奇对称响应最大;Op_deg：相位一致性方向图（限制在0~180度，避免梯度反转）
    掩膜遮盖：将无效区域的PC、Mmax、Mmin、MIM、Op全部置0。
"""
def compute_pc_and_moments(image, mask=None, nscale=4, norient=6, noise_threshold=0.05):
    # 读取图像行列，做2D傅里叶变换，转到频域做 Log-Gabor 滤波
    rows, cols = image.shape
    img_fft = fft2(image)
    filters = build_log_gabor_filters(rows, cols, nscale, norient)

    # 建立后续需要使用的参数并初始化为0
    PC_theta = np.zeros((norient, rows, cols), dtype=np.float32)    # PC_theta[o, r, c]：第o个方向上，每个像素的该方向相位一致性
    odd_amplitudes = np.zeros((norient, rows, cols), dtype=np.float32)  # odd_amplitudes[o]：方向o，所有尺度累加的奇对称响应（用来算主方向Op）
    theta_o = np.zeros(norient, dtype=np.float32)   # theta_o[o]：保存6个滤波器的主方向弧度

    # 外层：遍历方向o；内层遍历尺度s
    # 每次将图像的 FFT 频谱与滤波器相乘，逆变换回空域，分离出偶对称响应（even）和奇对称响应（odd）
    for o in range(norient):
        sum_even_o = np.zeros_like(image)
        sum_odd_o = np.zeros_like(image)
        sum_amp_o = np.zeros_like(image)

        # 第一轮循环：累加同一方向所有尺度的even、odd、amp
        for s in range(nscale):
            # 滤波器存储顺序：s*norient + o
            # 尺度0：方向0,1,2,3,4,5；尺度1：方向0,1,2,3,4,5...
            filt, angle = filters[s * norient + o]

            # 图像在频域与滤波器相乘并转回空域
            resp_fft = img_fft * filt
            resp = ifft2(resp_fft)

            even = np.real(resp)    # 取偶响应（实部）
            odd = np.imag(resp) # 取奇响应（虚部）
            amp = np.sqrt(even ** 2 + odd ** 2) # 取幅度

            sum_even_o += even  # 同一方向，全部尺度偶分量之和
            sum_odd_o += odd    # 同一方向，全部尺度奇分量之和
            sum_amp_o += amp    # 同一方向，全部尺度幅值之和

        # 计算该方向所有尺度的平均相位
        mean_phase = np.arctan2(sum_odd_o, sum_even_o)

        # 第二轮循环：重新遍历尺度，计算相位偏差
        sum_phase_dev_o = np.zeros_like(image)
        for s in range(nscale):
            filt, angle = filters[s * norient + o]
            resp_fft = img_fft * filt
            resp = ifft2(resp_fft)

            even = np.real(resp)
            odd = np.imag(resp)
            amp = np.sqrt(even ** 2 + odd ** 2)
            phase = np.arctan2(odd, even)   # 当前尺度的相位

            # 计算相位偏差 delta_phi = cos(phi - mean_phi) - |sin(phi - mean_phi)|
            delta_phi = np.cos(phase - mean_phase) - np.abs(np.sin(phase - mean_phase))

            # 累加 A * delta_phi
            sum_phase_dev_o += amp * delta_phi

        # 应用噪声阈值 T (Kovesi 公式中的 [ ... - T ])
        sum_phase_dev_o = np.maximum(sum_phase_dev_o - noise_threshold, 0.0)

        PC_theta[o] = sum_phase_dev_o / (sum_amp_o + 1e-8)
        PC_theta[o] = np.clip(PC_theta[o], 0, 1)    # 强制把当前方向 o 的 PC 图全部钳位在 0～1 之间，清除浮点计算带来的非法数值

        # 后面用来计算主方向Ox=A*cos（angle）, Oy=A*sin（angle）
        odd_amplitudes[o] = sum_odd_o #在方向o上的奇对称滤波器卷积结果
        theta_o[o] = angle  # 方向o的角度

    # 计算PC图，取6个方向PC的最大值
    # RIFT 的分方向方案（不是Kovesi原版PC：所有方向全部求和，输出单张 PC）
    pc_map = np.max(PC_theta, axis=0)

    # 基于每个方向的PC值计算矩
    # 用矩阵迹 (trace)、行列式 (det) 的标准特征值公式来写
    cos_theta = np.cos(theta_o).reshape(-1, 1, 1)
    sin_theta = np.sin(theta_o).reshape(-1, 1, 1)
    a = np.sum((PC_theta * cos_theta) ** 2, axis=0)
    b = 2 * np.sum((PC_theta * cos_theta) * (PC_theta * sin_theta), axis=0)
    c = np.sum((PC_theta * sin_theta) ** 2, axis=0)
    trace = a + c
    det = a * c - (b / 2) ** 2
    det[det < 0] = 0
    sqrt_term = np.sqrt(trace ** 2 - 4 * det + 1e-8)

    # Mmax 为最大特征值表征边缘强度，Mmin 为最小特征值，作为 PC 角点检测的响应图
    Mmax = (trace + sqrt_term) / 2
    Mmin = (trace - sqrt_term) / 2

    # 归一化，统一阈值，方便角点提取
    Mmax = (Mmax - np.min(Mmax)) / (np.max(Mmax) - np.min(Mmax) + 1e-8)
    Mmin = (Mmin - np.min(Mmin)) / (np.max(Mmin) - np.min(Mmin) + 1e-8)


    # 构造MIM：取奇响应绝对值最大的方向索引
    MIM = np.argmax(np.abs(odd_amplitudes), axis=0).astype(np.float32)

    # Op：像素主方向，RIFT描述子需要用到的角度
    Ox = np.sum(odd_amplitudes * cos_theta, axis=0)
    Oy = np.sum(odd_amplitudes * sin_theta, axis=0)
    Op = np.arctan2(Oy, Ox)
    Op_deg = (np.degrees(Op) + 360) % 180

    # 掩膜：无效区域置0（云掩膜/NODATA区域）
    if mask is not None:
        pc_map[mask == 0] = 0
        Mmax[mask == 0] = 0
        Mmin[mask == 0] = 0
        MIM[mask == 0] = 0
        Op_deg[mask == 0] = 0

    return pc_map, Mmax, Mmin, MIM, Op_deg


# 特征点检测
"""
    叠加矩图：M_sum = Mmax + Mmin，兼顾边缘点和角点
    Harris角点检测：在叠加图上运行 cv2.cornerHarris，提取响应极值点
    NMS与边界剔除：使用膨胀实现非极大值抑制，并剔除图像边缘border个像素内的点
    nms_win=3：非极大抑制窗口
"""
def detect_pc_keypoints(Mmax, Mmin, mask, thresh_ratio=0.01, nms_win=3):
    # 将最大矩和最小矩叠加，用于检测关键点
    M_sum = Mmax + Mmin
    M_sum = cv2.GaussianBlur(M_sum, (3, 3), 1.0)    # 高斯平滑，平滑 M_sum 图，抑制孤立噪声尖峰，减少伪角点
    # 将叠加图归一化到 0~1，消除 Mmax/Mmin 值域浮动，保证后面cornerHarris的阈值稳定
    M_sum_norm = cv2.normalize(M_sum, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    # 运行 Harris 角点检测
    # blockSize=2 表示计算 Harris 矩阵时使用的局部窗口较小，这对微小角点敏感
    # ksize：Sobel 梯度算子窗口
    # 输出harris_resp为Harris 角点响应图，每个像素的值代表该位置是角点的可能性
    harris_resp = cv2.cornerHarris(M_sum_norm, blockSize=2, ksize=3, k=0.04)

    # 非极大值抑制（NMS）与阈值过滤
    # 窗口大小由nms_win控制，代表nms_win * nms_win邻域内只保留一个角点
    kernel = np.ones((nms_win, nms_win), np.uint8)
    local_max = cv2.dilate(harris_resp, kernel) # cv2.dilate 膨胀操作找到每个像素邻域内的局部最大值
    peak_thresh = np.max(harris_resp) * thresh_ratio    # 是基于全局最大 Harris 响应的百分比阈值
    is_peak = (harris_resp == local_max) & (harris_resp > peak_thresh) & (mask > 0) # 同时满足三个条件：是局部最大值、大于全局阈值、且在有效掩模内则判定为关键点

    # 边界剔除，剔除图像边缘border个像素范围内的角点
    border = 5
    is_peak[:border, :] = False
    is_peak[-border:, :] = False
    is_peak[:, :border] = False
    is_peak[:, -border:] = False

    # 把二值峰值图is_peak的坐标转为 OpenCV 标准KeyPoint对象，供后续匹配（BFMatcher 等）使用
    # size=8：关键点邻域直径（特征描述子提取时的窗口大小）
    y_coords, x_coords = np.where(is_peak)
    kps = []
    for y, x in zip(y_coords, x_coords):
        kp = cv2.KeyPoint(x=float(x), y=float(y), size=8, response=harris_resp[y, x])
        kps.append(kp)

    return kps


# 描述子生成
"""
    基于 MIM（最大索引图），为每一个 Harris 角点，提取PC 局部描述子（RIFT 的描述子核心），实现旋转不变
    patch_size=48：以关键点为中心，截取邻域窗口大小 48×48
    sub_regions=6：patch 在 x、y 方向切分的子块数量
    归一化：对128维描述子进行L2归一化
"""
def compute_pc_descriptor(MIM, keypoints, norient=6, patch_size=48, sub_regions=6):

    h, w = MIM.shape # 为后续边界判断做准备
    des_list = []
    valid_kps = []
    half_patch = patch_size // 2    # half_patch=24，关键点上下左右各取24像素，窗口48×48
    sub_block = patch_size // sub_regions   # 48 / 6 = 8，每个子块大小8×8像素

    # 遍历所有 Harris 检测到的角点，检查其邻域是否越界
    for kp in keypoints:
        x = int(round(kp.pt[0]))
        y = int(round(kp.pt[1]))
        # 边界判断：如果窗口超出图像范围，直接丢弃该关键点
        if (x - half_patch < 0) or (x + half_patch >= w) or (y - half_patch < 0) or (y + half_patch >= h):
            continue

        # 提取邻域MIM
        patch_MIM = MIM[y - half_patch:y + half_patch, x - half_patch:x + half_patch].astype(np.int32)

        # 1. 计算主方向 (patch 内 MIM 方向统计，生成方向直方图)
        hist_main, _ = np.histogram(patch_MIM, bins=norient, range=(0, norient))
        main_idx = np.argmax(hist_main)  # 直方图峰值对应的方向索引，作为这个关键点局部邻域的主方向，主方向索引 (0~norient-1)

        # 2. 方向归一化：将MIM值相对于主方向旋转，实现旋转不变
        relative_MIM = (patch_MIM - main_idx + norient) % norient

        # 3. 构建描述子：sub_regions x sub_regions 个子块，每个子块统计norient维直方图
        # 将 48×48 的 patch 均匀切分成`6×6`个子块，每个子块大小 8×8
        # 对每一个子块，统计 6 个方向索引各自的像素数量，得到 6 维直方图
        desc = np.zeros((sub_regions, sub_regions, norient), dtype=np.float32)

        for i in range(sub_regions):
            for j in range(sub_regions):
                sb_MIM = relative_MIM[i * sub_block:(i + 1) * sub_block,
                j * sub_block:(j + 1) * sub_block]
                # 统计该子块内每个方向索引的出现次数
                for b in range(norient):
                    desc[i, j, b] = np.sum(sb_MIM == b)

        desc_flat = desc.reshape(-1)    # (sub_regions,sub_regions,orient) → sub_regions * sub_regions * orient维向量

        # L2归一化：消除光照 / 对比度线性变化带来的影响
        norm = np.linalg.norm(desc_flat) + 1e-8
        desc_flat /= norm

        des_list.append(desc_flat)
        valid_kps.append(kp)

    des = np.array(des_list, dtype=np.float32) if des_list else np.empty((0, norient * sub_regions * sub_regions), dtype=np.float32)

    # 返回过滤后的关键点列表，以及N×216的描述子矩阵
    return valid_kps, des


# 特征匹配与FSC
"""
    KNN匹配：bf.knnMatch 得到最近邻和次近邻。
    Lowe比率测试：m.distance < 0.85 * n.distance，剔除模糊匹配。
    RANSAC：cv2.estimateAffine2D 估计仿射变换（6参数），重投影误差阈值为ransacReprojThreshold
"""
def match_and_fsc(kp1, des1, kp2, des2, img1, img2,
                  ratio_dh=0.75,  # 严格阈值，构建高质量样本集 Ch
                  ratio_dl=0.95,  # 宽松阈值，构建候选一致集 C
                  reproj_thresh=3.0):  # 重投影误差阈值
    """
    FSC (Fast Sample Consensus) 特征匹配与几何验证。

    核心思路：
      - 双重 Lowe 阈值分层
      - 从严格集 Ch 采样拟合模型
      - 在宽松集 C 上评估内点
      - 用所有内点重新拟合最终模型
    """
    # 前置判断
    if len(des1) == 0 or len(des2) == 0:
        log_info("描述子为空，无法匹配")
        return None, None, None, 0

    # KNN 匹配
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    # 双阈值分层
    good_strict = []  # 严格集 Ch
    good_loose = []  # 宽松集 C

    for pair in raw_matches:
        # 处理 knnMatch 可能返回不足 2 个邻居的情况
        if len(pair) < 2:
            continue
        m, n = pair
        if n.distance <= 0:
            continue
        ratio = m.distance / n.distance

        # 宽松阈值: 加入一致集 C
        if ratio < ratio_dl:
            good_loose.append(m)
        # 严格阈值: 加入样本集 Ch
        if ratio < ratio_dh:
            good_strict.append(m)

    log_info(f"FSC: 严格集 Ch 数量 = {len(good_strict)}")
    log_info(f"FSC: 宽松集 C  数量 = {len(good_loose)}")

    # 安全检查
    # 相似变换至少需要 2 对点，但为了稳健用 3 对
    if len(good_strict) < 3:
        log_info("FSC: 严格集 Ch 点数不足 ( < 3 )，无法拟合")
        return None, None, None, len(good_loose)
    if len(good_loose) < 4:
        log_info("FSC: 宽松集 C 点数不足 ( < 4 )")
        return None, None, None, len(good_loose)

    # 从严格集 Ch 采样拟合模型
    pts1_strict = np.float32(
        [kp1[m.queryIdx].pt for m in good_strict]
    ).reshape(-1, 1, 2)
    pts2_strict = np.float32(
        [kp2[m.trainIdx].pt for m in good_strict]
    ).reshape(-1, 1, 2)

    # estimateAffinePartial2D 内部是 RANSAC，从 pts1_strict 中采样
    # 相当于实现了 FSC 的"从高质量子集采样"
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

    # 在宽松集 C 上评估内点
    pts1_loose = np.float32(
        [kp1[m.queryIdx].pt for m in good_loose]
    ).reshape(-1, 1, 2)
    pts2_loose = np.float32(
        [kp2[m.trainIdx].pt for m in good_loose]
    ).reshape(-1, 1, 2)

    # 用 H_rough 把 loose 点投影到目标图，计算重投影残差
    pts1_proj = cv2.transform(pts1_loose, H_rough).reshape(-1, 2)
    pts2_flat = pts2_loose.reshape(-1, 2)
    residuals = np.linalg.norm(pts2_flat - pts1_proj, axis=1)

    inlier_mask = residuals < reproj_thresh
    n_inliers = int(inlier_mask.sum())

    log_info(f"FSC: 在宽松集上评估，内点 = {n_inliers} / {len(good_loose)}")

    if n_inliers < 4:
        log_info("FSC: 内点数量不足 ( < 4 )")
        return None, None, None, len(good_loose)

    # 用所有内点重新拟合
    inlier_pts1 = pts1_loose[inlier_mask].reshape(-1, 1, 2)
    inlier_pts2 = pts2_loose[inlier_mask].reshape(-1, 1, 2)

    # 用所有内点做最小二乘精化（不带 RANSAC）
    H_refined, _ = cv2.estimateAffinePartial2D(
        inlier_pts1, inlier_pts2,
        method=cv2.LMEDS  # LMEDS 对残差更稳健
    )

    if H_refined is not None:
        H = np.eye(3, dtype=np.float64)
        H[0:2, :] = H_refined
    else:
        # 回退到 rough 模型
        H = np.eye(3, dtype=np.float64)
        H[0:2, :] = H_rough

    # 提取最终内点匹配
    inlier_matches = [good_loose[i]
                      for i in range(len(good_loose)) if inlier_mask[i]]
    inlier_ratio = len(inlier_matches) / len(good_loose)
    log_info(f"FSC: 最终内点数 = {len(inlier_matches)}")
    log_info(f"FSC: 最终内点率 = {inlier_ratio:.3f}")

    # 绘制匹配结果
    draw_matches = inlier_matches
    img1_draw = (img1 * 255).astype(np.uint8)
    img2_draw = (img2 * 255).astype(np.uint8)
    match_img = cv2.drawMatches(
        img1_draw, kp1, img2_draw, kp2, draw_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return H, inlier_matches, match_img, len(good_loose)

# 把`0~1`范围的浮点 numpy 数组，保存为 8 位灰度 PNG 图片,用来保存生成的PC图
def save_float2png(arr, save_path):
    arr_uint8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr_uint8).save(save_path)
