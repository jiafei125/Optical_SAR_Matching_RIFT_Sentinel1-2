# CompleteProcess_SNAP.py
"""
完整流程：几何 → 辐射 → 掩膜 → 滤波 → 重采样 → 归一化
"""
import os
import numpy as np
from common import read_band

# 处理模块
from m1_geometric import Implement1
from m2_radiometric import Implement2
from m3_mask import Implement3
from m4_filter import Implement4
from m5_resample import Implement5
from m6_normalize import Implement6


original_data_path = "data/optical_v3.tif"


print("----------------------几何校正----------------------")

m1_path, m1_mask_path = Implement1(data_path = original_data_path, DO_ORTHO = False)
m1_info = read_band(m1_path)


print("----------------辐射校正（标准表观反射率）----------------")

# "L2A_REFLECTANCE"针对Sentinel2等产品，不同的产品需要选择不同的参数
m2_path = Implement2(m1_path = m1_path, mask_path = m1_mask_path, product_type="L2A_REFLECTANCE", scale_factor=10000.0)
m2_info = read_band(m2_path)
m1_mask = np.load(m1_mask_path)


print("--------------------掩膜+云检测--------------------")

# use_shadow_detect:是否开启阴影检测；use_otsu：是否开启自适应阈值
m3_tif, m3_npy, valid3 = Implement3(m2_path = m2_path, m1_mask_path = m1_mask_path, cloud_threshold=0.4, use_shadow_detect=False, use_otsu = True)


print("----------------------滤波-----------------------")

m4_path, m4_eroded = Implement4(m2_path = m2_path, m3_mask_path = m3_npy)
m2_info = read_band(m2_path)
m4_info = read_band(m4_path)
valid_eroded = np.load(m4_eroded)


print("------------------重采样到工作网格------------------")

# 如果有参考SAR影像，传路径给ref_path
m5_path, m5_mask_path = Implement5(m4_path = m4_path, m3_mask_npy = m3_npy, m3_mask_tif = m3_tif, ref_path=None,  grid_path="output/work_grid.json")
m5_info = read_band(m5_path)
# 用自身作为参考校验（验证对齐结果一致性）


print("---------------------归一化--------------------")

m6_path = Implement6(m5_path = m5_path, m5_mask_path = m5_mask_path)
m5_info = read_band(m5_path)
m6_info = read_band(m6_path)
m5_mask = np.load(m5_mask_path)

print(f"最终输出: {m6_path}")


