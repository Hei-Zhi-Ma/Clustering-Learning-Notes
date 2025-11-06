# meanshift.py
# 说明：生成多峰数据（不同密度峰），演示 MeanShift 自动寻找峰值与聚类中心

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

# ---------------------------
# 1. 生成数据（多峰，密度不同）
# ---------------------------
np.random.seed(42)
c1 = np.random.randn(80,2) * 0.4 + np.array([0,0])   # 密度高
c2 = np.random.randn(40,2) * 0.6 + np.array([5,5])   # 密度中等
c3 = np.random.randn(60,2) * 0.5 + np.array([0,5])   # 密度偏高但簇更分散
data = np.vstack((c1, c2, c3))

# ---------------------------
# 2. 估计 bandwidth（可选）
# ---------------------------
# estimate_bandwidth 可以给出一个建议的 bandwidth 值（基于 quantile）
# quantile 越小 -> bandwidth 越小 -> 聚类更细
bandwidth = estimate_bandwidth(data, quantile=0.2)
print("Estimated bandwidth:", bandwidth)

# ---------------------------
# 3. 建模：MeanShift
# ---------------------------
# 如果不传 bandwidth，MeanShift 也会尝试估计（但显式传入更可控）
model = MeanShift(bandwidth=bandwidth, bin_seeding=True)  # bin_seeding 能加速
labels = model.fit_predict(data)
centers = model.cluster_centers_
print("MeanShift found centers:\n", centers)
print("Labels counts:", np.unique(labels, return_counts=True))

# ---------------------------
# 4. 可视化：数据点 + 中心
# ---------------------------
plt.figure(figsize=(6,5))
plt.scatter(data[:,0], data[:,1], c=labels, s=40, cmap='tab10', alpha=0.8)
plt.scatter(centers[:,0], centers[:,1], marker='x', s=200, c='red', edgecolor='k', label='centers')
plt.title("MeanShift on Multi-density Peaks")
plt.legend()
plt.xlabel("x1"); plt.ylabel("x2")
plt.tight_layout()
plt.show()

# ---------------------------
# 5. 小提示
# ---------------------------
# - bandwidth 非常关键：小导致簇细分，多噪声；大导致簇合并
# - 对大规模数据 MeanShift 计算量大，可考虑采样或其他密度方法
