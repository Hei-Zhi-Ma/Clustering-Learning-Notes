# dbscan.py
# 说明：用 make_moons 生成非球形数据并添加噪声，演示 DBSCAN 的噪声识别与任意形状簇能力。

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# ---------------------------
# 1. 生成数据（“月牙”形，适合 DBSCAN）
# ---------------------------
# make_moons 生成两条半月型分布，noise 控制数据扰动（噪声）
X, _ = make_moons(n_samples=300, noise=0.07, random_state=42)

# 向数据中额外添加孤立噪声点（可选，演示 DBSCAN 如何识别噪声）
np.random.seed(0)
outliers = np.random.uniform(low=-1.5, high=6.5, size=(10,2))  # 随机散点
data = np.vstack([X, outliers])

# ---------------------------
# 2. 建模：DBSCAN
# ---------------------------
# eps: 邻域半径（关键参数）
# min_samples: 邻域内点的最少数量（含自身）判定为核心点
# 调参建议：先通过 min_samples = 2*dim 或 >=4，然后用 k-distance 图找 eps（略）
model = DBSCAN(eps=0.22, min_samples=5)
labels = model.fit_predict(data)   # fit_predict 返回每点标签（噪声为 -1）

print("DBSCAN 标签分布（unique）：", np.unique(labels, return_counts=True))

# ---------------------------
# 3. 可视化：把噪声点标出来
# ---------------------------
plt.figure(figsize=(6,5))
unique_labels = set(labels)
for lab in unique_labels:
    pts = data[labels == lab]
    if lab == -1:
        # 噪声点用 X 标记
        plt.scatter(pts[:,0], pts[:,1], marker='x', c='k', label='noise', s=40)
    else:
        plt.scatter(pts[:,0], pts[:,1], s=40, label=f'cluster {lab}', alpha=0.7)
plt.title("DBSCAN on Moon-shaped Data (with outliers)")
plt.legend()
plt.xlabel("x1"); plt.ylabel("x2")
plt.tight_layout()
plt.show()

# ---------------------------
# 4. 小提示
# ---------------------------
# - 若几乎全部为 -1（噪声），说明 eps 太小或 min_samples 太大
# - 若只得到 1 个簇，说明 eps 太大
# - 对高维数据距离失效，DBSCAN 效果下降，可用降维或谱聚类
