# kmeans.py
# 说明：演示 K-Means 在球形簇上的效果。代码注释详尽，便于学习每一步的意义。

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ---------------------------
# 1. 生成数据（3 个球形簇）
# ---------------------------
# 为了让 KMeans 发挥最佳效果，我们生成 3 个方差相近的球形簇
np.random.seed(42)  # 随机种子，保证每次生成的数据一致，便于复现
cluster1 = np.random.randn(50, 2) + np.array([0, 0])   # 中心 (0,0)
cluster2 = np.random.randn(50, 2) + np.array([5, 5])   # 中心 (5,5)
cluster3 = np.random.randn(50, 2) + np.array([0, 5])   # 中心 (0,5)

data = np.vstack((cluster1, cluster2,cluater3))       # 合并为一个数据集 (150,2)

# ---------------------------
# 2. 建模：KMeans
# ---------------------------
# n_clusters: 指定簇个数 k（KMeans 必须）
# random_state: 随机初始化种子（推荐设定以保证可复现）
# n_init: 随机初始化运行次数，较大的 n_init 更稳定（sklearn 1.4+ 默认较高）
model = KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(data)                # 训练模型（迭代直到收敛或达到 max_iter）

labels = model.labels_         # 每个样本的簇标签（0,1,2）
centers = model.cluster_centers_  # 每个簇的中心坐标 (3,2)

# 打印关键结果，便于调试与理解
print("KMeans 簇标签样例（前10）：", labels[:10])
print("KMeans 聚类中心：\n", centers)

# ---------------------------
# 3. 可视化（简单明了）
# ---------------------------
# 散点图：样本按标签着色；质心用 'X' 标出
plt.figure(figsize=(6,5))
plt.scatter(data[:,0], data[:,1], c=labels, s=40, cmap='viridis', alpha=0.8)
plt.scatter(centers[:,0], centers[:,1], marker='X', s=200, c='red', edgecolor='k', label='centers')
plt.title("K-Means on Spherical Clusters")
plt.legend()
plt.xlabel("x1"); plt.ylabel("x2")
plt.tight_layout()
plt.show()

# ---------------------------
# 4. 小提示（注释）
# ---------------------------
# - 如果看到簇不均或算法收敛不好：尝试不同的 n_init 或标准化数据（StandardScaler）
# - KMeans 假设簇为凸/球形；若簇非球形，则考虑 DBSCAN 或 MeanShift
