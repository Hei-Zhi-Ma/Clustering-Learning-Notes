# hierarchical.py
# 说明：生成不同密度数据，演示层次聚类并显示树状图（dendrogram 可视化层次结构）

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch  # 用于画 dendrogram（需要 scipy）

# ---------------------------
# 1. 生成数据（密集簇 + 稀疏簇）
# ---------------------------
np.random.seed(42)
dense = np.random.randn(80, 2) * 0.3 + np.array([0, 0])   # 密集簇
sparse = np.random.randn(40, 2) * 1.2 + np.array([4, 4])  # 稀疏簇
data = np.vstack((dense, sparse))

# ---------------------------
# 2. 绘制树状图（dendrogram） —— 帮助理解层次结构
# ---------------------------
# linkage 方法 'ward' 与 AgglomerativeClustering 的 linkage 对应
plt.figure(figsize=(8,4))
# sch.linkage 返回聚合的层次信息矩阵，用于绘制 dendrogram
Z = sch.linkage(data, method='ward')
sch.dendrogram(Z, truncate_mode='level', p=5)  # truncate_mode 可控制显示深度
plt.title("Dendrogram (Ward linkage)")
plt.xlabel("Sample index or (cluster size)")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

# ---------------------------
# 3. 建模：AgglomerativeClustering
# ---------------------------
# n_clusters: 最终划分簇数（可以通过剪树的高度来决定）
model = AgglomerativeClustering(n_clusters=2, linkage="ward")
labels = model.fit_predict(data)

print("Hierarchical labels counts:", np.unique(labels, return_counts=True))

# ---------------------------
# 4. 可视化最终划分
# ---------------------------
plt.figure(figsize=(6,5))
plt.scatter(data[:,0], data[:,1], c=labels, s=40)
plt.title("Agglomerative Clustering Result (n_clusters=2)")
plt.xlabel("x1"); plt.ylabel("x2")
plt.tight_layout()
plt.show()

# ---------------------------
# 5. 小提示
# ---------------------------
# - 层次聚类时间复杂度高，数据量大时可能非常慢
# - dendrogram 有助于选择合理的簇数（剪树）
# - linkage 选择会显著影响结果（single 会有链效应）
